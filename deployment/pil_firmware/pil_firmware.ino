/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.
   Licensed under the Apache License, Version 2.0 (the "License") */

#include "mbed.h"
#include "model.h"
#include <Chirale_TensorFlowLite.h>
#include <SDRAM.h>

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <math.h> // lroundf

// Globals for interacting with the TensorFlow Lite Micro model
const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr;
TfLiteTensor *output = nullptr;

// Tensor arena: declared as a POINTER so that it can live in SDRAM
constexpr int kTensorArenaSize =
    4 * 1024 * 1024; // 4 MB for the larger models
uint8_t *tensor_arena = nullptr;

// AllOpsResolver is used because the Portenta H7 has plenty of flash (2 MB)
// and it avoids AllocateTensors() errors caused by missing operations
tflite::AllOpsResolver resolver;

// ---- SERIAL PROTOCOL ----
// Start marker: '#'  (0x23)
// End marker:   '@'  (0x40)
// The image bytes are sent RAW between the markers.
// If a data byte is itself '#', '@' or ESC, the sender must escape it (see
// the Python script). Escape: 0x1B (ESC) followed by the byte XOR 0x20.

// Buffer that receives the image over serial
// Raised to 400 KB to hold large images such as 320x320 RGB (307,200
// bytes)
#define IMAGE_BUFFER_SIZE (400 * 1024)

uint8_t *image_buffer = nullptr;
int image_bytes_received = 0;
bool receiving = false;

// ---- LATENCY MEASUREMENT: DWT->CYCCNT cycle counter ----
// The Cortex-M7 has a Data Watchpoint & Trace (DWT) unit with a 32-bit core
// cycle counter (CYCCNT). It counts 1 per clock cycle, so the resolution is
// 1/SystemCoreClock (2.5 ns at 400 MHz, the value of SystemCoreClock measured
// on arduino:mbed_portenta), far finer than millis()/micros(). It is enabled
// once in setup().
// At 400 MHz the 32-bit counter wraps every ~10.74 s; since each phase lasts
// far less than that, the unsigned subtraction (t_end - t_start) is correct
// even across one wrap.
static inline void dwt_enable_cycle_counter() {
  CoreDebug->DEMCR |=
      CoreDebug_DEMCR_TRCENA_Msk; // enable the trace block
#ifdef DWT_LAR_KEY
  DWT->LAR = 0xC5ACCE55; // unlock the DWT registers (some M7 cores need it)
#endif
  DWT->CYCCNT = 0;                     // reset the counter
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk; // start CYCCNT
}
static inline uint32_t dwt_cycles() { return DWT->CYCCNT; }

// Converts core cycles to microseconds using the actual clock of the MCU.
static inline float cycles_to_us(uint32_t cycles) {
  return (float)cycles * 1e6f / (float)SystemCoreClock;
}

// ---- Function: read an image from the serial port ----
// Returns true when a complete image has been received
bool readImageFromSerial() {
  while (Serial.available() > 0) {
    uint8_t byte_in = (uint8_t)Serial.read();

    if (!receiving) {
      // Waiting for the start marker '#'
      if (byte_in == '#') {
        receiving = true;
        image_bytes_received = 0;
      }
      // any other byte outside a packet is ignored
    } else {
      // Inside the packet
      if (byte_in == '@') {
        // End marker found
        receiving = false;
        return true; // complete image
      } else if (byte_in == 0x1B) {
        // Escape byte: the next byte is the actual data XOR 0x20
        // Wait for the next byte (blocking, with a short timeout)
        unsigned long t = millis();
        while (!Serial.available()) {
          if (millis() - t > 500) {
            // timeout: the packet is discarded
            receiving = false;
            return false;
          }
        }
        uint8_t escaped = (uint8_t)Serial.read() ^ 0x20;
        if (image_bytes_received < IMAGE_BUFFER_SIZE) {
          image_buffer[image_bytes_received++] = escaped;
        }
      } else {
        // Ordinary byte
        if (image_bytes_received < IMAGE_BUFFER_SIZE) {
          image_buffer[image_bytes_received++] = byte_in;
        }
      }
    }
  }
  return false; // image still incomplete
}

// ---- Function: load the image into the input tensor ----
void loadImageToInputTensor() {
  int expected_bytes = 0;

  if (input->type == kTfLiteInt8) {
    expected_bytes = input->bytes; // total bytes of the tensor (e.g. 160*120*1)

    // ACTUAL quantization parameters of the tensor (set by the TFLite
    // converter). Do NOT assume zero_point=-128 / scale=1/255: that is only
    // right for one particular model and fails with any other.
    const float in_scale = input->params.scale;
    const int32_t in_zero_point = input->params.zero_point;

    int to_copy = (image_bytes_received < expected_bytes) ? image_bytes_received
                                                          : expected_bytes;
    for (int i = 0; i < to_copy; ++i) {
      // 1) real value the model expects.
      //    This model was converted WITHOUT a Rescaling layer: it expects the
      //    RAW pixel in [0,255] (with scale~1.0, zp=-128 this reproduces the
      //    old pixel-128). If you later train with Rescaling(1./255) -> [0,1],
      //    use
      //      (float)image_buffer[i] / 255.0f
      //    If you use MobileNet's preprocess_input -> [-1,1]:
      //      ((float)image_buffer[i] / 127.5f) - 1.0f
      //    It must MATCH the preprocessing of the representative dataset of
      //    the TFLite converter (see process_images.py / the conversion).
      const float real_value = (float)image_buffer[i];
      // 2) quantize: q = round(real / scale) + zero_point
      int32_t q = (int32_t)lroundf(real_value / in_scale) + in_zero_point;
      // 3) saturate to the int8 range [-128, 127]
      if (q < -128)
        q = -128;
      if (q > 127)
        q = 127;
      input->data.int8[i] = (int8_t)q;
    }
    // Padding: use the zero_point (it represents the "real 0"), not a raw 0
    for (int i = to_copy; i < expected_bytes; ++i) {
      input->data.int8[i] = (int8_t)in_zero_point;
    }

  } else if (input->type == kTfLiteFloat32) {
    expected_bytes = input->bytes / 4; // number of floats
    int to_copy = (image_bytes_received < expected_bytes) ? image_bytes_received
                                                          : expected_bytes;
    for (int i = 0; i < to_copy; ++i) {
      // Normalize from [0,255] to [0.0, 1.0]
      input->data.f[i] = (float)image_buffer[i] / 255.0f;
    }
    for (int i = to_copy; i < expected_bytes; ++i) {
      input->data.f[i] = 0.0f;
    }
  }
}

// ---- Function: run the inference and report result + latencies ----
// Measures the latency BROKEN DOWN BY PHASE with DWT->CYCCNT:
//   - preprocess : load/quantize the image into the input tensor
//   - inference  : interpreter->Invoke()
//   - postprocess: argmax of the output
// The class (argmax) is printed as an integer on its own line to stay
// compatible with pil_benchmark.py. The latencies are emitted in lines with
// parseable prefixes (CYC_* and US_*).
void runInference() {
  // --- Phase 1: preprocessing (loading + quantizing the image) ---
  uint32_t t0 = dwt_cycles();
  loadImageToInputTensor();
  uint32_t t1 = dwt_cycles();

  // --- Phase 2: inference ---
  TfLiteStatus invoke_status = interpreter->Invoke();
  uint32_t t2 = dwt_cycles();

  if (invoke_status != kTfLiteOk) {
    Serial.println("ERROR: Invoke() failed.");
    return;
  }

  // --- Phase 3: postprocessing (argmax) ---
  int best_class = 0;
  if (output->type == kTfLiteInt8) {
    int num_classes = output->bytes;
    int8_t best_score = output->data.int8[0];
    for (int i = 0; i < num_classes; ++i) {
      if (output->data.int8[i] > best_score) {
        best_score = output->data.int8[i];
        best_class = i;
      }
    }
  } else if (output->type == kTfLiteFloat32) {
    int num_classes = output->bytes / 4;
    float best_score = output->data.f[0];
    for (int i = 0; i < num_classes; ++i) {
      if (output->data.f[i] > best_score) {
        best_score = output->data.f[i];
        best_class = i;
      }
    }
  }
  uint32_t t3 = dwt_cycles();

  // Cycles per phase (unsigned subtraction: robust to one CYCCNT wrap)
  uint32_t cyc_pre = t1 - t0;
  uint32_t cyc_inf = t2 - t1;
  uint32_t cyc_post = t3 - t2;
  uint32_t cyc_total = t3 - t0;

  // 1) Class (compatibility with the host: an integer alone on its line)
  Serial.println(best_class);

  // 2) Latency in core cycles (raw measurement, no conversion)
  Serial.print("CYC_PRE:");
  Serial.println(cyc_pre);
  Serial.print("CYC_INF:");
  Serial.println(cyc_inf);
  Serial.print("CYC_POST:");
  Serial.println(cyc_post);
  Serial.print("CYC_TOTAL:");
  Serial.println(cyc_total);

  // 3) Latency in microseconds (using the actual SystemCoreClock of the MCU)
  Serial.print("US_PRE:");
  Serial.println(cycles_to_us(cyc_pre), 3);
  Serial.print("US_INF:");
  Serial.println(cycles_to_us(cyc_inf), 3);
  Serial.print("US_POST:");
  Serial.println(cycles_to_us(cyc_post), 3);
  Serial.print("US_TOTAL:");
  Serial.println(cycles_to_us(cyc_total), 3);
}

void setup() {
  SDRAM.begin();

  // Allocate the image buffer in SDRAM
  image_buffer = (uint8_t *)ea_malloc(IMAGE_BUFFER_SIZE);

  // Memory with 16 extra bytes for alignment
  uint8_t *raw_arena = (uint8_t *)ea_malloc(kTensorArenaSize + 16);

  Serial.begin(115200);
  while (!Serial) {
  }
  delay(2000);

  // Enable the DWT cycle counter to measure latency with 1-cycle resolution.
  // It must be done only once, here.
  dwt_enable_cycle_counter();

  if (raw_arena == nullptr || image_buffer == nullptr) {
    Serial.println("ERROR: no SDRAM available for the tensor arena or the image buffer.");
    while (1)
      ;
  }

  // Align to 16 bytes
  tensor_arena = (uint8_t *)(((uintptr_t)raw_arena + 15) & ~15);

  tflite::InitializeTarget();

  Serial.println("\n--- STARTING DIAGNOSTICS ---");
  Serial.print("Tensor arena address in SDRAM: 0x");
  Serial.println((uint32_t)tensor_arena, HEX);
  Serial.print("SystemCoreClock (Hz): ");
  Serial.println(SystemCoreClock);
  Serial.flush();

  Serial.println("1. Registering operations (AllOpsResolver)...");
  Serial.flush();

  Serial.println("2. Reading the model from flash memory...");
  Serial.flush();

  model = tflite::GetModel(g_model);

  Serial.println("3. Checking the model version...");
  Serial.flush();
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Version mismatch!");
    while (1)
      ;
  }

  Serial.println("4. Creating the static interpreter...");
  Serial.flush();

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  Serial.println("5. Running AllocateTensors() [critical step]...");
  Serial.flush();

  TfLiteStatus allocate_status = interpreter->AllocateTensors();

  Serial.println("6. AllocateTensors finished.");
  Serial.flush();

  if (allocate_status != kTfLiteOk) {
    Serial.println("Failed to allocate tensors. Arena too small?");
    while (1)
      ;
  }

  Serial.println("AllocateTensors() OK.");

  input = interpreter->input(0);
  output = interpreter->output(0);

  // Quantization diagnostics: prints scale/zero_point of the input tensor.
  // If you see scale~0.00392 (=1/255) and zp=-128, the old "pixel-128"
  // shortcut was right BY CHANCE for that model; with the actual parameters
  // it works for every model.
  if (input->type == kTfLiteInt8) {
    Serial.print("Input int8 -> scale=");
    Serial.print(input->params.scale, 8);
    Serial.print(" zero_point=");
    Serial.println(input->params.zero_point);
  }

  Serial.println("Setup completed successfully.");
  Serial.print("Waiting for an image over serial (start='#', end='@') ...");
  Serial.println();
  Serial.flush();
}

void loop() {
  // Read an image from the serial port using the #...@ protocol
  if (readImageFromSerial()) {

    Serial.println("\n--- Image received. Starting inference... ---");

    // Thermal telemetry (outside the timed region): it is read BEFORE the
    // inference so that it does not contaminate the latency measurement.
    mbed::AnalogIn mcuADCVref(ADC_VREF);
    mbed::AnalogIn mcuADCTemp(ADC_TEMP);

    uint16_t rawVref = mcuADCVref.read_u16();
    uint16_t rawTemp = mcuADCTemp.read_u16();

    // Internal analog reference voltage (in mV)
    uint32_t mcuVref =
        __LL_ADC_CALC_VREFANALOG_VOLTAGE(rawVref, ADC_RESOLUTION_16B);

    // Final temperature using the factory calibration constants of the STM32H7
    int32_t mcuTemp =
        __HAL_ADC_CALC_TEMPERATURE(mcuVref, rawTemp, ADC_RESOLUTION_16B);
    Serial.print("TEMP_C:");
    Serial.println(mcuTemp);

    // NOTE: there is deliberately no delay() here. A fixed delay inside the
    // measurement cycle would contaminate any latency the host computed from
    // wall-clock time. The latency is measured by the firmware with
    // DWT->CYCCNT inside runInference() and reported in the US_*/CYC_*
    // lines.

    runInference();

    Serial.println("--- Done. Waiting for the next image... ---\n");
    Serial.flush();
  }
  // No fixed delay: the loop is non-blocking, it polls the serial port
}
