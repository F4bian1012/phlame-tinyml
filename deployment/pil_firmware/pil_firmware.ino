/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.
   Licensed under the Apache License, Version 2.0 (the "License") */
// Modified for PHLAME: this file derives from the TensorFlow Lite Micro example
// code and has been changed. A copy of the licence is in
// deployment/LICENSE-Apache-2.0.txt.

// ============================================================================
// PHLAME - PIL (processor-in-the-loop) firmware
// ----------------------------------------------------------------------------
// The host sends a preprocessed image over serial, the MCU runs the INT8
// model on it and answers with the predicted class and the latency of each
// phase. One sketch serves two boards (see BOARD SUPPORT below).
//
// Host companion: src/pil_benchmark.py
//
// Identity lines (parseable), printed at the end of the boot lines and again
// every time the host sends an EMPTY packet (#@), which runs no inference:
//   BOARD:<name>              PORTENTA_H7 | NICLA_VISION
//   INPUT_SHAPE:<H>x<W>x<C>   geometry of the input tensor
//   MODEL_BYTES:<int>         size of g_model[] in flash
//   MODEL_FNV1A:<hex32>       FNV-1a fingerprint of g_model[]: the bench
//                             compares it with the .tflite on the PC
//   ARENA_USED:<bytes>        interpreter->arena_used_bytes()
//   ARENA_MEMORY:<where>      where the tensor arena is: SDRAM | INTERNAL
//   READY_PIL                 last line: ready to receive images
// The boot lines are printed once, when the port is opened for the first time
// after a reset, so a bench that opens the port later asks with #@. BOARD is
// also printed early in the boot lines, so that the board is named even when
// the set-up stops before the model is loaded.
//
// Answer to each image (#...@ packet with at least one byte):
//   BOARD:<name>              repeated here because the PIL bench opens the
//                             port after the boot lines are gone
//   TEMP_C:<int>              MCU temperature (outside the measured region)
//   <int>                     predicted class (integer ALONE on its line)
//   CYC_PRE / CYC_INF / CYC_POST / CYC_TOTAL   DWT->CYCCNT cycles
//   US_PRE  / US_INF  / US_POST  / US_TOTAL    microseconds
//   --- Done. Waiting for the next image... ---   end-of-inference marker
// ============================================================================

// ============================================================================
// BOARD SUPPORT
// ----------------------------------------------------------------------------
// Everything that depends on the board is decided here: its name and where
// the tensor arena and the image buffer live.
//
//   Portenta H7     arduino:mbed_portenta:envie_m7
//       8 MB of external SDRAM: the arena (4 MB) and the image buffer
//       (400 KB) are allocated there with ea_malloc().
//   Nicla Vision    arduino:mbed_nicla:nicla_vision
//       no external RAM: the arena and the image buffer are static arrays in
//       the internal RAM, which offers 523,624 bytes for everything (the
//       flash offers 1,966,080 bytes for the firmware plus model.h). The
//       arena size decides which models fit; AllocateTensors() fails at boot
//       when it is too small, and the linker fails ("region RAM overflowed")
//       when it is too large.
//
// Both sizes can be set at build time without editing this file:
//   arduino-cli compile ... --build-property
//       "compiler.cpp.extra_flags=-DTENSOR_ARENA_KB=<n> -DIMAGE_BUFFER_BYTES=<n>"
//   TENSOR_ARENA_KB     tensor arena, in KB
//   IMAGE_BUFFER_BYTES  image buffer, in bytes (width * height * channels of
//                       the largest model input to be received)
//
// A third build option moves the arena of a board that has SDRAM:
//   TENSOR_ARENA_INTERNAL_RAM=1   (--arena_memory internal of
//                       compile_upload_arduino.py) the tensor arena is a static
//                       array of TENSOR_ARENA_KB KB in the internal RAM, as on
//                       the Nicla Vision, and the image buffer stays in SDRAM.
//                       TENSOR_ARENA_KB must then be given: the 4 MB default
//                       does not fit in the internal RAM. It is there to
//                       measure what the place of the arena costs in cycles
//                       with everything else unchanged. Without it, and on a
//                       board without SDRAM, the build is the one described
//                       above.
// Where the arena ended up is announced in the ARENA_MEMORY identity line.
// ============================================================================
#ifndef TENSOR_ARENA_INTERNAL_RAM
#define TENSOR_ARENA_INTERNAL_RAM 0
#endif

#if defined(ARDUINO_PORTENTA_H7_M7)
#define BOARD_NAME "PORTENTA_H7"
#define BOARD_HAS_SDRAM 1
#include <SDRAM.h>
#if TENSOR_ARENA_INTERNAL_RAM
#define ARENA_IN_SDRAM 0
#ifndef TENSOR_ARENA_KB
#error "TENSOR_ARENA_INTERNAL_RAM needs -DTENSOR_ARENA_KB=<n>: the 4096 KB default of the SDRAM build does not fit in the internal RAM."
#endif
#else
#define ARENA_IN_SDRAM 1
#ifndef TENSOR_ARENA_KB
#define TENSOR_ARENA_KB 4096 // 4 MB for the larger models
#endif
#endif
#ifndef IMAGE_BUFFER_BYTES
#define IMAGE_BUFFER_BYTES (400 * 1024) // holds up to 320x320 RGB (307,200 bytes)
#endif
#elif defined(ARDUINO_NICLA_VISION)
#define BOARD_NAME "NICLA_VISION"
#define BOARD_HAS_SDRAM 0
#define ARENA_IN_SDRAM 0
#ifndef TENSOR_ARENA_KB
#define TENSOR_ARENA_KB 368 // sized for MobileNetV2 160x120x1 (uses 356,404 bytes)
#endif
#ifndef IMAGE_BUFFER_BYTES
#define IMAGE_BUFFER_BYTES (160 * 120) // one 160x120 grayscale image
#endif
#else
#error "Unsupported board. Build for arduino:mbed_portenta:envie_m7 or arduino:mbed_nicla:nicla_vision."
#endif

// Where the tensor arena is: ARENA_MEMORY is the word of the boot line that
// prints its address, ARENA_MEMORY_ID the value of the ARENA_MEMORY identity
// line, which the bench records.
#if ARENA_IN_SDRAM
#define ARENA_MEMORY "SDRAM"
#define ARENA_MEMORY_ID "SDRAM"
#else
#define ARENA_MEMORY "internal RAM"
#define ARENA_MEMORY_ID "INTERNAL"
#endif

#include "mbed.h"
#include "model.h"
#include <Chirale_TensorFlowLite.h>

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

// Tensor arena: reached through a POINTER so that it can live in SDRAM
// (Portenta) or in a static array (Nicla Vision, or a Portenta built with
// TENSOR_ARENA_INTERNAL_RAM)
constexpr int kTensorArenaSize = TENSOR_ARENA_KB * 1024;
uint8_t *tensor_arena = nullptr;

// AllOpsResolver keeps the firmware independent of the model: it avoids
// AllocateTensors() errors caused by missing operations. It costs flash,
// which both boards can afford (about 2 MB each).
tflite::AllOpsResolver resolver;

// ---- SERIAL PROTOCOL ----
// Start marker: '#'  (0x23)
// End marker:   '@'  (0x40)
// The image bytes are sent RAW between the markers.
// If a data byte is itself '#', '@' or ESC, the sender must escape it (see
// the Python script). Escape: 0x1B (ESC) followed by the byte XOR 0x20.
// A raw '#' in the middle of a packet restarts the packet, so the host can
// send an image again after a dropped link without corrupting it.

// An EMPTY packet (#@) is not an image: it asks for the identity lines (see
// printIdentity) and runs no inference. Because '#' restarts a packet, the
// request is understood whatever the firmware was waiting for.

// Buffer that receives the image over serial (IMAGE_BUFFER_BYTES bytes)
uint8_t *image_buffer = nullptr;
int image_bytes_received = 0;
bool receiving = false;

// Geometry of the input tensor and fingerprint of the model, worked out once
// in setup() and reported in the identity lines
int model_h = 0, model_w = 0, model_c = 0;
uint32_t model_fnv1a = 0;

// Static storage in the internal RAM. Being static, the linker accounts for
// it: a combination of sizes that does not fit fails at build time instead of
// at run time.
#if !ARENA_IN_SDRAM
alignas(16) static uint8_t tensor_arena_storage[TENSOR_ARENA_KB * 1024];
#endif
#if !BOARD_HAS_SDRAM
static uint8_t image_buffer_storage[IMAGE_BUFFER_BYTES];
#endif

// ---- LATENCY MEASUREMENT: DWT->CYCCNT cycle counter ----
// The Cortex-M7 has a Data Watchpoint & Trace (DWT) unit with a 32-bit core
// cycle counter (CYCCNT). It counts 1 per clock cycle, so the resolution is
// 1/SystemCoreClock (2.5 ns at 400 MHz, the value of SystemCoreClock measured
// on arduino:mbed_portenta; the boot lines print the actual value of the
// board), far finer than millis()/micros(). It is enabled once in setup().
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

// ---- Function: reserve the tensor arena and the image buffer ----
// Returns false when the memory is not available (only possible with SDRAM).
bool allocateBuffers() {
#if BOARD_HAS_SDRAM
  SDRAM.begin();

  // Allocate the image buffer in SDRAM
  image_buffer = (uint8_t *)ea_malloc(IMAGE_BUFFER_BYTES);

#if ARENA_IN_SDRAM
  // Memory with 16 extra bytes for alignment
  uint8_t *raw_arena = (uint8_t *)ea_malloc(kTensorArenaSize + 16);

  if (raw_arena == nullptr || image_buffer == nullptr) {
    return false;
  }

  // Align to 16 bytes
  tensor_arena = (uint8_t *)(((uintptr_t)raw_arena + 15) & ~15);
#else
  // TENSOR_ARENA_INTERNAL_RAM: only the image buffer is in SDRAM
  if (image_buffer == nullptr) {
    return false;
  }
  tensor_arena = tensor_arena_storage;
#endif
#else
  image_buffer = image_buffer_storage;
  tensor_arena = tensor_arena_storage;
#endif
  return true;
}

// ---------------------------------------------------------------------------
// Model fingerprint. The PC bench computes the same FNV-1a over the .tflite
// and stops if they differ: it turns "the board was carrying another model"
// from a silent failure into a loud one. g_model[] is the .tflite byte for
// byte (tflite_to_c.py), so the fingerprints must be identical. Same function
// as in the HIL sketch. Cost: one pass over flash (~1-3 ms at 400 MHz per
// MB), paid once in setup().
// ---------------------------------------------------------------------------
static uint32_t fnv1a32(const uint8_t *p, size_t n) {
  uint32_t h = 2166136261u;
  for (size_t i = 0; i < n; i++) {
    h ^= p[i];
    h *= 16777619u;
  }
  return h;
}

// ---- Function: say what is running, in parseable lines ----
// Printed at the end of setup() and every time the host sends an empty packet
// (#@). READY_PIL is the last line: the host reads until it sees it.
static void printIdentity() {
  Serial.print("BOARD:");
  Serial.println(BOARD_NAME);
  Serial.print("INPUT_SHAPE:");
  Serial.print(model_h);
  Serial.print("x");
  Serial.print(model_w);
  Serial.print("x");
  Serial.println(model_c);
  Serial.print("MODEL_BYTES:");
  Serial.println(g_model_len);
  Serial.print("MODEL_FNV1A:");
  char hex[9];
  snprintf(hex, sizeof(hex), "%08lx", (unsigned long)model_fnv1a);
  Serial.println(hex);
  // How much of the arena the model actually needs: the figure to size
  // TENSOR_ARENA_KB with on a board without external RAM
  Serial.print("ARENA_USED:");
  Serial.println((unsigned long)interpreter->arena_used_bytes());
  // Where the arena is (SDRAM | INTERNAL): the bench records it with the
  // latencies, so that runs with the arena in different places are told apart
  Serial.print("ARENA_MEMORY:");
  Serial.println(ARENA_MEMORY_ID);
  Serial.println("READY_PIL");
  Serial.flush();
}

// ---- Function: read a packet from the serial port ----
// Returns true when a complete packet has been received: an image, or an
// empty packet (image_bytes_received == 0), which asks for the identity lines
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
      } else if (byte_in == '#') {
        // A start marker inside a packet: the sender began again (the host
        // resends the image after a dropped serial link). A '#' that belongs
        // to the image always arrives escaped, so this can only be a new
        // start: drop what was received and take the packet from here.
        image_bytes_received = 0;
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
        if (image_bytes_received < IMAGE_BUFFER_BYTES) {
          image_buffer[image_bytes_received++] = escaped;
        }
      } else {
        // Ordinary byte
        if (image_bytes_received < IMAGE_BUFFER_BYTES) {
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
  // Tensor arena and image buffer: SDRAM on the Portenta, static arrays on
  // the Nicla Vision (see BOARD SUPPORT for TENSOR_ARENA_INTERNAL_RAM)
  bool buffers_ok = allocateBuffers();

  Serial.begin(115200);
  while (!Serial) {
  }
  delay(2000);

  // Enable the DWT cycle counter to measure latency with 1-cycle resolution.
  // It must be done only once, here.
  dwt_enable_cycle_counter();

  if (!buffers_ok) {
    Serial.println("ERROR: no SDRAM available for the tensor arena or the image buffer.");
    while (1)
      ;
  }

  tflite::InitializeTarget();

  Serial.println("\n--- STARTING DIAGNOSTICS ---");
  Serial.print("BOARD:");
  Serial.println(BOARD_NAME);
  Serial.print("Tensor arena size (bytes): ");
  Serial.println(kTensorArenaSize);
  Serial.print("Tensor arena address (" ARENA_MEMORY "): 0x");
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
#if !BOARD_HAS_SDRAM
    Serial.println("This board has no external RAM: rebuild with a larger -DTENSOR_ARENA_KB (until the linker reports that RAM overflowed) or use a smaller model.");
#elif !ARENA_IN_SDRAM
    Serial.println("This build keeps the arena in internal RAM (TENSOR_ARENA_INTERNAL_RAM): rebuild with a larger -DTENSOR_ARENA_KB (until the linker reports that RAM overflowed) or without that option.");
#endif
    while (1)
      ;
  }

  Serial.println("AllocateTensors() OK.");

  input = interpreter->input(0);
  output = interpreter->output(0);

  // Geometry of the input tensor (NHWC), announced in the identity lines: the
  // bench refuses to send images of another size
  if (input->dims->size == 4) {
    model_h = input->dims->data[1];
    model_w = input->dims->data[2];
    model_c = input->dims->data[3];
  } else if (input->dims->size == 3) {
    model_h = input->dims->data[0];
    model_w = input->dims->data[1];
    model_c = input->dims->data[2];
  }

  // The image buffer holds one byte per input element. An input larger than
  // the buffer would be truncated and the model would predict on incomplete
  // data, so it is refused here with a clear message.
  const long input_elements = (input->type == kTfLiteFloat32)
                                  ? (long)(input->bytes / 4)
                                  : (long)input->bytes;
  if (input_elements > (long)IMAGE_BUFFER_BYTES) {
    Serial.print("ERROR: the model input needs ");
    Serial.print(input_elements);
    Serial.print(" bytes and the image buffer holds ");
    Serial.print((long)IMAGE_BUFFER_BYTES);
    Serial.println(". Rebuild with a larger -DIMAGE_BUFFER_BYTES or use a smaller input.");
    while (1)
      ;
  }

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

  // Identity of the loaded model: the bench compares it with the .tflite on
  // the PC before it measures anything
  model_fnv1a = fnv1a32(g_model, g_model_len);

  Serial.println("Setup completed successfully.");
  Serial.println("Send an image as #...@ ; an empty packet (#@) prints the identity lines again.");
  printIdentity();
}

void loop() {
  // Read a packet from the serial port using the #...@ protocol
  if (readImageFromSerial()) {

    // An empty packet is the host asking what is running: no inference
    if (image_bytes_received == 0) {
      printIdentity();
      return;
    }

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

    // Board identity, repeated with every image: the PIL bench opens the port
    // after the boot lines are gone, so this is where it learns which board
    // answered. Also outside the timed region.
    Serial.print("BOARD:");
    Serial.println(BOARD_NAME);
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
