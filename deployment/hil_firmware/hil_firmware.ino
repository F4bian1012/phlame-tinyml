/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.
   Licensed under the Apache License, Version 2.0 (the "License") */
// Modified for PHLAME: this file derives from the TensorFlow Lite Micro example
// code and has been changed. A copy of the licence is in
// deployment/LICENSE-Apache-2.0.txt.

// ============================================================================
// PHLAME - HIL (camera-in-the-loop) firmware
// ----------------------------------------------------------------------------
// Firmware DEDICATED to the HIL bench: the camera of the board captures the
// physical scene and the whole system runs on the MCU. One sketch serves two
// boards (see BOARD SUPPORT below). It is INDEPENDENT of the PIL bench: the
// PIL firmware deployment/pil_firmware/pil_firmware.ino (image over serial,
// #...@ protocol) is a separate sketch.
//
// Host companion: src/hil_benchmark.py
//
// Serial commands:
//   'T'      -> capture one frame from the camera and run the inference
//   'F' '1'  -> enable frame-dump: after each inference the captured frame
//               (160x120, 8 bits) is sent back (FRAME_BEGIN/END lines +
//               #...@ packet with ESC/XOR 0x20 escaping, the same scheme as
//               the PIL bench) for SIL-on-PC cross-validation on the real
//               frame
//   'F' '0'  -> disable frame-dump   (answers "FRAME_DUMP:ON|OFF")
//
// Boot handshake (parseable lines):
//   BOARD:<name>              PORTENTA_H7 | NICLA_VISION
//   CAM_INIT:OK | CAM_INIT:FAIL   the camera is initialized FIRST, before
//                                 the memory and TFLite (see the note in
//                                 setup())
//   INPUT_SHAPE:<H>x<W>x<C>   geometry of the input tensor
//   MODEL_BYTES / MODEL_FNV1A / ARENA_USED   identity of the flashed model
//   READY_HIL                 ready to receive triggers
//
// Telemetry per inference:
//   TEMP_C:<int>              MCU temperature (outside the measured region)
//   TS_MS:<uint32>            millis() timestamp of the inference
//   <int>                     predicted class (integer ALONE on its line)
//   CYC_CAPTURE / CYC_PRE / CYC_INF / CYC_POST / CYC_TOTAL   DWT->CYCCNT cycles
//   US_CAPTURE  / US_PRE  / US_INF  / US_POST  / US_TOTAL    microseconds
//   (PRE includes the color->luminance conversion where the sensor is a color
//    one, the camera->model-geometry resize and the quantization;
//    TOTAL = CAPTURE + PRE + INF + POST = sensor->prediction latency)
//   --- Done. Waiting for the next trigger... ---   end-of-inference marker
// ============================================================================

// ============================================================================
// BOARD SUPPORT
// ----------------------------------------------------------------------------
// Everything that depends on the board is decided here: its name, its camera
// and where the tensor arena and the image buffer live.
//
//   Portenta H7 + Vision Shield    arduino:mbed_portenta:envie_m7
//       HM01B0 camera: monochrome sensor, delivers the 8-bit grayscale frame
//       directly. 8 MB of external SDRAM: the arena (4 MB) and the image
//       buffer (400 KB) are allocated there with ea_malloc().
//   Nicla Vision                   arduino:mbed_nicla:nicla_vision
//       GC2145 camera (built in): color sensor. It is read as RGB565 and
//       converted to luminance on the MCU, because the grayscale mode of its
//       driver does not deliver luminance (it falls back to raw Bayer data).
//       No external RAM: the arena, the camera frame and the image buffer are
//       static in the internal RAM, which offers 523,624 bytes for everything
//       (the flash offers 1,966,080 bytes for the firmware plus model.h). The
//       arena size decides which models fit; AllocateTensors() fails at boot
//       when it is too small, and the linker fails ("region RAM overflowed")
//       when it is too large.
//
// Both sizes can be set at build time without editing this file:
//   arduino-cli compile ... --build-property
//       "compiler.cpp.extra_flags=-DTENSOR_ARENA_KB=<n> -DIMAGE_BUFFER_BYTES=<n>"
//   TENSOR_ARENA_KB     tensor arena, in KB
//   IMAGE_BUFFER_BYTES  image buffer, in bytes (width * height * channels of
//                       the model input)
// ============================================================================

// Capture geometry of the HIL bench on both boards: 160x120, 30 fps. After
// the capture the pipeline always works on an 8-bit luminance frame of this
// size.
#define CAM_WIDTH 160
#define CAM_HEIGHT 120
#define CAM_FPS 30

#if defined(ARDUINO_PORTENTA_H7_M7)
#define BOARD_NAME "PORTENTA_H7"
#define BOARD_HAS_SDRAM 1
#define ARENA_MEMORY "SDRAM"
#include <SDRAM.h>
#define CAMERA_NAME "HM01B0"
#define CAMERA_FAIL_TEXT "ERROR: HM01B0 camera not detected. Is the Vision Shield attached? System halted."
#define CAMERA_PIXEL_FORMAT CAMERA_GRAYSCALE
#define CAMERA_DELIVERS_RGB565 0
#ifndef TENSOR_ARENA_KB
#define TENSOR_ARENA_KB 4096 // 4 MB for the larger models
#endif
#ifndef IMAGE_BUFFER_BYTES
#define IMAGE_BUFFER_BYTES (400 * 1024) // comfortably covers up to 320x320 RGB
#endif
#elif defined(ARDUINO_NICLA_VISION)
#define BOARD_NAME "NICLA_VISION"
#define BOARD_HAS_SDRAM 0
#define ARENA_MEMORY "internal RAM"
#define CAMERA_NAME "GC2145"
#define CAMERA_FAIL_TEXT "ERROR: GC2145 camera did not respond. System halted."
#define CAMERA_PIXEL_FORMAT CAMERA_RGB565
#define CAMERA_DELIVERS_RGB565 1
#ifndef TENSOR_ARENA_KB
#define TENSOR_ARENA_KB 368 // sized for MobileNetV2 160x120x1 (uses 356,404 bytes)
#endif
#ifndef IMAGE_BUFFER_BYTES
#define IMAGE_BUFFER_BYTES (160 * 120) // one 160x120 grayscale image
#endif
#else
#error "Unsupported board. Build for arduino:mbed_portenta:envie_m7 or arduino:mbed_nicla:nicla_vision."
#endif

#include "mbed.h"
#include "model.h"
#include <Chirale_TensorFlowLite.h>

// --- Camera ---
#include "camera.h"
#if defined(ARDUINO_PORTENTA_H7_M7)
#include "himax.h" // HM01B0 (Portenta Vision Shield)
#else
#include "gc2145.h" // GC2145 (built into the Nicla Vision)
#endif

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
// (Portenta) or in a static array (Nicla Vision)
constexpr int kTensorArenaSize = TENSOR_ARENA_KB * 1024;
uint8_t *tensor_arena = nullptr;

// AllOpsResolver keeps the firmware independent of the model: it avoids
// AllocateTensors() errors caused by missing operations. It costs flash,
// which both boards can afford (about 2 MB each).
tflite::AllOpsResolver resolver;

// ---- CAMERA ----
#if defined(ARDUINO_PORTENTA_H7_M7)
HM01B0 himax;
Camera cam(himax);
#else
GC2145 galaxyCore;
Camera cam(galaxyCore);
#endif
FrameBuffer fb;
bool frame_dump = false; // 'F1'/'F0': send the captured frame back to the host

// Geometry of the input tensor (read from input->dims in setup())
int model_h = 0, model_w = 0, model_c = 1;

// Working buffer: the frame resized to the model geometry
// (IMAGE_BUFFER_BYTES bytes).
uint8_t *image_buffer = nullptr;
int image_bytes_ready = 0;

#if !BOARD_HAS_SDRAM
// Static storage in the internal RAM. Being static, the linker accounts for
// it: a combination of sizes that does not fit fails at build time instead of
// at run time.
alignas(16) static uint8_t tensor_arena_storage[TENSOR_ARENA_KB * 1024];

// Camera frame, RGB565 (2 bytes per pixel). It is handed to the camera
// library with fb.setBuffer() so that the library does not take it from the
// heap; the library requires it aligned to the 32-byte cache line.
alignas(32) static uint8_t camera_frame[CAM_WIDTH * CAM_HEIGHT * 2];

// The luminance frame is computed in place over the first half of
// camera_frame, which leaves its second half (CAM_WIDTH * CAM_HEIGHT bytes)
// free: the image buffer lives there when it fits, at no RAM cost. A larger
// image buffer gets its own array.
#if IMAGE_BUFFER_BYTES > (CAM_WIDTH * CAM_HEIGHT)
static uint8_t image_buffer_storage[IMAGE_BUFFER_BYTES];
#endif
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

  // Working buffer for the resized frame, in SDRAM
  image_buffer = (uint8_t *)ea_malloc(IMAGE_BUFFER_BYTES);

  // Memory with 16 extra bytes for alignment
  uint8_t *raw_arena = (uint8_t *)ea_malloc(kTensorArenaSize + 16);

  if (raw_arena == nullptr || image_buffer == nullptr) {
    return false;
  }

  // Align to 16 bytes
  tensor_arena = (uint8_t *)(((uintptr_t)raw_arena + 15) & ~15);
#else
  tensor_arena = tensor_arena_storage;
#if IMAGE_BUFFER_BYTES > (CAM_WIDTH * CAM_HEIGHT)
  image_buffer = image_buffer_storage;
#else
  image_buffer = camera_frame + CAM_WIDTH * CAM_HEIGHT;
#endif
#endif
  return true;
}

#if CAMERA_DELIVERS_RGB565
// ---- Function: RGB565 -> 8-bit luminance, in place ----
// The camera library delivers RGB565 with the high byte first:
//   byte 0 = RRRRRGGG, byte 1 = GGGBBBBB.
// Each channel is expanded to 8 bits by bit replication and the three are
// combined with the ITU-R BT.601 weights in 16-bit fixed point,
//   Y = (19595 R + 38470 G + 7471 B + 32768) >> 16,
// which is exactly what Pillow's convert('L') computes on the host (checked
// against Pillow 12.3.0 for every RGB565 value), so "grayscale" means the same
// thing on both sides. Pixel i reads bytes 2i and 2i+1 and writes byte i, so
// a forward pass can overwrite the buffer: afterwards its first n_pixels
// bytes hold the luminance frame.
void rgb565ToLuminanceInPlace(uint8_t *buf, int n_pixels) {
  for (int i = 0; i < n_pixels; ++i) {
    const uint32_t hi = buf[2 * i];
    const uint32_t lo = buf[2 * i + 1];
    const uint32_t r5 = hi >> 3;
    const uint32_t g6 = ((hi & 0x07) << 3) | (lo >> 5);
    const uint32_t b5 = lo & 0x1F;
    const uint32_t r = (r5 << 3) | (r5 >> 2);
    const uint32_t g = (g6 << 2) | (g6 >> 4);
    const uint32_t b = (b5 << 3) | (b5 >> 2);
    buf[i] = (uint8_t)((19595u * r + 38470u * g + 7471u * b + 32768u) >> 16);
  }
}
#endif

// ---- Function: resize/crop the camera frame into the image buffer ----
// Reproduces the SAME preprocessing contract as the training pipeline and the
// PIL bench: RAW uint8 pixel [0,255], HWC row-major order. The interpolation
// is nearest-neighbor (deterministic and cheap on the M7). If the model has 3
// channels, the gray value is replicated in R=G=B, the same as converting
// 'L'->'RGB' in Pillow.
// Known difference between levels: the PIL bench resizes with LANCZOS on the
// host; nearest-neighbor is used here. When the model geometry is 160x120
// (the capture geometry) there is no resize and both paths are identical.
// This difference is part of the preprocessing contract and must be reported
// with any result obtained at another geometry.
void cameraFrameToImageBuffer(const uint8_t *frame) {
  int idx = 0;
  if (model_w == CAM_WIDTH && model_h == CAM_HEIGHT) {
    // Fast path without resize (canonical 160x120 case)
    if (model_c == 1) {
      memcpy(image_buffer, frame, CAM_WIDTH * CAM_HEIGHT);
      idx = CAM_WIDTH * CAM_HEIGHT;
    } else {
      for (int i = 0; i < CAM_WIDTH * CAM_HEIGHT; ++i) {
        image_buffer[idx++] = frame[i];
        image_buffer[idx++] = frame[i];
        image_buffer[idx++] = frame[i];
      }
    }
  } else {
    // Nearest-neighbor: map each destination pixel to the closest source one
    for (int y = 0; y < model_h; ++y) {
      int sy = (int)((uint32_t)y * CAM_HEIGHT / model_h);
      for (int x = 0; x < model_w; ++x) {
        int sx = (int)((uint32_t)x * CAM_WIDTH / model_w);
        uint8_t px = frame[sy * CAM_WIDTH + sx];
        if (model_c == 1) {
          image_buffer[idx++] = px;
        } else {
          image_buffer[idx++] = px;
          image_buffer[idx++] = px;
          image_buffer[idx++] = px;
        }
      }
    }
  }
  image_bytes_ready = idx;
}

// ---- Function: load the image into the input tensor ----
// Identical to the PIL bench: raw pixel [0,255] quantized with the ACTUAL
// parameters of the tensor (scale/zero_point from the TFLite converter).
void loadImageToInputTensor() {
  int expected_bytes = 0;

  if (input->type == kTfLiteInt8) {
    expected_bytes = input->bytes; // total bytes of the tensor (e.g. 160*120*1)

    const float in_scale = input->params.scale;
    const int32_t in_zero_point = input->params.zero_point;

    int to_copy = (image_bytes_ready < expected_bytes) ? image_bytes_ready
                                                       : expected_bytes;
    for (int i = 0; i < to_copy; ++i) {
      // This model was converted WITHOUT a Rescaling layer: it expects the
      // RAW pixel in [0,255]. If you later train with Rescaling(1./255) ->
      // [0,1]:
      //   (float)image_buffer[i] / 255.0f
      // If you use MobileNet's preprocess_input -> [-1,1]:
      //   ((float)image_buffer[i] / 127.5f) - 1.0f
      // It must MATCH the preprocessing of the representative dataset of the
      // TFLite converter (see process_images.py / the conversion).
      const float real_value = (float)image_buffer[i];
      int32_t q = (int32_t)lroundf(real_value / in_scale) + in_zero_point;
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
    int to_copy = (image_bytes_ready < expected_bytes) ? image_bytes_ready
                                                       : expected_bytes;
    for (int i = 0; i < to_copy; ++i) {
      input->data.f[i] = (float)image_buffer[i] / 255.0f;
    }
    for (int i = to_copy; i < expected_bytes; ++i) {
      input->data.f[i] = 0.0f;
    }
  }
}

// ---- Function: run the inference and report result + latencies ----
// Phases measured with DWT->CYCCNT:
//   capture    : camera acquisition -> framebuffer (passed in cyc_capture)
//   preprocess : color->luminance and resize (extra_pre_cycles) +
//                loading/quantizing into the tensor
//   inference  : interpreter->Invoke()
//   postprocess: argmax of the output
void runInference(uint32_t cyc_capture, uint32_t extra_pre_cycles) {
  // --- Phase: preprocessing (loading + quantizing the image) ---
  uint32_t t0 = dwt_cycles();
  loadImageToInputTensor();
  uint32_t t1 = dwt_cycles();

  // --- Phase: inference ---
  TfLiteStatus invoke_status = interpreter->Invoke();
  uint32_t t2 = dwt_cycles();

  if (invoke_status != kTfLiteOk) {
    Serial.println("ERROR: Invoke() failed.");
    return;
  }

  // --- Phase: postprocessing (argmax) ---
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
  uint32_t cyc_pre = (t1 - t0) + extra_pre_cycles;
  uint32_t cyc_inf = t2 - t1;
  uint32_t cyc_post = t3 - t2;
  uint32_t cyc_total = cyc_capture + cyc_pre + cyc_inf + cyc_post;

  // 0) Timestamp of the event (to pair it with the ground truth on the host)
  Serial.print("TS_MS:");
  Serial.println(millis());

  // 1) Class (an integer ALONE on its line, parseable by the host)
  Serial.println(best_class);

  // 2) Latency in core cycles (raw measurement, no conversion)
  Serial.print("CYC_CAPTURE:");
  Serial.println(cyc_capture);
  Serial.print("CYC_PRE:");
  Serial.println(cyc_pre);
  Serial.print("CYC_INF:");
  Serial.println(cyc_inf);
  Serial.print("CYC_POST:");
  Serial.println(cyc_post);
  Serial.print("CYC_TOTAL:");
  Serial.println(cyc_total);

  // 3) Latency in microseconds (using the actual SystemCoreClock of the MCU)
  Serial.print("US_CAPTURE:");
  Serial.println(cycles_to_us(cyc_capture), 3);
  Serial.print("US_PRE:");
  Serial.println(cycles_to_us(cyc_pre), 3);
  Serial.print("US_INF:");
  Serial.println(cycles_to_us(cyc_inf), 3);
  Serial.print("US_POST:");
  Serial.println(cycles_to_us(cyc_post), 3);
  Serial.print("US_TOTAL:");
  Serial.println(cycles_to_us(cyc_total), 3);
}

// ---- Thermal telemetry (outside the timed region) ----
void printMcuTemperature() {
  mbed::AnalogIn mcuADCVref(ADC_VREF);
  mbed::AnalogIn mcuADCTemp(ADC_TEMP);

  uint16_t rawVref = mcuADCVref.read_u16();
  uint16_t rawTemp = mcuADCTemp.read_u16();

  uint32_t mcuVref =
      __LL_ADC_CALC_VREFANALOG_VOLTAGE(rawVref, ADC_RESOLUTION_16B);
  int32_t mcuTemp =
      __HAL_ADC_CALC_TEMPERATURE(mcuVref, rawTemp, ADC_RESOLUTION_16B);
  Serial.print("TEMP_C:");
  Serial.println(mcuTemp);
}

// ---- Function: send the captured frame back to the host (frame-dump) ----
// Sends the captured 160x120 8-bit frame BEFORE the resize to the model
// geometry: exactly as it left the sensor on the HM01B0, and the luminance
// frame derived from the RGB565 capture on the GC2145. It uses the escape
// scheme of the PIL protocol ('#'/'@'/ESC -> ESC, byte XOR 0x20). The host
// stores it and can run the SAME .tflite on the PC (SIL) on this frame: the
// PIL->HIL gap is thus split into (a) sensor/optics/display degradation and
// (b) chip-vs-PC execution divergence.
// It is sent OUTSIDE the timed region (after the latencies are reported): it
// does not contaminate the measurement.
void sendFrameToHost(const uint8_t *buf, size_t n) {
  Serial.print("FRAME_BEGIN:");
  Serial.println((uint32_t)n);
  Serial.flush();
  Serial.write('#');
  for (size_t i = 0; i < n; ++i) {
    uint8_t b = buf[i];
    if (b == '#' || b == '@' || b == 0x1B) {
      Serial.write((uint8_t)0x1B);
      Serial.write((uint8_t)(b ^ 0x20));
    } else {
      Serial.write(b);
    }
  }
  Serial.write('@');
  Serial.println();
  Serial.println("FRAME_END");
  Serial.flush();
}

// ---- HIL cycle: capture a real frame + sensor->prediction inference ----
void runHilCapture() {
  Serial.println("\n--- HIL trigger. Capturing a frame from the " CAMERA_NAME "... ---");

  // Thermal telemetry BEFORE the measured region
  printMcuTemperature();

  // --- CAPTURE phase: camera -> framebuffer (measured with DWT->CYCCNT) ---
  uint32_t tc0 = dwt_cycles();
  int grab_status = cam.grabFrame(fb, 3000); // 3 s timeout
  uint32_t tc1 = dwt_cycles();

  if (grab_status != 0) {
    Serial.println("ERROR: grabFrame() failed (timeout or camera disconnected).");
    Serial.println("--- Done. Waiting for the next trigger... ---\n");
    Serial.flush();
    return;
  }
  uint32_t cyc_capture = tc1 - tc0;

  // --- Color -> luminance (color sensor only) and resize/crop camera ->
  // model geometry; both count as PRE ---
  uint32_t tr0 = dwt_cycles();
#if CAMERA_DELIVERS_RGB565
  rgb565ToLuminanceInPlace(fb.getBuffer(), CAM_WIDTH * CAM_HEIGHT);
#endif
  cameraFrameToImageBuffer(fb.getBuffer());
  uint32_t tr1 = dwt_cycles();

  runInference(cyc_capture, tr1 - tr0);

  Serial.println("--- Done. Waiting for the next trigger... ---\n");
  Serial.flush();

  // Frame-dump (outside the measurement): send the captured frame back to the
  // host for traceability and SIL-on-PC cross-validation.
  if (frame_dump) {
    sendFrameToHost(fb.getBuffer(), (size_t)(CAM_WIDTH * CAM_HEIGHT));
  }
}

// ---- Serial command dispatcher ('T', 'F0'/'F1') ----
void handleSerialCommands() {
  while (Serial.available() > 0) {
    uint8_t byte_in = (uint8_t)Serial.read();

    if (byte_in == 'T') {
      runHilCapture();
    } else if (byte_in == 'F') {
      // Frame-dump command: wait for the second byte ('0' or '1')
      unsigned long t = millis();
      while (!Serial.available()) {
        if (millis() - t > 500)
          return; // incomplete command: discard it
      }
      uint8_t fd_byte = (uint8_t)Serial.read();
      if (fd_byte == '1') {
        frame_dump = true;
        Serial.println("FRAME_DUMP:ON");
      } else if (fd_byte == '0') {
        frame_dump = false;
        Serial.println("FRAME_DUMP:OFF");
      }
      Serial.flush();
    }
    // any other byte is ignored (noise)
  }
}

// ---------------------------------------------------------------------------
// Model fingerprint. The PC bench computes the same FNV-1a over the .tflite
// and aborts if they differ: it turns "the board was carrying another model"
// from a silent failure into a loud one. g_model[] is the .tflite byte for
// byte (tflite_to_c.py), so the fingerprints must be identical.
// Cost: one pass over flash (~1-3 ms at 400 MHz per MB). Only in setup().
// ---------------------------------------------------------------------------
static uint32_t fnv1a32(const uint8_t *p, size_t n) {
  uint32_t h = 2166136261u;
  for (size_t i = 0; i < n; i++) {
    h ^= p[i];
    h *= 16777619u;
  }
  return h;
}

static void printModelFingerprint() {
  Serial.print("MODEL_BYTES:");
  Serial.println(g_model_len);
  Serial.print("MODEL_FNV1A:");
  char hex[9];
  snprintf(hex, sizeof(hex), "%08lx", (unsigned long)fnv1a32(g_model, g_model_len));
  Serial.println(hex);
  if (interpreter != nullptr) {
    Serial.print("ARENA_USED:");
    Serial.println((unsigned long)interpreter->arena_used_bytes());
  }
  Serial.flush();
}

void setup() {
  Serial.begin(115200);
  // BOUNDED wait: an unlimited 'while (!Serial)' leaves the board hanging in
  // silence if nobody opens the port, and from the host that cannot be told
  // apart from a firmware that does not boot.
  uint32_t t_serial = millis();
  while (!Serial && (millis() - t_serial) < 5000) {
  }
  delay(2000);

  // Enable the DWT cycle counter to measure latency with 1-cycle resolution.
  // It must be done only once, here.
  dwt_enable_cycle_counter();

  // Board identity: the bench stores it with the results
  Serial.print("BOARD:");
  Serial.println(BOARD_NAME);

  // --- Initialize the camera (160x120, 30 fps) -------------------------------
  // It is initialized before the memory and TFLite so that a sensor failure
  // is reported as early as possible.
  // MIND the return value: Camera::begin() returns bool (true = success),
  // unlike the rest of the library API (grabFrame, setFrameRate...), which
  // follows the C convention of 0 = success. Comparing it with == 0 inverts
  // the logic and reports CAM_INIT:FAIL exactly when the camera starts fine.
#if !BOARD_HAS_SDRAM
  // The camera library captures into the static frame instead of the heap
  fb.setBuffer(camera_frame);
#endif
  if (cam.begin(CAMERA_R160x120, CAMERA_PIXEL_FORMAT, CAM_FPS)) {
    Serial.println("CAM_INIT:OK");
  } else {
    Serial.println("CAM_INIT:FAIL");
    Serial.println(CAMERA_FAIL_TEXT);
    Serial.flush();
    while (1)
      ;
  }

  // --- Memory: image buffer + tensor arena -----------------------------------
  // SDRAM on the Portenta, static arrays on the Nicla Vision
  if (!allocateBuffers()) {
    Serial.println("ERROR: no SDRAM available for the tensor arena or the image buffer.");
    while (1)
      ;
  }

  tflite::InitializeTarget();

  Serial.println("\n--- STARTING DIAGNOSTICS (HIL/camera) ---");
  Serial.print("Tensor arena size (bytes): ");
  Serial.println(kTensorArenaSize);
  Serial.print("Tensor arena address (" ARENA_MEMORY "): 0x");
  Serial.println((uint32_t)tensor_arena, HEX);
  Serial.print("SystemCoreClock (Hz): ");
  Serial.println(SystemCoreClock);
  Serial.flush();

  Serial.println("1. Registering operations (AllOpsResolver)...");
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

  Serial.println("5. Running AllocateTensors()...");
  Serial.flush();

  TfLiteStatus allocate_status = interpreter->AllocateTensors();

  if (allocate_status != kTfLiteOk) {
    Serial.println("Failed to allocate tensors. Arena too small?");
#if !BOARD_HAS_SDRAM
    Serial.println("This board has no external RAM: rebuild with a larger -DTENSOR_ARENA_KB (until the linker reports that RAM overflowed) or use a smaller model.");
#endif
    while (1)
      ;
  }
  Serial.println("AllocateTensors() OK.");

  input = interpreter->input(0);
  output = interpreter->output(0);

  // Geometry of the input tensor (NHWC): used by the camera->model resize
  if (input->dims->size == 4) {
    model_h = input->dims->data[1];
    model_w = input->dims->data[2];
    model_c = input->dims->data[3];
  } else if (input->dims->size == 3) {
    model_h = input->dims->data[0];
    model_w = input->dims->data[1];
    model_c = input->dims->data[2];
  }
  Serial.print("INPUT_SHAPE:");
  Serial.print(model_h);
  Serial.print("x");
  Serial.print(model_w);
  Serial.print("x");
  Serial.println(model_c);

  // The image buffer holds the frame at the model geometry, one byte per
  // input element. An input larger than the buffer would overflow it, so it
  // is refused here with a clear message.
  const long input_elements = (long)model_h * model_w * model_c;
  if (input_elements > (long)IMAGE_BUFFER_BYTES) {
    Serial.print("ERROR: the model input needs ");
    Serial.print(input_elements);
    Serial.print(" bytes and the image buffer holds ");
    Serial.print((long)IMAGE_BUFFER_BYTES);
    Serial.println(". Rebuild with a larger -DIMAGE_BUFFER_BYTES or use a smaller input.");
    while (1)
      ;
  }

  // Quantization diagnostics of the input tensor
  if (input->type == kTfLiteInt8) {
    Serial.print("Input int8 -> scale=");
    Serial.print(input->params.scale, 8);
    Serial.print(" zero_point=");
    Serial.println(input->params.zero_point);
  }

  // Identity of the loaded model: the bench compares it with the .tflite on the PC
  printModelFingerprint();

  Serial.println("Setup completed successfully.");
  Serial.println("Commands: 'T'=capture+infer  'F1'/'F0'=frame-dump on/off");
  Serial.println("READY_HIL");
  Serial.flush();
}

void loop() {
  // Non-blocking loop: continuous polling of commands from the host
  handleSerialCommands();
}
