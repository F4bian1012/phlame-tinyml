/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.
   Licensed under the Apache License, Version 2.0 (the "License") */

// ============================================================================
// PHLAME - Firmware HIL (camera-in-the-loop) para Portenta H7 + Vision Shield
// ----------------------------------------------------------------------------
// Firmware DEDICADO al banco HIL real (Fase 1C): la camara HM01B0 captura la
// escena fisica y el sistema completo corre en el MCU. Es INDEPENDIENTE del
// banco PIL: el firmware canonico deployment/hil_firmware/hil_firmware.ino
// (imagen por serial, protocolo #...@) queda intacto como fallback publicable.
//
// Host companion: src/hil_camera_benchmark.py
//
// Comandos serie:
//   'T'      -> capturar un frame de la HM01B0 y ejecutar la inferencia
//   'F' '1'  -> activar frame-dump: tras cada inferencia se devuelve el frame
//               crudo del sensor (lineas FRAME_BEGIN/END + paquete #...@ con
//               escape ESC/XOR 0x20, el mismo esquema del banco PIL) para
//               validacion cruzada SIL-en-PC sobre el frame real
//   'F' '0'  -> desactivar frame-dump   (responde "FRAME_DUMP:ON|OFF")
//
// Handshake de arranque (lineas parseables):
//   INPUT_SHAPE:<H>x<W>x<C>   geometria del tensor de entrada
//   CAM_INIT:OK | CAM_INIT:FAIL
//   READY_HIL                 listo para recibir triggers
//
// Telemetria por inferencia:
//   TEMP_C:<int>              temperatura del MCU (fuera de la region medida)
//   TS_MS:<uint32>            timestamp millis() de la inferencia
//   <int>                     clase predicha (entero SOLO en su linea)
//   CYC_CAPTURE / CYC_PRE / CYC_INF / CYC_POST / CYC_TOTAL   ciclos DWT->CYCCNT
//   US_CAPTURE  / US_PRE  / US_INF  / US_POST  / US_TOTAL    microsegundos
//   (PRE incluye el resize camara->geometria del modelo y la cuantizacion;
//    TOTAL = CAPTURE + PRE + INF + POST = latencia sensor->prediccion)
// ============================================================================

#include "mbed.h"
#include "model.h"
#include <Chirale_TensorFlowLite.h>
#include <SDRAM.h>

// --- Camara HM01B0 (Portenta Vision Shield) ---
#include "camera.h"
#include "himax.h"

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

// Tensor Arena: Definido como PUNTERO para usar SDRAM
constexpr int kTensorArenaSize =
    4 * 1024 * 1024; // 4 MB para modelos mas grandes
uint8_t *tensor_arena = nullptr;

// Usamos AllOpsResolver porque la Portenta H7 tiene mucha memoria Flash (2MB)
// y esto previene errores de "AllocateTensors()" por operaciones faltantes
tflite::AllOpsResolver resolver;

// ---- CAMARA HM01B0 ----
// Resolucion nativa de captura del banco HIL: 160x120 gris, 30 fps.
#define CAM_WIDTH 160
#define CAM_HEIGHT 120
#define CAM_FPS 30

HM01B0 himax;
Camera cam(himax);
FrameBuffer fb;
bool frame_dump = false; // 'F1'/'F0': devolver el frame capturado al host

// Geometria del tensor de entrada (se lee de input->dims en setup())
int model_h = 0, model_w = 0, model_c = 1;

// Buffer de trabajo: frame redimensionado a la geometria del modelo.
// 400KB en SDRAM cubre holgado hasta 320x320 RGB.
#define IMAGE_BUFFER_SIZE (400 * 1024)
uint8_t *image_buffer = nullptr;
int image_bytes_ready = 0;

// ---- MEDICION DE LATENCIA: contador de ciclos DWT->CYCCNT ----
// El Cortex-M7 lleva una unidad Data Watchpoint & Trace (DWT) con un contador
// de ciclos de nucleo de 32 bits (CYCCNT). Cuenta 1 por ciclo de reloj, asi
// que la resolucion es 1/SystemCoreClock (~2.08 ns a 480 MHz), muchisimo mas
// fina que millis()/micros(). Se habilita una vez en setup().
// A 480 MHz el contador de 32 bits desborda cada ~8.95 s; como cada fase dura
// muy por debajo de eso, la resta unsigned (t_fin - t_ini) es correcta incluso
// con un desbordamiento.
static inline void dwt_enable_cycle_counter() {
  CoreDebug->DEMCR |=
      CoreDebug_DEMCR_TRCENA_Msk; // habilitar el bloque de trace
#ifdef DWT_LAR_KEY
  DWT->LAR = 0xC5ACCE55; // desbloquear registros DWT (algunos M7 lo requieren)
#endif
  DWT->CYCCNT = 0;                     // reiniciar contador
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk; // arrancar CYCCNT
}
static inline uint32_t dwt_cycles() { return DWT->CYCCNT; }

// Convierte ciclos de nucleo a microsegundos usando el reloj real del MCU.
static inline float cycles_to_us(uint32_t cycles) {
  return (float)cycles * 1e6f / (float)SystemCoreClock;
}

// ---- Funcion: resize/crop del frame de camara al buffer de imagen ----
// Reproduce el MISMO contrato de preprocesado que el pipeline de
// entrenamiento y el banco PIL: pixel CRUDO uint8 [0,255], orden HWC
// row-major. La interpolacion es nearest-neighbor (determinista y barata en
// el M7). Si el modelo es de 3 canales, el gris se replica en R=G=B, igual
// que convertir 'L'->'RGB' en PIL.
// NOTA para el paper: el banco PIL usa LANCZOS en el host; aqui se usa
// nearest. Si la geometria del modelo == 160x120 (nativa de la HM01B0) no
// hay resize y ambos caminos son identicos. Documentar esta diferencia como
// parte del contrato (checklist Fase 1C).
void cameraFrameToImageBuffer(const uint8_t *frame) {
  int idx = 0;
  if (model_w == CAM_WIDTH && model_h == CAM_HEIGHT) {
    // Camino rapido sin resize (caso canonico 160x120)
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
    // Nearest-neighbor: mapear cada pixel destino al origen mas cercano
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

// ---- Funcion: cargar imagen al tensor de entrada ----
// Identica al banco PIL: pixel crudo [0,255] cuantizado con los parametros
// REALES del tensor (scale/zero_point del TFLite converter).
void loadImageToInputTensor() {
  int expected_bytes = 0;

  if (input->type == kTfLiteInt8) {
    expected_bytes = input->bytes; // total bytes del tensor (e.g. 160*120*1)

    const float in_scale = input->params.scale;
    const int32_t in_zero_point = input->params.zero_point;

    int to_copy = (image_bytes_ready < expected_bytes) ? image_bytes_ready
                                                       : expected_bytes;
    for (int i = 0; i < to_copy; ++i) {
      // Este modelo fue convertido SIN capa Rescaling: espera el pixel CRUDO
      // en [0,255]. Si en el futuro entrenas con Rescaling(1./255) -> [0,1]:
      //   (float)image_buffer[i] / 255.0f
      // Si usas preprocess_input de MobileNet -> [-1,1]:
      //   ((float)image_buffer[i] / 127.5f) - 1.0f
      // Debe COINCIDIR con el preprocesado del dataset representativo del
      // TFLite converter (mira process_images.py / la conversion).
      const float real_value = (float)image_buffer[i];
      int32_t q = (int32_t)lroundf(real_value / in_scale) + in_zero_point;
      if (q < -128)
        q = -128;
      if (q > 127)
        q = 127;
      input->data.int8[i] = (int8_t)q;
    }
    // Relleno: usar el zero_point (representa el "0 real"), no un 0 crudo
    for (int i = to_copy; i < expected_bytes; ++i) {
      input->data.int8[i] = (int8_t)in_zero_point;
    }

  } else if (input->type == kTfLiteFloat32) {
    expected_bytes = input->bytes / 4; // numero de floats
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

// ---- Funcion: ejecutar inferencia y reportar resultado + latencias ----
// Fases medidas con DWT->CYCCNT:
//   capture    : adquisicion camara -> framebuffer (pasada en cyc_capture)
//   preprocess : resize (extra_pre_cycles) + carga/cuantizacion al tensor
//   inference  : interpreter->Invoke()
//   postprocess: argmax de la salida
void runInference(uint32_t cyc_capture, uint32_t extra_pre_cycles) {
  // --- Fase: preprocesado (carga + cuantizacion de la imagen) ---
  uint32_t t0 = dwt_cycles();
  loadImageToInputTensor();
  uint32_t t1 = dwt_cycles();

  // --- Fase: inferencia ---
  TfLiteStatus invoke_status = interpreter->Invoke();
  uint32_t t2 = dwt_cycles();

  if (invoke_status != kTfLiteOk) {
    Serial.println("ERROR: Invoke() fallo.");
    return;
  }

  // --- Fase: postprocesado (argmax) ---
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

  // Ciclos por fase (resta unsigned: robusta ante un desbordamiento del CYCCNT)
  uint32_t cyc_pre = (t1 - t0) + extra_pre_cycles;
  uint32_t cyc_inf = t2 - t1;
  uint32_t cyc_post = t3 - t2;
  uint32_t cyc_total = cyc_capture + cyc_pre + cyc_inf + cyc_post;

  // 0) Timestamp del evento (para emparejar con el ground truth en el host)
  Serial.print("TS_MS:");
  Serial.println(millis());

  // 1) Clase (entero SOLO en su linea, parseable por el host)
  Serial.println(best_class);

  // 2) Latencia en ciclos de nucleo (medida cruda, sin conversion)
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

  // 3) Latencia en microsegundos (usando SystemCoreClock real del MCU)
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

// ---- Telemetria termica (fuera de la region cronometrada) ----
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

// ---- Funcion: devolver el frame capturado al host (frame-dump) ----
// Envia el frame CRUDO de la HM01B0 (160x120, tal cual salio del sensor,
// ANTES del resize a la geometria del modelo) con el esquema de escape del
// protocolo PIL ('#'/'@'/ESC -> ESC, byte XOR 0x20). El host lo guarda y
// puede correr el MISMO .tflite en PC (SIL) sobre este frame: asi la brecha
// PIL->HIL se descompone en (a) degradacion sensor/optica/pantalla y
// (b) divergencia de ejecucion chip vs PC.
// Se envia FUERA de la region cronometrada (despues de reportar latencias):
// no contamina la medicion.
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

// ---- Ciclo HIL: capturar frame real + inferencia sensor->prediccion ----
void runHilCapture() {
  Serial.println("\n--- Trigger HIL. Capturando frame de la HM01B0... ---");

  // Telemetria termica ANTES de la region medida
  printMcuTemperature();

  // --- Fase CAPTURE: camara -> framebuffer (medida con DWT->CYCCNT) ---
  uint32_t tc0 = dwt_cycles();
  int grab_status = cam.grabFrame(fb, 3000); // timeout 3 s
  uint32_t tc1 = dwt_cycles();

  if (grab_status != 0) {
    Serial.println("ERROR: grabFrame() fallo (timeout o camara desconectada).");
    Serial.println("--- Listo. Esperando siguiente trigger... ---\n");
    Serial.flush();
    return;
  }
  uint32_t cyc_capture = tc1 - tc0;

  // --- Resize/crop camara -> geometria del modelo (cuenta como PRE) ---
  uint32_t tr0 = dwt_cycles();
  cameraFrameToImageBuffer(fb.getBuffer());
  uint32_t tr1 = dwt_cycles();

  runInference(cyc_capture, tr1 - tr0);

  Serial.println("--- Listo. Esperando siguiente trigger... ---\n");
  Serial.flush();

  // Frame-dump (fuera de la medicion): devolver el frame crudo al host para
  // trazabilidad y validacion cruzada SIL-en-PC.
  if (frame_dump) {
    sendFrameToHost(fb.getBuffer(), (size_t)(CAM_WIDTH * CAM_HEIGHT));
  }
}

// ---- Dispatcher de comandos serie ('T', 'F0'/'F1') ----
void handleSerialCommands() {
  while (Serial.available() > 0) {
    uint8_t byte_in = (uint8_t)Serial.read();

    if (byte_in == 'T') {
      runHilCapture();
    } else if (byte_in == 'F') {
      // Comando frame-dump: esperar el segundo byte ('0' o '1')
      unsigned long t = millis();
      while (!Serial.available()) {
        if (millis() - t > 500)
          return; // comando incompleto: descartar
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
    // cualquier otro byte se ignora (ruido)
  }
}

void setup() {
  SDRAM.begin();

  // Buffer de trabajo del frame redimensionado, en SDRAM
  image_buffer = (uint8_t *)ea_malloc(IMAGE_BUFFER_SIZE);

  // Memoria con 16 bytes extra para alineacion
  uint8_t *raw_arena = (uint8_t *)ea_malloc(kTensorArenaSize + 16);

  Serial.begin(115200);
  while (!Serial) {
  }
  delay(2000);

  // Habilitar el contador de ciclos DWT para medir latencia con precision de
  // 1 ciclo. Debe hacerse una sola vez, aqui.
  dwt_enable_cycle_counter();

  if (raw_arena == nullptr || image_buffer == nullptr) {
    Serial.println("ERROR: No hay SDRAM disponible para Arena o Image Buffer.");
    while (1)
      ;
  }

  // Alineacion a 16 bytes
  tensor_arena = (uint8_t *)(((uintptr_t)raw_arena + 15) & ~15);

  tflite::InitializeTarget();

  Serial.println("\n--- INICIANDO DIAGNOSTICO (HIL/camara) ---");
  Serial.print("Direccion del Tensor Arena en SDRAM: 0x");
  Serial.println((uint32_t)tensor_arena, HEX);
  Serial.print("SystemCoreClock (Hz): ");
  Serial.println(SystemCoreClock);
  Serial.flush();

  Serial.println("1. Registrando operaciones (AllOpsResolver)...");
  Serial.println("2. Leyendo el modelo de la memoria Flash...");
  Serial.flush();

  model = tflite::GetModel(g_model);

  Serial.println("3. Verificando version del modelo...");
  Serial.flush();
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Version mismatch!");
    while (1)
      ;
  }

  Serial.println("4. Creando el interprete estatico...");
  Serial.flush();

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  Serial.println("5. Ejecutando AllocateTensors()...");
  Serial.flush();

  TfLiteStatus allocate_status = interpreter->AllocateTensors();

  if (allocate_status != kTfLiteOk) {
    Serial.println("Fallo al asignar tensores. Arena muy pequeno?");
    while (1)
      ;
  }
  Serial.println("AllocateTensors() EXITOSO!");

  input = interpreter->input(0);
  output = interpreter->output(0);

  // Geometria del tensor de entrada (NHWC): la usa el resize camara->modelo
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

  // Diagnostico de cuantizacion del tensor de entrada
  if (input->type == kTfLiteInt8) {
    Serial.print("Input int8 -> scale=");
    Serial.print(input->params.scale, 8);
    Serial.print(" zero_point=");
    Serial.println(input->params.zero_point);
  }

  // --- Inicializar la camara HM01B0 (160x120 gris, 30 fps) ---
  if (cam.begin(CAMERA_R160x120, CAMERA_GRAYSCALE, CAM_FPS) == 0) {
    Serial.println("CAM_INIT:OK");
  } else {
    Serial.println("CAM_INIT:FAIL");
    Serial.println("ERROR: Vision Shield no detectado. Sistema detenido.");
    Serial.flush();
    while (1)
      ;
  }

  Serial.println("Setup completado exitosamente.");
  Serial.println("Comandos: 'T'=capturar+inferir  'F1'/'F0'=frame-dump on/off");
  Serial.println("READY_HIL");
  Serial.flush();
}

void loop() {
  // Loop no bloqueante: polling continuo de comandos del host
  handleSerialCommands();
}
