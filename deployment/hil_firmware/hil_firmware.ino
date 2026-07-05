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

// Tensor Arena: Definido como PUNTERO para usar SDRAM
constexpr int kTensorArenaSize =
    4 * 1024 * 1024; // 4 MB para modelos mas grandes
uint8_t *tensor_arena = nullptr;

// Usamos AllOpsResolver porque la Portenta H7 tiene mucha memoria Flash (2MB)
// y esto previene errores de "AllocateTensors()" por operaciones faltantes
tflite::AllOpsResolver resolver;

// ---- PROTOCOLO SERIAL ----
// Marcador de inicio: '#'  (0x23)
// Marcador de fin:    '@'  (0x40)
// Los bytes de imagen se envian RAW entre los marcadores.
// Si el propio dato es '#' o '@', el emisor debe escaparlos (ver script
// Python). Escape: 0x1B (ESC) seguido del byte XOR 0x20.

// Buffer para recibir la imagen por serial
// Aumentado a 400KB para soportar imagenes grandes como 320x320 RGB (307,200
// bytes)
#define IMAGE_BUFFER_SIZE (400 * 1024)

uint8_t *image_buffer = nullptr;
int image_bytes_received = 0;
bool receiving = false;

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

// ---- Funcion: leer imagen del serial ----
// Retorna true si se recibio una imagen completa
bool readImageFromSerial() {
  while (Serial.available() > 0) {
    uint8_t byte_in = (uint8_t)Serial.read();

    if (!receiving) {
      // Esperamos el marcador de inicio '#'
      if (byte_in == '#') {
        receiving = true;
        image_bytes_received = 0;
      }
      // ignorar cualquier otro byte fuera del paquete
    } else {
      // Estamos dentro del paquete
      if (byte_in == '@') {
        // Marcador de fin detectado
        receiving = false;
        return true; // imagen completa
      } else if (byte_in == 0x1B) {
        // Byte de escape: el siguiente byte es el dato real XOR 0x20
        // Esperamos el siguiente byte (bloqueante con timeout corto)
        unsigned long t = millis();
        while (!Serial.available()) {
          if (millis() - t > 500) {
            // timeout: descartamos el paquete
            receiving = false;
            return false;
          }
        }
        uint8_t escaped = (uint8_t)Serial.read() ^ 0x20;
        if (image_bytes_received < IMAGE_BUFFER_SIZE) {
          image_buffer[image_bytes_received++] = escaped;
        }
      } else {
        // Byte normal
        if (image_bytes_received < IMAGE_BUFFER_SIZE) {
          image_buffer[image_bytes_received++] = byte_in;
        }
      }
    }
  }
  return false; // imagen aun incompleta
}

// ---- Funcion: cargar imagen al tensor de entrada ----
void loadImageToInputTensor() {
  int expected_bytes = 0;

  if (input->type == kTfLiteInt8) {
    expected_bytes = input->bytes; // total bytes del tensor (e.g. 160*120*1)

    // Parametros de cuantizacion REALES del tensor (los fija el TFLite
    // converter). NO asumir zero_point=-128 / scale=1/255: eso solo es
    // correcto para un modelo concreto y falla con cualquier otro.
    const float in_scale = input->params.scale;
    const int32_t in_zero_point = input->params.zero_point;

    int to_copy = (image_bytes_received < expected_bytes) ? image_bytes_received
                                                          : expected_bytes;
    for (int i = 0; i < to_copy; ++i) {
      // 1) valor real que el modelo espera.
      //    Este modelo fue convertido SIN capa Rescaling: espera el pixel CRUDO
      //    en [0,255] (con scale~1.0, zp=-128 esto reproduce el viejo
      //    pixel-128). Si en el futuro entrenas con Rescaling(1./255) -> [0,1]:
      //    usa
      //      (float)image_buffer[i] / 255.0f
      //    Si usas preprocess_input de MobileNet -> [-1,1]:
      //      ((float)image_buffer[i] / 127.5f) - 1.0f
      //    Debe COINCIDIR con el preprocesado del dataset representativo del
      //    TFLite converter (mira process_images.py / la conversion).
      const float real_value = (float)image_buffer[i];
      // 2) cuantizar: q = round(real / scale) + zero_point
      int32_t q = (int32_t)lroundf(real_value / in_scale) + in_zero_point;
      // 3) saturar al rango int8 [-128, 127]
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
    int to_copy = (image_bytes_received < expected_bytes) ? image_bytes_received
                                                          : expected_bytes;
    for (int i = 0; i < to_copy; ++i) {
      // Normalizar de [0,255] a [0.0, 1.0]
      input->data.f[i] = (float)image_buffer[i] / 255.0f;
    }
    for (int i = to_copy; i < expected_bytes; ++i) {
      input->data.f[i] = 0.0f;
    }
  }
}

// ---- Funcion: ejecutar inferencia y reportar resultado + latencias ----
// Mide la latencia DESAGREGADA POR FASE con DWT->CYCCNT:
//   - preprocess : cargar/cuantizar la imagen al tensor de entrada
//   - inference  : interpreter->Invoke()
//   - postprocess: argmax de la salida
// La clase (argmax) se imprime como entero en su propia linea para mantener la
// compatibilidad con hil_benchmark.py. Las latencias se emiten en lineas con
// prefijos parseables (CYC_* y US_*).
void runInference() {
  // --- Fase 1: preprocesado (carga + cuantizacion de la imagen) ---
  uint32_t t0 = dwt_cycles();
  loadImageToInputTensor();
  uint32_t t1 = dwt_cycles();

  // --- Fase 2: inferencia ---
  TfLiteStatus invoke_status = interpreter->Invoke();
  uint32_t t2 = dwt_cycles();

  if (invoke_status != kTfLiteOk) {
    Serial.println("ERROR: Invoke() fallo.");
    return;
  }

  // --- Fase 3: postprocesado (argmax) ---
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
  uint32_t cyc_pre = t1 - t0;
  uint32_t cyc_inf = t2 - t1;
  uint32_t cyc_post = t3 - t2;
  uint32_t cyc_total = t3 - t0;

  // 1) Clase (compatibilidad con el host: entero solo en su linea)
  Serial.println(best_class);

  // 2) Latencia en ciclos de nucleo (medida cruda, sin conversion)
  Serial.print("CYC_PRE:");
  Serial.println(cyc_pre);
  Serial.print("CYC_INF:");
  Serial.println(cyc_inf);
  Serial.print("CYC_POST:");
  Serial.println(cyc_post);
  Serial.print("CYC_TOTAL:");
  Serial.println(cyc_total);

  // 3) Latencia en microsegundos (usando SystemCoreClock real del MCU)
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

  // Asignamos la memoria del buffer de imagen en la SDRAM
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

  Serial.println("\n--- INICIANDO DIAGNOSTICO ---");
  Serial.print("Direccion del Tensor Arena en SDRAM: 0x");
  Serial.println((uint32_t)tensor_arena, HEX);
  Serial.print("SystemCoreClock (Hz): ");
  Serial.println(SystemCoreClock);
  Serial.flush();

  Serial.println("1. Registrando operaciones (AllOpsResolver)...");
  Serial.flush();

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

  Serial.println("5. Ejecutando AllocateTensors() [PELIGRO]...");
  Serial.flush();

  TfLiteStatus allocate_status = interpreter->AllocateTensors();

  Serial.println("6. AllocateTensors terminado.");
  Serial.flush();

  if (allocate_status != kTfLiteOk) {
    Serial.println("Fallo al asignar tensores. Arena muy pequeno?");
    while (1)
      ;
  }

  Serial.println("AllocateTensors() EXITOSO!");

  input = interpreter->input(0);
  output = interpreter->output(0);

  // Diagnostico de cuantizacion: imprime scale/zero_point del tensor de
  // entrada. Si ves scale~0.00392 (=1/255) y zp=-128, el atajo viejo
  // "pixel-128" era correcto POR CASUALIDAD para este modelo; con el fix
  // funciona para todos.
  if (input->type == kTfLiteInt8) {
    Serial.print("Input int8 -> scale=");
    Serial.print(input->params.scale, 8);
    Serial.print(" zero_point=");
    Serial.println(input->params.zero_point);
  }

  Serial.println("Setup completado exitosamente.");
  Serial.print("Esperando imagen por Serial (inicio='#', fin='@') ...");
  Serial.println();
  Serial.flush();
}

void loop() {
  // Leer imagen del serial usando el protocolo #...@
  if (readImageFromSerial()) {

    Serial.println("\n--- Imagen recibida. Iniciando inferencia... ---");

    // Telemetria termica (fuera de la region cronometrada): se lee ANTES de la
    // inferencia para no contaminar la medida de latencia.
    mbed::AnalogIn mcuADCVref(ADC_VREF);
    mbed::AnalogIn mcuADCTemp(ADC_TEMP);

    uint16_t rawVref = mcuADCVref.read_u16();
    uint16_t rawTemp = mcuADCTemp.read_u16();

    // Voltaje de referencia analogica interno (en mV)
    uint32_t mcuVref =
        __LL_ADC_CALC_VREFANALOG_VOLTAGE(rawVref, ADC_RESOLUTION_16B);

    // Temperatura final usando las constantes de fabrica del STM32H7
    int32_t mcuTemp =
        __HAL_ADC_CALC_TEMPERATURE(mcuVref, rawTemp, ADC_RESOLUTION_16B);
    Serial.print("TEMP_C:");
    Serial.println(mcuTemp);

    // NOTA: se ELIMINO el delay(1000) que habia aqui. Ese retardo se ejecutaba
    // dentro del ciclo de medicion y contaminaba cualquier latencia calculada
    // por el host con millis(). La latencia real ahora la mide el firmware con
    // DWT->CYCCNT dentro de runInference() y se reporta en las lineas
    // US_*/CYC_*.

    runInference();

    Serial.println("--- Listo. Esperando siguiente imagen... ---\n");
    Serial.flush();
  }
  // No hay delay fijo: el loop es no bloqueante, polling continuo del serial
}
