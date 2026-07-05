/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.
   Licensed under the Apache License, Version 2.0 (the "License") */

#include "mbed.h"
#include "model.h"
#include <Chirale_TensorFlowLite.h>
#include <SDRAM.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Globals for interacting with the TensorFlow Lite Micro model
const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr;
TfLiteTensor *output = nullptr;

// Tensor Arena: Definido como PUNTERO para usar SDRAM
constexpr int kTensorArenaSize =
    4 * 1024 * 1024; // 4 MB para modelos más grandes
uint8_t *tensor_arena = nullptr;

// Usamos AllOpsResolver porque la Portenta H7 tiene mucha memoria Flash (2MB)
// y esto previene errores de "AllocateTensors()" por operaciones faltantes
tflite::AllOpsResolver resolver;

// ---- PROTOCOLO SERIAL ----
// Marcador de inicio: '#'  (0x23)
// Marcador de fin:    '@'  (0x40)
// Los bytes de imagen se envían RAW entre los marcadores.
// Si el propio dato es '#' o '@', el emisor debe escaparlos (ver script
// Python). Escape: 0x1B (ESC) seguido del byte XOR 0x20.

// Buffer para recibir la imagen por serial
// Aumentado a 400KB para soportar imágenes grandes como 320x320 RGB (307,200
// bytes)
#define IMAGE_BUFFER_SIZE (400 * 1024)

uint8_t *image_buffer = nullptr;
int image_bytes_received = 0;
bool receiving = false;

// ---- Función: leer imagen del serial ----
// Retorna true si se recibió una imagen completa
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
  return false; // imagen aún incompleta
}

// ---- Función: cargar imagen al tensor de entrada ----
void loadImageToInputTensor() {
  int expected_bytes = 0;

  if (input->type == kTfLiteInt8) {
    expected_bytes = input->bytes; // total bytes del tensor (e.g. 48*48*1)
    Serial.print("expected_bytes: ");
    Serial.println(expected_bytes);
    int to_copy = (image_bytes_received < expected_bytes) ? image_bytes_received
                                                          : expected_bytes;
    for (int i = 0; i < to_copy; ++i) {
      // Los bytes recibidos son uint8 (0-255); convertir a int8 restando 128
      input->data.int8[i] = (int8_t)((int)image_buffer[i] - 128);
    }
    // Rellenar con 0 si se recibieron menos bytes de los esperados
    for (int i = to_copy; i < expected_bytes; ++i) {
      input->data.int8[i] = 0;
    }

  } else if (input->type == kTfLiteFloat32) {
    expected_bytes = input->bytes / 4; // número de floats
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

// ---- Función: ejecutar inferencia y reportar resultado ----
void runInference() {
  loadImageToInputTensor();

  TfLiteStatus invoke_status = interpreter->Invoke();

  if (invoke_status != kTfLiteOk) {
    Serial.println("ERROR: Invoke() fallo.");
    return;
  }

  if (output->type == kTfLiteInt8) {
    int num_classes = output->bytes;
    int best_class = 0;
    int8_t best_score = output->data.int8[0];
    for (int i = 0; i < num_classes; ++i) {
      if (output->data.int8[i] > best_score) {
        best_score = output->data.int8[i];
        best_class = i;
      }
    }
    Serial.println(best_class);

  } else if (output->type == kTfLiteFloat32) {
    int num_classes = output->bytes / 4;
    int best_class = 0;
    float best_score = output->data.f[0];
    for (int i = 0; i < num_classes; ++i) {
      if (output->data.f[i] > best_score) {
        best_score = output->data.f[i];
        best_class = i;
      }
    }
    Serial.println(best_class);
  }
}

void setup() {
  SDRAM.begin();

  // Asignamos la memoria del buffer de imagen en la SDRAM
  image_buffer = (uint8_t *)ea_malloc(IMAGE_BUFFER_SIZE);

  // Memoria con 16 bytes extra para alineación
  uint8_t *raw_arena = (uint8_t *)ea_malloc(kTensorArenaSize + 16);

  Serial.begin(115200);
  while (!Serial) {
  }
  delay(2000);

  if (raw_arena == nullptr || image_buffer == nullptr) {
    Serial.println("ERROR: No hay SDRAM disponible para Arena o Image Buffer.");
    while (1)
      ;
  }

  // Alineación a 16 bytes
  tensor_arena = (uint8_t *)(((uintptr_t)raw_arena + 15) & ~15);

  tflite::InitializeTarget();

  Serial.println("\n--- INICIANDO DIAGNOSTICO ---");
  Serial.print("Direccion del Tensor Arena en SDRAM: 0x");
  Serial.println((uint32_t)tensor_arena, HEX);
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
    Serial.println("Fallo al asignar tensores. ¿Arena muy pequeño?");
    while (1)
      ;
  }

  Serial.println("¡AllocateTensors() EXITOSO!");

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Setup completado exitosamente.");
  Serial.print("Esperando imagen por Serial (inicio='#', fin='@') ...");
  Serial.println();
  Serial.flush();
}

void loop() {
  // Leer imagen del serial usando el protocolo #...@
  if (readImageFromSerial()) {

    Serial.println("\n--- Imagen recibida. Iniciando inferencia... ---");

    mbed::AnalogIn mcuADCVref(ADC_VREF);
    mbed::AnalogIn mcuADCTemp(ADC_TEMP);

    uint16_t rawVref = mcuADCVref.read_u16();
    uint16_t rawTemp = mcuADCTemp.read_u16();

    // Voltaje de referencia analógica interno (en mV)
    uint32_t mcuVref =
        __LL_ADC_CALC_VREFANALOG_VOLTAGE(rawVref, ADC_RESOLUTION_16B);

    // Temperatura final usando las constantes de fábrica del STM32H7
    int32_t mcuTemp =
        __HAL_ADC_CALC_TEMPERATURE(mcuVref, rawTemp, ADC_RESOLUTION_16B);
    Serial.print(" mV | Temperatura Interna CPU: ");
    Serial.print(mcuTemp);
    Serial.println(" °C");
    // if (mcuTemp>=40){
    delay(1000);
    // }

    runInference();

    Serial.println("--- Listo. Esperando siguiente imagen... ---\n");
    Serial.flush();
  }
  // No hay delay fijo: el loop es no bloqueante, polling continuo del serial
}