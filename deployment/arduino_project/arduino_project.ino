/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.
   Licensed under the Apache License, Version 2.0 (the "License") */

#include <Chirale_TensorFlowLite.h>
#include <SDRAM.h>
// include static array definition of pre-trained model
#include "model1.h"

// --- ESTOS SON LOS INCLUDES CORRECTOS ---
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
// ----------------------------------------

// Globals for interacting with the TensorFlow Lite Micro model
const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr;
TfLiteTensor *output = nullptr;

// Tensor Arena: Definido como PUNTERO para usar SDRAM
constexpr int kTensorArenaSize = 1024 * 1024; // 2 MB
uint8_t *tensor_arena = nullptr;

// AQUÍ ESTÁ EL CAMBIO CLAVE: Usamos MicroMutableOpResolver en lugar de
// AllOpsResolver
tflite::MicroMutableOpResolver<20> resolver;

void setup() {
  SDRAM.begin();

  // 1. Asignamos la memoria con 16 bytes extra
  uint8_t *raw_arena = (uint8_t *)ea_malloc(kTensorArenaSize + 16);

  Serial.begin(115200);
  while (!Serial) {
  }
  delay(2000);

  if (raw_arena == nullptr) {
    Serial.println("ERROR: No hay SDRAM disponible.");
    while (1)
      ;
  }

  // Alineación a 16 bytes
  tensor_arena = (uint8_t *)(((uintptr_t)raw_arena + 15) & ~15);

  tflite::InitializeTarget();

  Serial.println("\n--- INICIANDO DIAGNOSTICO ---");
  Serial.print("Direccion del Tensor Arena en SDRAM: 0x");
  Serial.println((uint32_t)tensor_arena, HEX);
  Serial.flush(); // Obliga a imprimir inmediatamente

  Serial.println("1. Registrando operaciones...");
  Serial.flush();

  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddAdd();
  resolver.AddRelu();
  resolver.AddRelu6();
  resolver.AddMean();
  resolver.AddReshape();
  resolver.AddPad();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddConcatenation();
  resolver.AddMul();
  resolver.AddSub();

  Serial.println("2. Leyendo el modelo de la memoria Flash...");
  Serial.flush();

  // Si colapsa aquí, el problema es el array g_model en model1.h
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

  // Usamos instanciación estática (más segura para Mbed OS que 'new')
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  Serial.println("5. Ejecutando AllocateTensors() [PELIGRO]...");
  Serial.flush();

  // Si colapsa aquí, el problema sigue siendo la RAM externa (SDRAM)
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
  Serial.flush();
}

void loop() {
  // 1. Fill the input tensor with test data.
  if (input->type == kTfLiteInt8) {
    for (int i = 0; i < input->bytes; ++i) {
      input->data.int8[i] = 0; // Dummy zero values
    }
  } else if (input->type == kTfLiteFloat32) {
    for (int i = 0; i < input->bytes / 4; ++i) {
      input->data.f[i] = 0.0f; // Dummy zero values
    }
  }

  // 2. Invoke the model
  unsigned long start_time = millis();

  TfLiteStatus invoke_status = interpreter->Invoke();

  unsigned long duration = millis() - start_time;

  if (invoke_status != kTfLiteOk) {
    MicroPrintf("Invoke failed");
    return;
  }

  // 3. Process the results.
  MicroPrintf("Inferencia exitosa. Tiempo: %d ms", duration);

  // Just print the first class probability or value as a sanity check
  if (output->type == kTfLiteInt8) {
    MicroPrintf("Output[0] int8 = %d", output->data.int8[0]);
  } else if (output->type == kTfLiteFloat32) {
    float f_val = static_cast<float>(output->data.f[0]);
    MicroPrintf("Output[0] float = %f", f_val);
  }

  // Wait a little before the next inference
  delay(1000);
}