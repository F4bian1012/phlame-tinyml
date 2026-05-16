/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.
   Licensed under the Apache License, Version 2.0 (the "License") */

// TFLite Micro Includes
#include <Chirale_TensorFlowLite.h>
#include <SDRAM.h>
#include "model1.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Camera Includes
#include "camera.h"
#include "himax.h"
HM01B0 himax;
Camera cam(himax);

#define IMAGE_MODE CAMERA_GRAYSCALE
FrameBuffer fb;

// Globals for interacting with the TensorFlow Lite Micro model
const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr;
TfLiteTensor *output = nullptr;

// Tensor Arena: Definido como PUNTERO para usar SDRAM
constexpr int kTensorArenaSize = 1024 * 1024; // 1 MB
uint8_t *tensor_arena = nullptr;

tflite::MicroMutableOpResolver<20> resolver;

void setup() {
  SDRAM.begin();

  // Asignamos la memoria con 16 bytes extra
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
  Serial.flush();

  // Inicializar camara
  Serial.println("0. Inicializando Camara...");
  if (!cam.begin(CAMERA_R160x120, IMAGE_MODE, 30)) {
    Serial.println("Failed to initialize camera!");
    while (1);
  }
  Serial.println("Camara inicializada exitosamente (160x120 Grayscale)");

  Serial.println("1. Registrando operaciones...");
  Serial.flush();

  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddAdd();
  resolver.AddRelu();
  resolver.AddRelu6();
  // resolver.AddMean();
  // resolver.AddReshape();
  // resolver.AddPad();
  resolver.AddFullyConnected();
  // resolver.AddSoftmax();
  resolver.AddConcatenation();
  resolver.AddMul();
  // resolver.AddSub();

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
    Serial.println("Fallo al asignar tensores. ¿Arena muy pequeno?");
    while (1)
      ;
  }

  Serial.println("¡AllocateTensors() EXITOSO!");

  input = interpreter->input(0);
  output = interpreter->output(0);
  
  Serial.print("Dimensiones de entrada esperadas: ");
  for (int i = 0; i < input->dims->size; i++) {
    Serial.print(input->dims->data[i]);
    Serial.print(" ");
  }
  Serial.println();

  Serial.println("Setup completado exitosamente.");
  Serial.flush();
}

void loop() {
  // 1. Capturar la imagen
  if (cam.grabFrame(fb, 3000) != 0) {
    Serial.println("Fallo al capturar imagen.");
    delay(1000);
    return;
  }

  // Dimensiones de la cámara y del modelo
  int cam_w = 160;
  int cam_h = 120;
  int model_h = input->dims->data[1]; // asumiendo NHWC (Batch, Height, Width, Channels)
  int model_w = input->dims->data[2];
  
  if (model_w > cam_w || model_h > cam_h) {
    Serial.println("Error: El modelo requiere mayor resolucion que la camara (160x120).");
    delay(1000);
    return;
  }

  // Recortar la imagen desde el centro si la camara es mas grande que el modelo
  int start_x = (cam_w - model_w) / 2;
  int start_y = (cam_h - model_h) / 2;

  uint8_t* buffer = fb.getBuffer();
  int input_index = 0;

  // 2. Preprocesar y llenar el tensor de entrada
  for (int y = 0; y < model_h; y++) {
    for (int x = 0; x < model_w; x++) {
      int fb_index = (start_y + y) * cam_w + (start_x + x);
      uint8_t pixel = buffer[fb_index];

      if (input->type == kTfLiteInt8) {
        // En TFLite Int8 el rango de uint8 (0 a 255) normalmente se mapea a int8 (-128 a 127)
        input->data.int8[input_index] = (int8_t)pixel - 128;
      } else if (input->type == kTfLiteFloat32) {
        // En TFLite Float32 normalizar entre 0 y 1
        input->data.f[input_index] = pixel / 255.0f;
      }
      input_index++;
    }
  }

  // 3. Invocar el modelo
  unsigned long start_time = millis();
  TfLiteStatus invoke_status = interpreter->Invoke();
  unsigned long duration = millis() - start_time;

  if (invoke_status != kTfLiteOk) {
    MicroPrintf("Invoke failed");
    return;
  }

  // 4. Procesar y mostrar los resultados
  MicroPrintf("Inferencia exitosa. Tiempo: %d ms", duration);

  if (output->type == kTfLiteInt8) {
    int max_idx = 0;
    int8_t max_val = -128;
    int num_classes = output->dims->data[output->dims->size - 1];
    
    for (int i = 0; i < num_classes; i++) {
      int8_t val = output->data.int8[i];
      if (val > max_val) {
        max_val = val;
        max_idx = i;
      }
      MicroPrintf("Clase %d: %d", i, val);
    }
    MicroPrintf("-> Prediccion principal: Clase %d", max_idx);
    
  } else if (output->type == kTfLiteFloat32) {
    int max_idx = 0;
    float max_val = -1.0f;
    int num_classes = output->dims->data[output->dims->size - 1];
    
    for (int i = 0; i < num_classes; i++) {
      float val = output->data.f[i];
      if (val > max_val) {
        max_val = val;
        max_idx = i;
      }
      MicroPrintf("Clase %d: %f", i, val);
    }
    MicroPrintf("-> Prediccion principal: Clase %d", max_idx);
  }

  // Esperar un poco antes de la siguiente inferencia
  delay(1000);
}
