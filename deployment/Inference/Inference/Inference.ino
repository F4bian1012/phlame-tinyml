/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.
   Licensed under the Apache License, Version 2.0 (the "License") */

// ---- Librerías de cámara (HM01B0 / Portenta Vision Shield) ----
#include "camera.h"
#include "himax.h"

// ---- Librerías TFLite Micro ----
#include <Chirale_TensorFlowLite.h>
#include <SDRAM.h>

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// ---- Modelo embebido ----
#include "model1.h"

// ============================================================
// CONFIGURACIÓN DE CÁMARA
// ============================================================
HM01B0 himax;
Camera cam(himax);

#define CAM_RESOLUTION  CAMERA_R160x120
#define CAM_WIDTH       160
#define CAM_HEIGHT      120
#define IMAGE_MODE      CAMERA_GRAYSCALE

FrameBuffer fb;

// ============================================================
// CONFIGURACIÓN DEL MODELO
// ============================================================
#define MODEL_CHANNELS  1   // 1 = escala de grises

// Nombres de clases — ajusta según tu modelo
const char* CLASS_NAMES[] = {
  "Clase_0",
  "Clase_1",
  
};
const int NUM_CLASSES = sizeof(CLASS_NAMES) / sizeof(CLASS_NAMES[0]);

// ============================================================
// TENSOR ARENA EN SDRAM
// ============================================================
constexpr int kTensorArenaSize = 2048 * 1024; // 2 MB
uint8_t* tensor_arena = nullptr;

// ============================================================
// GLOBALES TFLite Micro
// ============================================================
const tflite::Model* tfl_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input  = nullptr;
TfLiteTensor* output = nullptr;

tflite::MicroMutableOpResolver<20> resolver;

// ============================================================
// FUNCIÓN: Redimensionar frame de cámara al tensor de entrada
//          usando interpolación por vecino más cercano.
// ============================================================
void resizeGrayscaleToTensor(const uint8_t* src,
                              int src_w, int src_h,
                              int dst_w, int dst_h) {
  for (int dy = 0; dy < dst_h; dy++) {
    for (int dx = 0; dx < dst_w; dx++) {
      int sx = (dx * src_w) / dst_w;
      int sy = (dy * src_h) / dst_h;
      uint8_t pixel = src[sy * src_w + sx];
      int idx = dy * dst_w + dx;

      if (input->type == kTfLiteInt8) {
        input->data.int8[idx] = (int8_t)((int)pixel - 128);
      } else if (input->type == kTfLiteFloat32) {
        input->data.f[idx] = (float)pixel / 255.0f;
      }
    }
  }
}

// ============================================================
// FUNCIÓN: Controlar LED RGB según la clase predicha
//          LED Portenta H7 — activo en LOW
//          Clase 1 → verde
//          Clase 2 → azul
//          Otro    → todos apagados
// ============================================================
void setLED(int best_class) {
  // Apagar todos primero
  digitalWrite(LEDR, HIGH);
  digitalWrite(LEDG, HIGH);
  digitalWrite(LEDB, HIGH);

  if (best_class == 0) {
    digitalWrite(LEDG, LOW);   // Verde → clase 1
  } else if (best_class == 1) {
    digitalWrite(LEDB, LOW);   // Azul  → clase 2
  }
  // Clase 0 u otras: todos apagados
}

// ============================================================
// FUNCIÓN: Ejecutar inferencia, controlar LED e imprimir clase
// ============================================================
void runInference() {
  unsigned long t0 = millis();
  TfLiteStatus status = interpreter->Invoke();
  unsigned long dt = millis() - t0;

  if (status != kTfLiteOk) {
    //Serial.println("ERROR: Invoke() fallo.");
    return;
  }

  int best_class = 0;

  if (output->type == kTfLiteInt8) {
    int num_classes = output->bytes;
    int8_t best_score = output->data.int8[0];
    for (int i = 1; i < num_classes; i++) {
      if (output->data.int8[i] > best_score) {
        best_score = output->data.int8[i];
        best_class = i;
      }
    }
  } else if (output->type == kTfLiteFloat32) {
    int num_classes = output->bytes / 4;
    float best_score = output->data.f[0];
    for (int i = 1; i < num_classes; i++) {
      if (output->data.f[i] > best_score) {
        best_score = output->data.f[i];
        best_class = i;
      }
    }
  }

  // Controlar LED según la clase
  setLED(best_class);

  // Imprimir resultado por //Serial
  //Serial.print("CLASE: ");
  //Serial.print(best_class);
  if (best_class < NUM_CLASSES) {
    //Serial.print(" (");
    //Serial.print(CLASS_NAMES[best_class]);
    //Serial.print(")");
  }
  //Serial.print("  | Tiempo inferencia: ");
  //Serial.print(dt);
  //Serial.println(" ms");
  //Serial.flush();
}

// ============================================================
// SETUP
// ============================================================
void setup() {
  // --- LEDs RGB del Portenta H7 (activo en LOW) ---
  pinMode(LEDR, OUTPUT);
  pinMode(LEDG, OUTPUT);
  pinMode(LEDB, OUTPUT);
  digitalWrite(LEDR, HIGH); // apagado
  digitalWrite(LEDG, HIGH); // apagado
  digitalWrite(LEDB, HIGH); // apagado

  // --- Inicializar SDRAM ---
  SDRAM.begin();
  uint8_t* raw_arena = (uint8_t*)ea_malloc(kTensorArenaSize + 16);

  // --- Puerto //Serial ---
  Serial.begin(115200);
  while (!Serial) {}
  delay(1000);

  if (raw_arena == nullptr) {
    //Serial.println("ERROR: No hay SDRAM disponible.");
    while (1);
  }
  tensor_arena = (uint8_t*)(((uintptr_t)raw_arena + 15) & ~15);

  //Serial.println("\n=== Iniciando sistema de inferencia con camara ===");

  // --- Inicializar cámara ---
  //Serial.println("Inicializando camara HM01B0...");
  if (!cam.begin(CAM_RESOLUTION, IMAGE_MODE, 30)) {
    //Serial.println("ERROR: No se pudo inicializar la camara.");
    while (1);
  }
  //Serial.println("Camara OK.");

  // --- Inicializar TFLite Micro ---
  tflite::InitializeTarget();

  //Serial.println("Registrando operaciones...");
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

  //Serial.println("Leyendo modelo desde Flash...");
  tfl_model = tflite::GetModel(g_model);

  if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
    //Serial.println("ERROR: Version del modelo no coincide con el schema.");
    while (1);
  }

  //Serial.println("Creando interprete...");
  static tflite::MicroInterpreter static_interpreter(
      tfl_model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  //Serial.println("Asignando tensores...");
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    //Serial.println("ERROR: AllocateTensors() fallo. Arena demasiado pequeno?");
    while (1);
  }

  input  = interpreter->input(0);
  output = interpreter->output(0);

  //Serial.print("Tensor de entrada: ");
  //Serial.print(input->dims->data[1]); //Serial.print("x");
  //Serial.print(input->dims->data[2]); //Serial.print("x");
  //Serial.println(input->dims->data[3]);

  //Serial.print("Tensor de salida: ");
  //Serial.print(output->bytes / (output->type == kTfLiteInt8 ? 1 : 4));
  //Serial.println(" clases");

  //Serial.println("=== Setup completado. Iniciando captura continua... ===\n");
  //Serial.flush();
}

// ============================================================
// LOOP
// ============================================================
void loop() {
  // Capturar frame de la camara
  if (cam.grabFrame(fb, 3000) == 0) {
    Serial.write(fb.getBuffer(),cam.frameSize());    
  } 

  const uint8_t* frame_data = fb.getBuffer();

  // Redimensionar al tamaño del tensor de entrada del modelo
  int model_w = input->dims->data[1];
  int model_h = input->dims->data[2];
  resizeGrayscaleToTensor(frame_data, CAM_WIDTH, CAM_HEIGHT, model_w, model_h);

  // Inferencia + LED + //Serial
  runInference();

  // Pausa entre capturas
  delay(50);
}