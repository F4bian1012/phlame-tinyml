# Investigación TinyML - Arduino Portenta H7

Este proyecto sigue una arquitectura **MLOps** rigurosa adaptada para Sistemas Embebidos (TinyML). El objetivo es garantizar la reproducibilidad de los experimentos, la trazabilidad de los modelos y una clara separación entre el entrenamiento (Python) y el despliegue (C++).

## 📂 Filosofía de la Estructura de Directorios

La organización de carpetas ha sido diseñada siguiendo las mejores prácticas de la industria:

### 1. `data/` - El ciclo de vida de los datos
Separamos los datos en tres estados inmutables para evitar corrupción y fugas de información:
- **`raw/`**: Datos crudos tal como se obtuvieron del sensor o fuente. **Nunca se sobreescriben.**
- **`processed/`**: Datos limpios, normalizados y listos para entrar al modelo.
- **`augmented/`**: Versiones generadas artificialmente para robustecer el entrenamiento (ruido añadido, cambios de tono, etc.).

### 2. `models/` - Gestión de Artefactos
- **`checkpoints/`**: Guardamos los modelos "pesados" de entrenamiento (formato `.keras` o `.h5`). Estos contienen el estado completo del optimizador.
- **`tflite/`**: Aquí residen únicamente los modelos cuantizados y optimizados para el microcontrolador (`.tflite`) y sus conversiones a arreglos de C (`model.h`).

### 3. `src/` vs `deployment/`
- **`src/`**: Contiene todo el código Python para la "ciencia" del proyecto (entrenamiento, evaluación, scripts de utilidad). Es el entorno del Data Scientist.
- **`deployment/arduino_project/`**: Contiene el código fuente C++ final que se cargará en la Portenta H7. Es el entorno del Ingeniero Embebido. Mantener esto separado evita conflictos de dependencias entre Python y C++.

### 4. `logs/` y `config/`
- **`logs/`**: Almacena gráficas de pérdidas (loss curves), métricas de precisión y registros de experimentos. Fundamental para comparar qué iteración del modelo funciona mejor.
- **`config/`**: Archivos de configuración o hiperparámetros.

## 🚀 Cómo empezar

1. Instala las dependencias de Python:
   ```bash
   pip install -r requirements.txt
   ```
2. Realiza tus experimentos en `notebooks/` o scripts en `src/`.
4. Copia el arreglo C generado a `deployment/arduino_project/` para compilarlo en Arduino IDE.

## 🔄 Flujo de Trabajo Principal (MLOps)

Scripts orquestadores para el ciclo de vida completo del modelo:

### 1. Procesamiento de Imágenes (`src/process_images.py`)
Convierte imágenes a escala de grises controlando la compresión para evitar que aumente el tamaño del archivo, preparándolas para el entrenamiento.

**Uso básico (rutas por defecto):**
```bash
python src/process_images.py
```

**Uso con rutas personalizadas:**
```bash
python src/process_images.py --raw_path "ruta/a/imagenes_crudas" --path_processed "ruta/a/destino"
```

### 2. Redimensionamiento de Imágenes (`src/reshape_images.py`)
Redimensiona las imágenes procesadas al tamaño objetivo para TinyML, conservando la estructura de carpetas (clases).

**Uso básico (rutas por defecto, busca recursivamente en `data/processed/grayscale`):**
```bash
python src/reshape_images.py --width 96 --height 96
```

**Uso con directorio de entrada personalizado:**
```bash
python src/reshape_images.py --input_dir "ruta/personalizada" --width 96 --height 96
```

### 3. Entrenamiento Modelo Base (`src/train_model.py`)
Entrena el modelo base (CNN) con datos de Fashion MNIST normalizados.
- **Entrada:** Dataset Fashion MNIST (descarga automática).
- **Salida:** `models/checkpoints/best_model.keras`.
- **Uso:** `python src/train_model.py`

### 3.1 Entrenamiento MobileNet - Datos Personalizados (`src/train_mobilenet.py`)
Entrena un modelo de la familia MobileNet con Transfer Learning sobre los datos procesados, ideal para inferencia en el Portenta H7.

**Entradas Automáticas:**
- **Entrada:** Las imágenes se cargan automáticamente desde `data/processed/{width}x{height}/` según la resolución indicada.

**Salidas (Checkpoints y Logs Dinámicos):** 
Se guardan de forma dinámica incorporando los parámetros en el nombre del archivo para facilitar el tracking y versionado de experimentos:
- **Modelo:** `models/checkpoints/{base_model}+{batch_size}+{epochs}+{learning_rate}+{validation_split}+{width}+{height}.keras`
- **Gráfica:** `tensorboard_logs/{base_model}_training_history+{batch_size}+{epochs}+{learning_rate}+{validation_split}+{width}+{height}.png`

**Hiperparámetros (Soportados por CLI):**
- `--base_model` (Opciones: `MobileNet`, `MobileNetV2`, `MobileNetV3Large`, `MobileNetV3Small` | Por defecto: `MobileNetV2`)
- `--width` y `--height` (Por defecto: 96)
- `--batch_size` (Por defecto: 32)
- `--epochs` (Por defecto: 20)
- `--learning_rate` (Por defecto: 0.0001)
- `--validation_split` (Por defecto: 0.2)

**Uso:**
```bash
# Entrenamiento con valores por defecto (MobileNetV2)
python src/train_mobilenet.py

# Uso con el modelo base original (MobileNet v1)
python src/train_mobilenet.py --base_model "MobileNet"

# Personalizando los hiperparámetros
python src/train_mobilenet.py --base_model "MobileNetV3Large" --width 96 --height 96 --batch_size 16 --epochs 30 --learning_rate 0.00005 --validation_split 0.25
```

### 4. Evaluación de Modelos (`src/test_model.py`)
Script dedicado a calcular métricas estadísticas extensas (Accuracy, Precision, Recall, F1-Score) y visualizar el desempeño clase por clase generando una Matriz de Confusión.

- **Entrada Automática:** Si no se especifica explícitamente, el script infiere la ruta del modelo y del directorio de datos basándose en los parámetros de inferencia (`--width`, `--height`, `--learning_rate`).
- **Salida:** Imprime métricas globales y por clase detalladas en la terminal. Paralelamente, guarda de forma dinámica el gráfico de la matriz de confusión usando el nombre de tu modelo evaluado (`Matriz_{nombre_del_modelo}.png`).

**Hiperparámetros (Soportados por CLI):**
- `--width` y `--height` (Por defecto: 96, definen resolución de carga de imágenes y carpetas)
- `--learning_rate` (Por defecto: 0.0001)
- `--model_path` (Opcional, ignora las rutas inferidas si se proporciona)
- `--data_dir` (Opcional, por defecto resuelve a `data/processed/{width}x{height}`)

**Uso Estándar (Deducción de rutas usando argumentos):**
```bash
# El script asume parámetros por defecto para buscar el modelo y el dataset
python src/test_model.py

# Modificando hiperparámetros (Inferirá modelo MobileNet+32+20+0.0005+0.2+160+160.keras)
python src/test_model.py --width 160 --height 160 --learning_rate 0.0005
```

**Uso con ruta definida (Evita la deducción automática):**
```bash
python src/test_model.py --model_path "models/checkpoints/MobileNet+32+20+0.0001+0.2+320+320.keras"
```

### 5. Cuantización a INT8 (`src/quantize_int8_basic.py`)
Convierte un modelo entrenado en formato `.keras` a un modelo optimizado `.tflite` utilizando Full Integer Quantization (INT8). Este proceso es indispensable para poder ejecutar el modelo de manera eficiente en hardware con recursos limitados (como Arduino Portenta H7).

- **Entrada Automática:** Busca automáticamente el primer modelo disponible en `models/checkpoints/` si no se especifica.
- **Calibración:** Utiliza un dataset representativo extraído de `data/processed/` para calibrar las activaciones.
- **Salida:** Un modelo TFLite completamente cuantizado en `models/tflite/`.

**Hiperparámetros (Soportados por CLI):**
- `--model_path` (Opcional, ruta específica del modelo `.keras` a cuantizar)

**Uso:**
```bash
# Búsqueda y proceso automático
python src/quantize_int8_basic.py

# Especificando la ruta exacta al modelo
python src/quantize_int8_basic.py --model_path "models/checkpoints/MobileNet+32+20+0.0001+0.2+96+96.keras"
```

### 6. Evaluación de Modelos Cuantizados TFLite (`src/test_tflite_model.py`)
Recrea la lógica de evaluación (matriz de confusión y reporte de métricas) orientada exclusivamente a los modelos `.tflite` exportados con validación INT8. Identifica los tensores, los cuantiza automáticamente previo a la inferencia de TFLite y descuantiza los resultados antes de evaluarlos.

- **Entrada:** Modelo INT8 ubicado en `models/tflite/` y el dataset original de procesamiento en escala de grises.
- **Salida:** Métricas extensas por la terminal y creación de la matriz de confusión estéticamente amigable junto al modelo evaluado.

**Hiperparámetros (Soportados por CLI):**
- `--width` y `--height` (Por defecto: 96, define resolución final exigida por el modelo y para carga de imágenes)
- `--model_path` (Opcional, ruta al modelo TFLite a cargar)
- `--data_dir` y `--class_names_path` (Opcionales, rutas personalizadas a características de los datos)

**Uso:**
```bash
# Inferencia estándar indicando la ruta del modelo .tflite y resolución del sensor
python src/test_tflite_model.py --model_path "models/tflite/Modelo_int8.tflite" --width 96 --height 96
```

## 📸 Captura y Visualización de Imágenes

Herramientas para capturar y visualizar datos desde la cámara de la Portenta H7.

### 1. Firmware Arduino (`deployment/arduino/image_capture/image_capture.ino`)
Script para la Portenta H7 que captura imágenes en escala de grises (160x120) y las envía como bytes crudos a través del puerto serial.
- **Configuración:** QQVGA (160x120), Grayscale, 30 FPS.
- **Uso:** Cargar en la placa usando Arduino IDE.

### 2. Visualizador Python (`src/visualize_serial_image.py`)
Script para recibir y renderizar en tiempo real las imágenes enviadas por el Arduino.
- **Detección automática de puerto:** Intenta encontrar el puerto serial de la Portenta si no se especifica.
- **Renderizado:** Utiliza `matplotlib` para mostrar el stream de video.

**Uso:**
```bash
python src/visualize_serial_image.py
# O especificando el puerto manualmente:
# En mac
python src/visualize_serial_image.py --port /dev/tty.usbmodem1301 
# En windows
python src/visualize_serial_image.py --port COM7
```
