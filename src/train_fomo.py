"""
train_fomo.py
=============
Entrenamiento de un modelo FOMO (Faster Objects, More Objects) para detección
de objetos en TinyML / microcontroladores (ej. Arduino Portenta H7).

¿Qué es FOMO?
─────────────
FOMO es una arquitectura de detección ligera creada por Edge Impulse.
En lugar de usar cajas delimitadoras (bounding boxes), produce un mapa de
calor de resolución reducida donde cada celda predice SI hay un objeto y de
qué clase.  Esto permite detectar múltiples objetos simultáneamente con un
modelo que cabe en pocos KB de RAM/Flash.

Diferencias clave respecto a train_mobilenet.py
───────────────────────────────────────────────
 • La salida NO es un vector de probabilidades por clase (clasificación global).
 • La salida ES un tensor 3-D: (H/stride, W/stride, num_classes+1)
   donde la última dimensión es un mapa de calor por clase + "fondo".
 • La función de pérdida combina Binary Crossentropy (presencia) +
   Categorical Crossentropy (clase) en forma de Focal Loss suave.
 • Los datos deben incluir anotaciones de bounding-box que se convierten a
   mapas de calor (heatmaps) como etiquetas.

Estructura de datos esperada
────────────────────────────
 data/processed/{W}x{H}/
     <clase>/
         imagen.jpg          ← imágenes (como en clasificación)
         imagen.json         ← anotación COCO-lite  (bounding boxes)

Formato JSON por imagen (COCO-lite simplificado):
{
  "bboxes": [
      {"label": "placa", "x": 10, "y": 20, "w": 30, "h": 25},
      ...
  ]
}

Si no hay archivo JSON para una imagen, se asume "sin objeto" (fondo).

Uso
───
python train_fomo.py --width 96 --height 96 --epochs 30

Argumentos opcionales:
  --width          Ancho de imagen (default: 96)
  --height         Alto de imagen (default: 96)
  --batch_size     Tamaño de batch (default: 32)
  --epochs         Épocas de entrenamiento (default: 30)
  --learning_rate  Tasa de aprendizaje Adam (default: 0.0001)
  --validation_split   Fracción de validación (default: 0.2)
  --alpha          Factor de ancho MobileNetV2 (default: 0.35)
  --output_stride  Stride total de la red (default: 8)
  --dropout_rate   Dropout antes de la cabeza final (default: 0.2)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib
matplotlib.use('Agg')          # Evita errores GUI en entornos sin display
import matplotlib.pyplot as plt
import os
import json
import datetime
import argparse
import glob


# ==========================================
# CONFIGURACIÓN GLOBAL
# ==========================================

LOG_DIR = "tensorboard_logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

os.makedirs("models/checkpoints", exist_ok=True)
os.makedirs("tensorboard_logs", exist_ok=True)


# ==========================================
# FOCAL LOSS (estabiliza entrenamiento FOMO)
# ==========================================

def focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    """
    Focal Loss binaria por celda del mapa de calor.
    Reduce el peso de ejemplos fáciles (fondo) para que el modelo
    se concentre en los objetos poco frecuentes.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: factor de enfoque (default 2.0)
        alpha: peso para la clase positiva (default 0.25)
    Returns:
        función de pérdida compilable por Keras
    """
    def _focal(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        bce = - (y_true * tf.math.log(y_pred) +
                 (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        p_t  = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        a_t  = y_true * alpha  + (1.0 - y_true) * (1.0 - alpha)
        fl   = a_t * tf.pow(1.0 - p_t, gamma) * bce
        return tf.reduce_mean(fl)
    _focal.__name__ = "focal_loss"
    return _focal


# ==========================================
# GENERADOR DE DATOS CON HEATMAPS
# ==========================================

class FOMODataGenerator(tf.keras.utils.Sequence):
    """
    Generador por lotes para FOMO.

    Por cada imagen:
      1. Carga la imagen redimensionada a (img_height, img_width).
      2. Lee el archivo JSON de anotaciones (si existe).
      3. Convierte cada bounding-box en un punto de calor gaussiano sobre
         una grilla de tamaño (img_height // output_stride,
                               img_width  // output_stride).
      4. Devuelve (imagen_normalizada, heatmap) donde heatmap tiene forma:
         (grid_h, grid_w, num_classes)   ← una capa por clase
         Los valores están en [0, 1]; la celda de la grilla que contiene
         el centro del bbox se pone a 1.0 (o distribución gaussiana suave).

    Args:
        image_paths : lista de rutas absolutas a imágenes
        labels      : lista de listas de dicts {'label': str, 'x','y','w','h'}
        class_names : lista ordenada de nombres de clase
        img_height  : alto de la imagen de entrada
        img_width   : ancho de la imagen de entrada
        output_stride: stride acumulado de la red (reduce resolución del heatmap)
        batch_size  : imágenes por batch
        augment     : aplicar augmentación de datos
        color_mode  : 'grayscale' o 'rgb'
    """

    def __init__(self, image_paths, labels, class_names,
                 img_height, img_width, output_stride,
                 batch_size=32, augment=False, color_mode='grayscale'):
        self.image_paths   = image_paths
        self.labels        = labels
        self.class_names   = class_names
        self.num_classes   = len(class_names)
        self.img_height    = img_height
        self.img_width     = img_width
        self.output_stride = output_stride
        self.grid_h        = img_height // output_stride
        self.grid_w        = img_width  // output_stride
        self.batch_size    = batch_size
        self.augment       = augment
        self.color_mode    = color_mode
        self.indices       = np.arange(len(image_paths))

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def on_epoch_end(self):
        """Mezcla los índices al final de cada época."""
        np.random.shuffle(self.indices)

    def _load_image(self, path):
        """Carga y normaliza una imagen a [0, 1]."""
        channels = 1 if self.color_mode == 'grayscale' else 3
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=channels, expand_animations=False)
        img = tf.image.resize(img, [self.img_height, self.img_width])
        img = tf.cast(img, tf.float32) / 255.0
        return img.numpy()

    def _augment_image(self, img):
        """Augmentación básica (flip horizontal + variación de brillo)."""
        if np.random.rand() > 0.5:
            img = np.fliplr(img)
        img = np.clip(img * np.random.uniform(0.8, 1.2), 0.0, 1.0)
        return img

    def _build_heatmap(self, bboxes, orig_w, orig_h):
        """
        Convierte lista de bboxes en un tensor heatmap de forma
        (grid_h, grid_w, num_classes).

        La celda de la grilla correspondiente al centro normalizado de
        cada bbox se establece a 1.0.  Si hay solapamiento se toma la
        clase más reciente (sin pérdida de información práctica a esta
        resolución).
        """
        heatmap = np.zeros((self.grid_h, self.grid_w, self.num_classes),
                           dtype=np.float32)
        for bbox in bboxes:
            label = bbox.get('label', '')
            if label not in self.class_names:
                continue
            cls_idx = self.class_names.index(label)

            # Centro normalizado del bbox (independiente del tamaño de imagen orig.)
            cx_norm = (bbox['x'] + bbox['w'] / 2.0) / orig_w
            cy_norm = (bbox['y'] + bbox['h'] / 2.0) / orig_h

            # Celda de la grilla
            gx = int(cx_norm * self.grid_w)
            gy = int(cy_norm * self.grid_h)
            gx = min(gx, self.grid_w  - 1)
            gy = min(gy, self.grid_h  - 1)

            heatmap[gy, gx, cls_idx] = 1.0
        return heatmap

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        images   = []
        heatmaps = []

        for i in batch_idx:
            path   = self.image_paths[i]
            bboxes = self.labels[i]

            # Dimensiones originales (necesitamos saber el tamaño de la imagen
            # original para normalizar coordenadas; leemos headers rápido)
            try:
                raw = tf.io.read_file(path)
                orig_img = tf.image.decode_image(raw, expand_animations=False)
                orig_h, orig_w = orig_img.shape[0], orig_img.shape[1]
            except Exception:
                orig_h, orig_w = self.img_height, self.img_width

            img     = self._load_image(path)
            heatmap = self._build_heatmap(bboxes, orig_w, orig_h)

            if self.augment:
                img = self._augment_image(img)

            images.append(img)
            heatmaps.append(heatmap)

        return np.array(images, dtype=np.float32), np.array(heatmaps, dtype=np.float32)


# ==========================================
# CARGA DE DATOS Y ANOTACIONES
# ==========================================

SUPPORTED_IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}

def _find_images(data_dir):
    """Recorre data_dir y devuelve lista de rutas de imagen."""
    paths = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in SUPPORTED_IMG_EXTS:
                paths.append(os.path.join(root, f))
    return sorted(paths)

def _load_annotations(image_path):
    """
    Busca un JSON de anotaciones con el mismo nombre base que la imagen.
    Si no existe, devuelve lista vacía (imagen de fondo / sin objetos).
    """
    base   = os.path.splitext(image_path)[0]
    json_p = base + '.json'
    if not os.path.exists(json_p):
        return []
    try:
        with open(json_p, 'r') as f:
            data = json.load(f)
        return data.get('bboxes', [])
    except (json.JSONDecodeError, KeyError):
        return []

def load_fomo_dataset(data_dir, validation_split=0.2):
    """
    Carga el dataset FOMO desde data_dir.

    Devuelve:
        train_paths, train_labels  : rutas e anotaciones de entrenamiento
        val_paths,   val_labels    : rutas e anotaciones de validación
        class_names                : lista de nombres de clase (orden alfabético)
    """
    if not os.path.exists(data_dir):
        print(f"Error: {data_dir} no existe.")
        return None, None, None, None, None

    all_paths = _find_images(data_dir)
    if not all_paths:
        print(f"Error: No se encontraron imágenes en {data_dir}")
        return None, None, None, None, None

    # Inferir clases desde nombres de subcarpeta
    class_names = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    if not class_names:
        print("Error: No se encontraron subcarpetas de clase.")
        return None, None, None, None, None

    print(f"Clases encontradas: {class_names}")
    print(f"Total imágenes    : {len(all_paths)}")

    # Anotaciones
    all_labels = [_load_annotations(p) for p in all_paths]

    # División train/val reproducible
    np.random.seed(42)
    idx_perm  = np.random.permutation(len(all_paths))
    split_at  = int(len(all_paths) * (1 - validation_split))

    train_idx = idx_perm[:split_at]
    val_idx   = idx_perm[split_at:]

    train_paths  = [all_paths[i]  for i in train_idx]
    train_labels = [all_labels[i] for i in train_idx]
    val_paths    = [all_paths[i]  for i in val_idx]
    val_labels   = [all_labels[i] for i in val_idx]

    print(f"  Entrenamiento: {len(train_paths)} imágenes")
    print(f"  Validación   : {len(val_paths)} imágenes")

    return train_paths, train_labels, val_paths, val_labels, class_names


# ==========================================
# ARQUITECTURA FOMO
# ==========================================

def build_fomo_model(num_classes, img_shape, alpha=0.35, output_stride=8,
                     dropout_rate=0.2, learning_rate=0.0001):
    """
    Construye un modelo FOMO basado en MobileNetV2 como backbone.

    ──────────────────────────────────────────────────────────────
    ARQUITECTURA FOMO
    ──────────────────────────────────────────────────────────────
    Entrada: (H, W, C)   [grayscale → replicado a 3 canales internamente]

    Backbone: MobileNetV2(alpha, include_top=False, weights='imagenet')
              Se trunca en el nivel con output_stride acumulado = 8 ó 16,
              lo que produce un mapa de características de tamaño H/8 × W/8.

    Cabeza de detección:
        Conv2D(num_classes, 1x1, padding='same') → BN → ReLU
        Conv2D(num_classes, 1x1, padding='same') → Sigmoid por clase

    Salida: (N, grid_h, grid_w, num_classes)
            Cada celda predice la probabilidad de presencia de cada clase.

    Pérdida: Focal Loss binaria (estabiliza desbalance fondo/objeto).

    Métricas: binary_accuracy (presencia de objeto en cada celda).
    ──────────────────────────────────────────────────────────────

    Args:
        num_classes    : número de clases de objetos
        img_shape      : tupla (H, W, C) — C puede ser 1 (grayscale) o 3
        alpha          : factor de ancho MobileNetV2 (0.35 → muy ligero)
        output_stride  : stride acumulado del backbone (8 ó 16)
        dropout_rate   : dropout en la cabeza de detección
        learning_rate  : tasa de aprendizaje Adam
    Returns:
        modelo Keras compilado
    """
    print(f"\nConstruyendo modelo FOMO:")
    print(f"  Input shape   : {img_shape}")
    print(f"  Num classes   : {num_classes}")
    print(f"  MobileNetV2 α : {alpha}")
    print(f"  Output stride : {output_stride}  (grilla = {img_shape[0]//output_stride}×{img_shape[1]//output_stride})")
    print(f"  Learning rate : {learning_rate}")

    # ── Entrada ──────────────────────────────────────────────────
    inputs = tf.keras.Input(shape=img_shape, name="fomo_input")
    x      = inputs

    # Grayscale → 3 canales (MobileNetV2 fue entrenado en RGB)
    if img_shape[-1] == 1:
        x = layers.Concatenate(axis=-1, name="gray2rgb")([x, x, x])

    # Normalización Edge Impulse style: [0,1] → [-1, 1]
    x = layers.Lambda(lambda t: t * 2.0 - 1.0, name="normalize")(x)

    # ── Backbone: MobileNetV2 sin cabeza ─────────────────────────
    # Seleccionamos la capa de salida según output_stride
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(img_shape[0], img_shape[1], 3),
        alpha=alpha,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False          # Congelado en fase 1

    # Nombre de capas de salida para stride 8 y 16:
    #   stride=8  → 'block_6_expand_relu'  (13×13 para 96×96 entrada)
    #   stride=16 → 'out_relu'             (3×3  para 96×96 entrada)
    # Calculamos la capa apropiada según el output_stride solicitado.
    stride_to_layer = {
        8 : 'block_6_expand_relu',   # Feature map H/8  × W/8
        16: 'block_13_expand_relu',  # Feature map H/16 × W/16
    }
    output_layer_name = stride_to_layer.get(
        output_stride, 'block_6_expand_relu'
    )

    # Si el nombre no existe en el modelo (varía con alpha), usamos el global
    layer_names = {l.name for l in base_model.layers}
    if output_layer_name not in layer_names:
        # Fallback: última capa convolucional disponible
        output_layer_name = base_model.layers[-1].name
        print(f"  ⚠ Capa '{output_layer_name}' no disponible, usando: {output_layer_name}")

    feature_extractor = tf.keras.Model(
        inputs  = base_model.input,
        outputs = base_model.get_layer(output_layer_name).output,
        name    = "fomo_backbone"
    )

    # Conectar backbone al grafo de entrada
    # (usamos la capa x que ya tiene 3 canales y está normalizada)
    backbone_input = tf.keras.Input(shape=(img_shape[0], img_shape[1], 3),
                                    name="backbone_input_tmp")
    features = feature_extractor(x, training=False)

    # ── Cabeza de detección ──────────────────────────────────────
    # Reducción de canales con punto de bottleneck
    feat = layers.Conv2D(
        max(16, num_classes * 4), (1, 1),
        padding='same', use_bias=False, name="head_conv1"
    )(features)
    feat = layers.BatchNormalization(name="head_bn1")(feat)
    feat = layers.ReLU(6.0, name="head_relu1")(feat)

    if dropout_rate > 0:
        feat = layers.Dropout(dropout_rate, name="head_dropout")(feat)

    # Capa de salida: num_classes canales, sigmoid por celda/clase
    outputs = layers.Conv2D(
        num_classes, (1, 1),
        padding='same',
        activation='sigmoid',
        name="fomo_output"
    )(feat)

    # ── Compilación ───────────────────────────────────────────────
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="FOMO")

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss      = focal_loss(gamma=2.0, alpha=0.25),
        metrics   = [tf.keras.metrics.BinaryAccuracy(name='binary_accuracy',
                                                      threshold=0.5)]
    )

    return model


# ==========================================
# FINE-TUNING (descongelar backbone)
# ==========================================

def unfreeze_backbone(model, learning_rate=1e-5):
    """
    Descongela el backbone para fine-tuning con LR muy bajo.
    Se llama después de la fase 1 de entrenamiento (solo cabeza).
    """
    print("\nDescongelando backbone para fine-tuning...")
    for layer in model.layers:
        if hasattr(layer, 'trainable'):
            layer.trainable = True

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss      = focal_loss(gamma=2.0, alpha=0.25),
        metrics   = [tf.keras.metrics.BinaryAccuracy(name='binary_accuracy',
                                                      threshold=0.5)]
    )
    model.summary()
    return model


# ==========================================
# VISUALIZACIÓN
# ==========================================

def plot_history(history_phase1, history_phase2, plot_path):
    """
    Genera y guarda la gráfica de entrenamiento (fase 1 + fase 2).
    """
    def _concat(h1, h2, key):
        v1 = h1.history.get(key, [])
        v2 = h2.history.get(key, []) if h2 else []
        return v1 + v2

    loss     = _concat(history_phase1, history_phase2, 'loss')
    val_loss = _concat(history_phase1, history_phase2, 'val_loss')
    acc      = _concat(history_phase1, history_phase2, 'binary_accuracy')
    val_acc  = _concat(history_phase1, history_phase2, 'val_binary_accuracy')
    ep_range = range(len(loss))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("FOMO — Historial de Entrenamiento", fontsize=14, fontweight='bold')

    ax1.plot(ep_range, acc,     label='Train Accuracy',      color='steelblue')
    ax1.plot(ep_range, val_acc, label='Validation Accuracy', color='tomato',
             linestyle='--')
    if history_phase2:
        phase2_start = len(history_phase1.history['loss'])
        ax1.axvline(phase2_start, color='gray', linestyle=':', label='Fine-tuning start')
    ax1.set_title('Binary Accuracy (por celda)')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Época')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(ep_range, loss,     label='Train Loss',      color='steelblue')
    ax2.plot(ep_range, val_loss, label='Validation Loss', color='tomato',
             linestyle='--')
    if history_phase2:
        ax2.axvline(phase2_start, color='gray', linestyle=':', label='Fine-tuning start')
    ax2.set_title('Focal Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Época')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Gráficas guardadas en: {plot_path}")


def visualize_heatmap_prediction(model, generator, output_dir, num_samples=4):
    """
    Guarda imágenes de muestra con el heatmap predicho superpuesto.
    Útil para verificar visualmente que el modelo detecta correctamente.
    """
    os.makedirs(output_dir, exist_ok=True)
    sample_idx = np.random.choice(len(generator.image_paths),
                                  size=min(num_samples, len(generator.image_paths)),
                                  replace=False)
    class_names = generator.class_names

    for k, i in enumerate(sample_idx):
        img_path = generator.image_paths[i]
        img_raw  = generator._load_image(img_path)
        img_in   = img_raw[np.newaxis, ...]           # (1, H, W, C)
        pred_hm  = model.predict(img_in, verbose=0)[0] # (grid_h, grid_w, num_cls)

        n_cls    = len(class_names)
        fig, axes = plt.subplots(1, n_cls + 1, figsize=(4 * (n_cls + 1), 4))
        fig.suptitle(os.path.basename(img_path), fontsize=9)

        # Imagen original
        axes[0].imshow(img_raw.squeeze(), cmap='gray')
        axes[0].set_title("Imagen")
        axes[0].axis('off')

        # Heatmap por clase
        for c, cls in enumerate(class_names):
            hm = pred_hm[..., c]
            axes[c + 1].imshow(hm, cmap='hot', vmin=0, vmax=1,
                               interpolation='nearest')
            axes[c + 1].set_title(f"Heatmap: {cls}")
            axes[c + 1].axis('off')

        out_path = os.path.join(output_dir, f"preview_{k:03d}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=120)
        plt.close()

    print(f"Visualizaciones guardadas en: {output_dir}")


# ==========================================
# FUNCIÓN PRINCIPAL
# ==========================================

def main(img_width, img_height, batch_size, epochs, learning_rate,
         validation_split, alpha, output_stride, dropout_rate,
         fine_tune_epochs, fine_tune_lr):

    model_name = "FOMO"
    data_dir   = f"data/processed/{img_width}x{img_height}"
    img_shape  = (img_height, img_width, 1)  # Escala de grises (como en MobileNet)

    plot_path = (
        f"tensorboard_logs/{model_name}_training_history"
        f"+{batch_size}+{epochs}+{learning_rate}+{validation_split}"
        f"+{img_width}x{img_height}.png"
    )
    checkpoint_path = (
        f"models/checkpoints/{model_name}"
        f"+{batch_size}+{epochs}+{learning_rate}+{validation_split}"
        f"+{img_width}x{img_height}.keras"
    )
    preview_dir = f"tensorboard_logs/{model_name}_previews"

    # ----------------------------------------------------------
    # 1. Cargar Dataset
    # ----------------------------------------------------------
    print("=" * 55)
    print(" FOMO — Cargando dataset")
    print("=" * 55)
    (train_paths, train_labels,
     val_paths,   val_labels,
     class_names) = load_fomo_dataset(data_dir, validation_split)

    if train_paths is None:
        print("\nNo se pudo cargar el dataset. Verifica la estructura de datos.")
        print("  Estructura esperada:")
        print(f"    {data_dir}/<clase>/imagen.jpg")
        print(f"    {data_dir}/<clase>/imagen.json  (opcional, bboxes)")
        return

    num_classes = len(class_names)
    grid_h = img_height // output_stride
    grid_w = img_width  // output_stride
    print(f"\nGrilla de salida: {grid_h}×{grid_w} celdas, {num_classes} clase(s)")

    # ----------------------------------------------------------
    # 2. Generadores de datos
    # ----------------------------------------------------------
    train_gen = FOMODataGenerator(
        train_paths, train_labels, class_names,
        img_height, img_width, output_stride,
        batch_size=batch_size, augment=True, color_mode='grayscale'
    )
    val_gen = FOMODataGenerator(
        val_paths, val_labels, class_names,
        img_height, img_width, output_stride,
        batch_size=batch_size, augment=False, color_mode='grayscale'
    )

    # ----------------------------------------------------------
    # 3. Construir Modelo FOMO
    # ----------------------------------------------------------
    print("\n" + "=" * 55)
    print(" FOMO — Construyendo arquitectura")
    print("=" * 55)
    model = build_fomo_model(
        num_classes   = num_classes,
        img_shape     = img_shape,
        alpha         = alpha,
        output_stride = output_stride,
        dropout_rate  = dropout_rate,
        learning_rate = learning_rate
    )
    model.summary()

    # ----------------------------------------------------------
    # 4. Callbacks
    # ----------------------------------------------------------
    callbacks_phase1 = [
        keras.callbacks.ModelCheckpoint(
            filepath     = checkpoint_path,
            monitor      = "val_binary_accuracy",
            save_best_only = True,
            mode         = "max",
            verbose      = 1
        ),
        keras.callbacks.EarlyStopping(
            monitor              = "val_loss",
            patience             = 7,
            restore_best_weights = True,
            verbose              = 1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor   = "val_loss",
            factor    = 0.5,
            patience  = 3,
            min_lr    = 1e-7,
            verbose   = 1
        ),
        keras.callbacks.TensorBoard(
            log_dir        = LOG_DIR + "_phase1",
            histogram_freq = 0
        )
    ]

    # ----------------------------------------------------------
    # 5. FASE 1: Entrenar solo la cabeza (backbone congelado)
    # ----------------------------------------------------------
    print("\n" + "=" * 55)
    print(" FASE 1 — Entrenando cabeza de detección (backbone congelado)")
    print("=" * 55)
    history_phase1 = None
    history_phase2 = None

    try:
        history_phase1 = model.fit(
            train_gen,
            epochs          = epochs,
            validation_data = val_gen,
            callbacks       = callbacks_phase1
        )
    except Exception as e:
        print(f"Error en Fase 1: {e}")
        return

    # ----------------------------------------------------------
    # 6. FASE 2: Fine-tuning (backbone descongelado, LR bajo)
    # ----------------------------------------------------------
    if fine_tune_epochs > 0:
        print("\n" + "=" * 55)
        print(f" FASE 2 — Fine-tuning ({fine_tune_epochs} épocas, LR={fine_tune_lr})")
        print("=" * 55)
        model = unfreeze_backbone(model, learning_rate=fine_tune_lr)

        checkpoint_ft = checkpoint_path.replace('.keras', '_finetuned.keras')
        callbacks_phase2 = [
            keras.callbacks.ModelCheckpoint(
                filepath       = checkpoint_ft,
                monitor        = "val_binary_accuracy",
                save_best_only = True,
                mode           = "max",
                verbose        = 1
            ),
            keras.callbacks.EarlyStopping(
                monitor              = "val_loss",
                patience             = 5,
                restore_best_weights = True,
                verbose              = 1
            ),
            keras.callbacks.TensorBoard(
                log_dir        = LOG_DIR + "_phase2",
                histogram_freq = 0
            )
        ]
        try:
            history_phase2 = model.fit(
                train_gen,
                epochs          = fine_tune_epochs,
                validation_data = val_gen,
                callbacks       = callbacks_phase2
            )
        except Exception as e:
            print(f"Error en Fase 2 (fine-tuning): {e}")
            # Continúa para guardar resultados de fase 1

    # ----------------------------------------------------------
    # 7. Visualización y guardado
    # ----------------------------------------------------------
    plot_history(history_phase1, history_phase2, plot_path)
    visualize_heatmap_prediction(model, val_gen, preview_dir, num_samples=4)

    print(f"\nModelo guardado en: {checkpoint_path}")

    with open("models/class_names.txt", "w") as f:
        for name in class_names:
            f.write(f"{name}\n")
    print("Nombres de clases guardados en: models/class_names.txt")

    # Resumen de parámetros útil para el paso de cuantización
    print("\n" + "=" * 55)
    print(" Resumen para cuantización / despliegue")
    print("=" * 55)
    total_params = model.count_params()
    print(f"  Parámetros totales : {total_params:,}")
    print(f"  Tamaño aprox. FP32 : {total_params * 4 / 1024:.1f} KB")
    print(f"  Tamaño aprox. INT8  : {total_params / 1024:.1f} KB")
    print(f"  Tamaño de entrada   : {img_height}×{img_width}×{img_shape[2]}")
    print(f"  Tamaño de salida    : {grid_h}×{grid_w}×{num_classes}")
    print(f"  Clases              : {class_names}")
    print("\nSiguiente paso:")
    print("  python src/quantize_int8_basic.py --path_model", checkpoint_path)


# ==========================================
# ENTRY POINT
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Entrena un modelo FOMO (Faster Objects, More Objects) "
            "para detección de objetos en TinyML."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--width", type=int, default=96,
        help="Ancho de imagen en píxeles"
    )
    parser.add_argument(
        "--height", type=int, default=96,
        help="Alto de imagen en píxeles"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Tamaño de batch"
    )
    parser.add_argument(
        "--epochs", type=int, default=30,
        help="Épocas de entrenamiento (Fase 1, backbone congelado)"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0001,
        help="Tasa de aprendizaje Adam (Fase 1)"
    )
    parser.add_argument(
        "--validation_split", type=float, default=0.2,
        help="Fracción del dataset para validación"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.35,
        help="Factor de ancho MobileNetV2 (0.35 = más ligero, 1.0 = más preciso)"
    )
    parser.add_argument(
        "--output_stride", type=int, default=8, choices=[8, 16],
        help="Stride total de la red (8 → grilla mayor, 16 → grilla menor)"
    )
    parser.add_argument(
        "--dropout_rate", type=float, default=0.2,
        help="Dropout en la cabeza de detección"
    )
    parser.add_argument(
        "--fine_tune_epochs", type=int, default=10,
        help="Épocas de fine-tuning (Fase 2, backbone descongelado). 0 = sin fine-tuning"
    )
    parser.add_argument(
        "--fine_tune_lr", type=float, default=1e-5,
        help="Tasa de aprendizaje para fine-tuning (Fase 2)"
    )

    args = parser.parse_args()

    main(
        img_width        = args.width,
        img_height       = args.height,
        batch_size       = args.batch_size,
        epochs           = args.epochs,
        learning_rate    = args.learning_rate,
        validation_split = args.validation_split,
        alpha            = args.alpha,
        output_stride    = args.output_stride,
        dropout_rate     = args.dropout_rate,
        fine_tune_epochs = args.fine_tune_epochs,
        fine_tune_lr     = args.fine_tune_lr,
    )
