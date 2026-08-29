import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import datetime
import argparse

LOG_DIR = "tensorboard_logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

os.makedirs("models/checkpoints", exist_ok=True)
os.makedirs("tensorboard_logs", exist_ok=True)


def load_custom_data(splits_dir, img_width, img_height, batch_size):
    """
    Carga las particiones train/val que split_dataset.py deja en disco.
    Estructura esperada:
        data/splits/
            train/
                clase_a/
                    img1.jpg
            val/
                clase_a/
                    img2.jpg
    La partición vive en disco (en vez de un validation_split en memoria) para
    que los cuatro niveles de la escalera MIL -> SIL -> PIL -> HIL evalúen
    exactamente las mismas imágenes reservadas.
    Soporta imágenes en escala de grises (color_mode='grayscale').
    """
    train_dir = os.path.join(splits_dir, "train")
    val_dir = os.path.join(splits_dir, "val")

    for path in (train_dir, val_dir):
        if not os.path.isdir(path) or not os.listdir(path):
            print(f"ERROR: {path} no existe o está vacío.")
            print("Ejecuta primero la partición del dataset:")
            print(f"  python src/split_dataset.py"
                  f" --input_dir data/processed/{img_width}x{img_height}"
                  f" --output_dir {splits_dir}")
            sys.exit(1)

    print(f"Cargando datos desde {splits_dir}...")

    try:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            color_mode='grayscale',
            image_size=(img_height, img_width),
            batch_size=batch_size
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_dir,
            color_mode='grayscale',
            image_size=(img_height, img_width),
            batch_size=batch_size,
            shuffle=False
        )
    except ValueError as e:
        print(f"Error al cargar datos: {e}")
        print(f"Asegúrate de tener subcarpetas por clase, ej: '{train_dir}/placa/imagen.jpg'.")
        return None, None, None

    class_names = train_ds.class_names
    print(f"Clases encontradas: {class_names}")

    # Ambas particiones deben exponer el mismo orden de etiquetas: si no, el
    # argmax que devuelve el firmware no significaría lo mismo al entrenar y
    # al evaluar.
    if val_ds.class_names != class_names:
        print("ERROR: el orden de clases difiere entre las particiones.")
        print(f"  train: {class_names}")
        print(f"  val  : {val_ds.class_names}")
        sys.exit(1)

    if len(class_names) == 1:
        print(f"\nADVERTENCIA: Solo se encontró 1 clase ('{class_names[0]}').")
        print("   Entrenar un clasificador con solo muestras positivas resultará en un modelo")
        print("   que predice esta clase para TODOS los inputs (accuracy trivial).")
        print(f"   Agrega una carpeta de clase 'background' o 'negativo' con imágenes aleatorias\n")

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names


def residual_block(x, filters, stride=1, name_prefix="res"):
    """
    Bloque residual básico de ResNet18/34 (BasicBlock).

    Estructura:
        Conv 3x3 → BN → ReLU → Conv 3x3 → BN → (+shortcut) → ReLU

    Si stride > 1 o los canales cambian, el shortcut aplica
    una proyección 1x1 (Conv + BN) para igualar dimensiones.

    Args:
        x: tensor de entrada
        filters: número de filtros para las convoluciones 3x3
        stride: stride de la primera convolución (controla downsampling)
        name_prefix: prefijo para nombrar todas las capas del bloque
    Returns:
        tensor de salida del bloque residual
    """
    shortcut = x
    in_channels = x.shape[-1]

    x = layers.Conv2D(
        filters, (3, 3),
        strides=stride,
        padding='same',
        use_bias=False,
        name=f"{name_prefix}_conv1"
    )(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = layers.ReLU(name=f"{name_prefix}_relu1")(x)

    x = layers.Conv2D(
        filters, (3, 3),
        strides=1,
        padding='same',
        use_bias=False,
        name=f"{name_prefix}_conv2"
    )(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)

    if stride != 1 or in_channels != filters:
        shortcut = layers.Conv2D(
            filters, (1, 1),
            strides=stride,
            padding='same',
            use_bias=False,
            name=f"{name_prefix}_shortcut_conv"
        )(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name_prefix}_shortcut_bn")(shortcut)

    x = layers.Add(name=f"{name_prefix}_add")([x, shortcut])
    x = layers.ReLU(name=f"{name_prefix}_relu2")(x)

    return x


def build_resnet(input_shape, num_classes, dropout_rate=0.5, base_model="ResNet18"):
    """
    Construye ResNet adaptado para TinyML / clasificación personalizada.

    Arquitectura ResNet18:
        Conv7x7/2 → BN → ReLU → MaxPool/2
        → Layer1: 2x BasicBlock(64,  stride=1)
        → Layer2: 2x BasicBlock(128, stride=2)
        → Layer3: 2x BasicBlock(256, stride=2)
        → Layer4: 2x BasicBlock(512, stride=2)
        → GlobalAveragePooling → Dense(num_classes)

    Arquitectura ResNet8:
        Conv7x7/2 → BN → ReLU → MaxPool/2
        → Layer1: 1x BasicBlock(64,  stride=1)
        → Layer2: 1x BasicBlock(128, stride=2)
        → Layer3: 1x BasicBlock(256, stride=2)
        → GlobalAveragePooling → Dense(num_classes)

    Adaptaciones para imágenes pequeñas (< 96x96):
      - Si la resolución de entrada es <= 64px, el stem usa Conv 3x3/1
        en lugar de Conv 7x7/2 para evitar colapsar el mapa de características.
      - Soporta entradas en escala de grises (C=1), expandidas a 3 canales.
      - Añade Data Augmentation integrada en el modelo.
      - Normalización de píxeles [0, 255] → [-1, 1] dentro del grafo.

    Args:
        input_shape: tupla (H, W, C)
        num_classes: número de clases de salida
        dropout_rate: tasa de dropout antes de la capa clasificadora
        base_model: "ResNet8" o "ResNet18"
    Returns:
        (modelo compilado, función de pérdida)
    """
    inputs = tf.keras.Input(shape=input_shape, name=f"input_{base_model.lower()}")

    # Si la imagen es en escala de grises, expande a 3 canales
    if input_shape[-1] == 1:
        x = layers.Concatenate(axis=-1, name="grayscale_to_rgb")([inputs, inputs, inputs])
    else:
        x = inputs

    # Normalización: escalar píxeles [0, 255] → [-1, 1]
    x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="normalization")(x)

    # Data Augmentation (solo activa en entrenamiento)
    x = layers.RandomFlip('horizontal', name="aug_flip")(x)
    x = layers.RandomRotation(0.15, name="aug_rotation")(x)
    x = layers.RandomZoom(0.1, name="aug_zoom")(x)

    # ---------- Stem ----------
    # Para entradas pequeñas (≤ 64px) usamos Conv 3x3/1 para no perder resolución
    if input_shape[0] <= 64:
        x = layers.Conv2D(
            64, (3, 3), strides=(1, 1),
            padding='same', use_bias=False, name="stem_conv"
        )(x)
    else:
        x = layers.Conv2D(
            64, (7, 7), strides=(2, 2),
            padding='same', use_bias=False, name="stem_conv"
        )(x)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name="stem_pool")(x)

    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)

    # ---------- Capas residuales ----------
    if base_model == "ResNet8":
        # Layer 1: 1 BasicBlock, 64 filtros, sin downsampling
        x = residual_block(x, filters=64,  stride=1, name_prefix="layer1_block1")

        # Layer 2: 1 BasicBlock, 128 filtros, downsampling (stride=2)
        x = residual_block(x, filters=128, stride=2, name_prefix="layer2_block1")

        # Layer 3: 1 BasicBlock, 256 filtros, downsampling (stride=2)
        x = residual_block(x, filters=256, stride=2, name_prefix="layer3_block1")
    else: # ResNet18
        # Layer 1: 2 BasicBlocks, 64 filtros, sin downsampling
        x = residual_block(x, filters=64,  stride=1, name_prefix="layer1_block1")
        x = residual_block(x, filters=64,  stride=1, name_prefix="layer1_block2")

        # Layer 2: 2 BasicBlocks, 128 filtros, downsampling (stride=2)
        x = residual_block(x, filters=128, stride=2, name_prefix="layer2_block1")
        x = residual_block(x, filters=128, stride=1, name_prefix="layer2_block2")

        # Layer 3: 2 BasicBlocks, 256 filtros, downsampling (stride=2)
        x = residual_block(x, filters=256, stride=2, name_prefix="layer3_block1")
        x = residual_block(x, filters=256, stride=1, name_prefix="layer3_block2")

        # Layer 4: 2 BasicBlocks, 512 filtros, downsampling (stride=2)
        x = residual_block(x, filters=512, stride=2, name_prefix="layer4_block1")
        x = residual_block(x, filters=512, stride=1, name_prefix="layer4_block2")

    # ---------- Cabeza de clasificación ----------
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)

    if num_classes == 1:
        outputs = layers.Dense(1, activation='sigmoid', name="output_sigmoid")(x)
        loss_fn = 'binary_crossentropy'
        print("Configuración binaria (Sigmoid) para clase única.")
    else:
        outputs = layers.Dense(num_classes, activation='softmax', name="output_softmax")(x)
        loss_fn = 'sparse_categorical_crossentropy'

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=base_model)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=loss_fn,
        metrics=['accuracy']
    )

    return model, loss_fn


def create_resnet_model(num_classes, img_shape, learning_rate, dropout_rate=0.5, base_model="ResNet18"):
    """
    Wrapper que construye, compila y devuelve el modelo ResNet.

    Args:
        num_classes: número de clases detectadas en el dataset
        img_shape: tupla (H, W, C)
        learning_rate: tasa de aprendizaje para Adam
        dropout_rate: tasa de dropout antes de la capa clasificadora
        base_model: arquitectura base ("ResNet8" o "ResNet18")
    Returns:
        modelo compilado
    """
    print(f"Construyendo {base_model} para {num_classes} clase(s)...")
    print(f"  Input shape  : {img_shape}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Dropout rate : {dropout_rate}")

    model, loss_fn = build_resnet(img_shape, num_classes, dropout_rate, base_model)

    loss = 'binary_crossentropy' if num_classes == 1 else 'sparse_categorical_crossentropy'

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=['accuracy']
    )

    return model


def plot_history(history, plot_path, base_model="ResNet18"):
    """Guarda gráfico de accuracy y loss del entrenamiento."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title(f'Training and Validation Accuracy - {base_model}')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title(f'Training and Validation Loss - {base_model}')

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Gráficas de entrenamiento guardadas en: {plot_path}")


def main(img_width, img_height, batch_size, epochs, learning_rate, dropout_rate, base_model, splits_dir):
    model_name = base_model
    img_shape = (img_height, img_width, 1)
    plot_path = (
        f"tensorboard_logs/{model_name}_training_history"
        f"+{batch_size}+{epochs}+{learning_rate}"
        f"+{img_width}+{img_height}.png"
    )
    checkpoint_path = (
        f"models/checkpoints/{model_name}"
        f"+{batch_size}+{epochs}+{learning_rate}"
        f"+{img_width}+{img_height}.keras"
    )

    train_ds, val_ds, class_names = load_custom_data(
        splits_dir, img_width, img_height, batch_size
    )
    if train_ds is None:
        return

    num_classes = len(class_names)

    model = create_resnet_model(num_classes, img_shape, learning_rate, dropout_rate, base_model)
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir=LOG_DIR,
            histogram_freq=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]

    print(f"\nIniciando entrenamiento de {base_model}...")
    try:
        history = model.fit(
            train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=callbacks
        )

        plot_history(history, plot_path, base_model)
        print(f"Modelo guardado en: {checkpoint_path}")

        with open("models/class_names.txt", "w") as f:
            for name in class_names:
                f.write(f"{name}\n")
        print("Nombres de clases guardados en models/class_names.txt")

    except Exception as e:
        print(f"Error durante el entrenamiento: {e}")
        print("Tip: Asegúrate de tener suficientes imágenes por clase (mínimo 10-20 para comenzar).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entrenamiento de ResNet para TinyML (imágenes en escala de grises)"
    )
    parser.add_argument(
        "--base_model", type=str, default="ResNet18", choices=["ResNet8", "ResNet18"],
        help="Arquitectura base (ResNet8 o ResNet18) (default: ResNet18)"
    )
    parser.add_argument(
        "--width", type=int, default=160,
        help="Ancho de imagen en píxeles (default: 160)"
    )
    parser.add_argument(
        "--height", type=int, default=120,
        help="Alto de imagen en píxeles (default: 120)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Tamaño de batch (default: 32)"
    )
    parser.add_argument(
        "--epochs", type=int, default=30,
        help="Número de épocas (default: 30)"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0001,
        help="Tasa de aprendizaje Adam (default: 0.0001)"
    )
    parser.add_argument(
        "--splits_dir", type=str, default="data/splits",
        help="Directorio con las particiones train/val de split_dataset.py (default: data/splits)"
    )
    parser.add_argument(
        "--dropout_rate", type=float, default=0.5,
        help="Tasa de dropout antes de la capa final (default: 0.5)"
    )

    args = parser.parse_args()
    
    main(
        args.width, args.height, args.batch_size, args.epochs, 
        args.learning_rate, args.dropout_rate, 
        args.base_model, args.splits_dir
    )
