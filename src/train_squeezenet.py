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
os.makedirs("results/training_curves", exist_ok=True)


def load_custom_data(splits_dir, img_width, img_height, batch_size):
    """
    Loads the train/val partitions written to disk by split_dataset.py.
    Expected structure:
        data/splits/
            train/
                class_a/
                    img1.jpg
            val/
                class_a/
                    img2.jpg
    The partition lives on disk (instead of an in-memory validation_split) so
    that the four levels of the MIL -> SIL -> PIL -> HIL ladder evaluate
    exactly the same held-out images.
    Supports grayscale images (color_mode='grayscale').
    """
    train_dir = os.path.join(splits_dir, "train")
    val_dir = os.path.join(splits_dir, "val")

    for path in (train_dir, val_dir):
        if not os.path.isdir(path) or not os.listdir(path):
            print(f"ERROR: {path} does not exist or is empty.")
            print("Run the dataset partition first:")
            print(f"  python src/split_dataset.py"
                  f" --input_dir data/processed/{img_width}x{img_height}"
                  f" --output_dir {splits_dir}")
            sys.exit(1)

    print(f"Loading data from {splits_dir}...")

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
        print(f"Error loading data: {e}")
        print(f"Make sure you have one subfolder per class, e.g. '{train_dir}/class_a/image.jpg'.")
        return None, None, None

    class_names = train_ds.class_names
    print(f"Classes found: {class_names}")

    # Both splits must expose the same label order: otherwise the argmax
    # returned by the firmware would not mean the same thing in training and
    # in evaluation.
    if val_ds.class_names != class_names:
        print("ERROR: class order differs between the splits.")
        print(f"  train: {class_names}")
        print(f"  val  : {val_ds.class_names}")
        sys.exit(1)

    if len(class_names) == 1:
        print(f"\nWARNING: Only 1 class found ('{class_names[0]}').")
        print("   Training a classifier with only positive samples will result in a model")
        print("   that predicts this class for ALL inputs (trivial accuracy).")
        print("   Add a 'background' or 'negative' class folder with random images\n")

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names


def fire_module(x, squeeze_filters, expand_filters, name_prefix):
    """
    SqueezeNet Fire module:
      - Squeeze layer: Conv 1x1 (reduces channels)
      - Expand layer: Conv 1x1 + Conv 3x3 concatenated (increases representation)
    
    Args:
        x: input tensor
        squeeze_filters: number of 1x1 filters in the squeeze layer
        expand_filters: number of filters for EACH expand branch (1x1 and 3x3)
        name_prefix: prefix used to name the layers
    Returns:
        output tensor of the fire module
    """
    x = layers.Conv2D(
        squeeze_filters, (1, 1),
        activation='relu',
        padding='same',
        name=f"{name_prefix}_squeeze"
    )(x)

    expand_1x1 = layers.Conv2D(
        expand_filters, (1, 1),
        activation='relu',
        padding='same',
        name=f"{name_prefix}_expand1x1"
    )(x)

    expand_3x3 = layers.Conv2D(
        expand_filters, (3, 3),
        activation='relu',
        padding='same',
        name=f"{name_prefix}_expand3x3"
    )(x)

    x = layers.Concatenate(axis=-1, name=f"{name_prefix}_concat")([expand_1x1, expand_3x3])
    return x


def build_squeezenet(input_shape, num_classes, dropout_rate=0.5):
    """
    Builds SqueezeNet 1.1 adapted for TinyML / custom classification.

    Differences from SqueezeNet 1.0:
      - Fewer parameters (50% fewer MACs than v1.0)
      - Compatible with small inputs (e.g. 96x96)
      
    Architecture:
        Conv2D -> MaxPool -> Fire(2) -> Fire(3) -> MaxPool -> Fire(4) -> Fire(5) ->
        MaxPool -> Fire(6) -> Fire(7) -> Fire(8) -> Fire(9) -> Dropout -> Conv2D -> GAP -> Softmax/Sigmoid

    Args:
        input_shape: tuple (H, W, C), accepts grayscale (C=1) or RGB (C=3)
        num_classes: number of output classes
        dropout_rate: dropout rate before the final classification layer
    Returns:
        (compiled Keras model, loss function)
    """
    inputs = tf.keras.Input(shape=input_shape, name="input_squeezenet")

    # If the image is grayscale, expand it to 3 channels
    # to stay compatible with the architecture's filters
    if input_shape[-1] == 1:
        x = layers.Concatenate(axis=-1, name="grayscale_to_rgb")([inputs, inputs, inputs])
    else:
        x = inputs

    # Normalization: scale pixels [0, 255] -> [-1, 1]
    x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="normalization")(x)

    # Data Augmentation (only active during training)
    x = layers.RandomFlip('horizontal', name="aug_flip")(x)
    x = layers.RandomRotation(0.15, name="aug_rotation")(x)
    x = layers.RandomZoom(0.1, name="aug_zoom")(x)

    # ---------- SqueezeNet 1.1 backbone ----------

    # Initial block
    x = layers.Conv2D(64, (3, 3), strides=(2, 2), activation='relu',
                      padding='same', name="conv1")(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name="pool1")(x)

    # Fire modules 2-3
    x = fire_module(x, squeeze_filters=16, expand_filters=64, name_prefix="fire2")
    x = fire_module(x, squeeze_filters=16, expand_filters=64, name_prefix="fire3")
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name="pool3")(x)

    # Fire modules 4-5
    x = fire_module(x, squeeze_filters=32, expand_filters=128, name_prefix="fire4")
    x = fire_module(x, squeeze_filters=32, expand_filters=128, name_prefix="fire5")
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name="pool5")(x)

    # Fire modules 6-9
    x = fire_module(x, squeeze_filters=48, expand_filters=192, name_prefix="fire6")
    x = fire_module(x, squeeze_filters=48, expand_filters=192, name_prefix="fire7")
    x = fire_module(x, squeeze_filters=64, expand_filters=256, name_prefix="fire8")
    x = fire_module(x, squeeze_filters=64, expand_filters=256, name_prefix="fire9")

    # ---------- Classification head ----------
    x = layers.Dropout(dropout_rate, name="dropout")(x)

    # Final conv -> Global Average Pooling (replaces Flatten + Dense, ~0 extra params)
    if num_classes == 1:
        x = layers.Conv2D(1, (1, 1), activation='relu', padding='same', name="conv10")(x)
        x = layers.GlobalAveragePooling2D(name="gap")(x)
        outputs = layers.Activation('sigmoid', name="output_sigmoid")(x)
        loss_fn = 'binary_crossentropy'
        print("Binary configuration (Sigmoid) for a single class.")
    else:
        x = layers.Conv2D(num_classes, (1, 1), activation='relu', padding='same', name="conv10")(x)
        x = layers.GlobalAveragePooling2D(name="gap")(x)
        outputs = layers.Activation('softmax', name="output_softmax")(x)
        loss_fn = 'sparse_categorical_crossentropy'

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="SqueezeNet_1_1")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=loss_fn,
        metrics=['accuracy']
    )

    return model, loss_fn


def create_squeezenet_model(num_classes, img_shape, learning_rate, dropout_rate=0.5):
    """
    Wrapper that builds, compiles and returns the SqueezeNet model.

    Args:
        num_classes: number of classes detected in the dataset
        img_shape: tuple (H, W, C)
        learning_rate: learning rate for Adam
        dropout_rate: dropout rate
    Returns:
        compiled model
    """
    print(f"Building SqueezeNet 1.1 for {num_classes} class(es)...")
    print(f"  Input shape : {img_shape}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Dropout rate: {dropout_rate}")

    model, loss_fn = build_squeezenet(img_shape, num_classes, dropout_rate)

    if num_classes == 1:
        loss = 'binary_crossentropy'
    else:
        loss = 'sparse_categorical_crossentropy'

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=['accuracy']
    )

    return model


def plot_history(history, plot_path):
    """Saves the training accuracy and loss plot."""
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
    plt.title('Training and Validation Accuracy - SqueezeNet')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss - SqueezeNet')

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Training plots saved to: {plot_path}")


def main(img_width, img_height, batch_size, epochs, learning_rate, dropout_rate, splits_dir):
    model_name = "SqueezeNet"
    img_shape = (img_height, img_width, 1)
    plot_path = (
        f"results/training_curves/{model_name}_training_history"
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

    model = create_squeezenet_model(num_classes, img_shape, learning_rate, dropout_rate)
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

    print("\nStarting training of SqueezeNet...")
    try:
        history = model.fit(
            train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=callbacks
        )

        plot_history(history, plot_path)
        print(f"Model saved to: {checkpoint_path}")

        with open("models/class_names.txt", "w") as f:
            for name in class_names:
                f.write(f"{name}\n")
        print("Class names saved to models/class_names.txt")

    except Exception as e:
        print(f"Error during training: {e}")
        print("Tip: Make sure you have enough images per class (at least 10-20 to start).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SqueezeNet 1.1 training for TinyML (grayscale images)"
    )
    parser.add_argument(
        "--width", type=int, default=160,
        help="Image width in pixels (default: 160)"
    )
    parser.add_argument(
        "--height", type=int, default=120,
        help="Image height in pixels (default: 120)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--epochs", type=int, default=30,
        help="Number of epochs (default: 30)"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0001,
        help="Adam learning rate (default: 0.0001)"
    )
    parser.add_argument(
        "--splits_dir", type=str, default="data/splits",
        help="Directory holding the train/val partitions produced by split_dataset.py (default: data/splits)"
    )
    parser.add_argument(
        "--dropout_rate", type=float, default=0.5,
        help="Dropout rate before the final layer (default: 0.5)"
    )

    args = parser.parse_args()
    main(
        args.width, args.height,
        args.batch_size, args.epochs,
        args.learning_rate,
        args.dropout_rate, args.splits_dir
    )
