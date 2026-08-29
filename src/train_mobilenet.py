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
    The partition is materialized on disk (instead of an in-memory
    validation_split) so that every level of the fidelity ladder
    MIL -> SIL -> PIL -> HIL evaluates exactly the same held-out images.
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
        print(f"Data Loading Error: {e}")
        print(f"Ensure each split has a subfolder per class, e.g., '{train_dir}/placa/image.jpg'.")
        return None, None, None

    class_names = train_ds.class_names
    print(f"Found classes: {class_names}")

    # Both splits must expose the same label order, or the argmax the firmware
    # returns would not mean the same thing in training and evaluation.
    if val_ds.class_names != class_names:
        print("ERROR: class order differs between the train and val splits.")
        print(f"  train: {class_names}")
        print(f"  val  : {val_ds.class_names}")
        sys.exit(1)
    
    if len(class_names) == 1:
        print("\nWARNING: Only 1 class found ('{}').".format(class_names[0]))
        print("   Training a classifier with ONLY positive samples will result in a model")
        print("   that predicts this class for EVERYTHING (Accuracy will be trivial).")
        print("   You should add a 'background' or 'negative' class folder with random images")
        print("   to teach the model what is NOT a '{}'.\n".format(class_names[0]))

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names

def create_mobilenet_model(num_classes, img_shape, learning_rate, base_model_name):
    """
    Creates a model using Transfer Learning based on the selected base_model.
    """
    print(f"Building {base_model_name} model for {num_classes} classes...")

    data_augmentation = keras.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.1),
    ])

    if base_model_name == 'MobileNetV2':
        base_model_class = tf.keras.applications.MobileNetV2
        preprocess_fn = tf.keras.applications.mobilenet_v2.preprocess_input
        alpha = 0.35
    elif base_model_name == 'MobileNetV3Large':
        base_model_class = tf.keras.applications.MobileNetV3Large
        preprocess_fn = tf.keras.applications.mobilenet_v3.preprocess_input
        alpha = 0.75
    elif base_model_name == 'MobileNetV3Small':
        base_model_class = tf.keras.applications.MobileNetV3Small
        preprocess_fn = tf.keras.applications.mobilenet_v3.preprocess_input
        alpha = 0.75
    elif base_model_name == 'MobileNet':
        base_model_class = tf.keras.applications.MobileNet
        preprocess_fn = tf.keras.applications.mobilenet.preprocess_input
        alpha = 0.25
    else:
        raise ValueError(f"Modelo base no soportado: {base_model_name}")

    base_input_shape = list(img_shape)
    if base_input_shape[-1] == 1:
        base_input_shape[-1] = 3
    base_input_shape = tuple(base_input_shape)

    base_model = base_model_class(
        input_shape=base_input_shape,
        include_top=False,
        weights='imagenet',
        alpha=alpha
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=img_shape)
    x = data_augmentation(inputs)
    
    if img_shape[-1] == 1:
        x = layers.Concatenate(axis=-1)([x, x, x])
        
    x = preprocess_fn(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    
    if num_classes == 1:
        print("Using Binary classification configuration (Sigmoid) for single class.")
        outputs = layers.Dense(1, activation='sigmoid')(x)
        loss_fn = 'binary_crossentropy'
    else:
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        loss_fn = 'sparse_categorical_crossentropy'
    
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=['accuracy']
    )
    
    return model

def plot_history(history, plot_path):
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
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig(plot_path)
    print(f"Training plots saved to {plot_path}")

def main(img_width, img_height, batch_size, epochs, learning_rate, base_model_name, splits_dir):
    img_shape = (img_height, img_width, 1)
    plot_path = f"results/training_curves/{base_model_name}_training_history+{batch_size}+{epochs}+{learning_rate}+{img_width}+{img_height}.png"
    checkpoint_path = f"models/checkpoints/{base_model_name}+{batch_size}+{epochs}+{learning_rate}+{img_width}+{img_height}.keras"

    train_ds, val_ds, class_names = load_custom_data(splits_dir, img_width, img_height, batch_size)
    
    if train_ds is None:
        return

    num_classes = len(class_names)

    model = create_mobilenet_model(num_classes, img_shape, learning_rate, base_model_name)
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
        )
    ]

    print("Starting training...")
    try:
        history = model.fit(
            train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=callbacks
        )

        plot_history(history, plot_path)
        print(f"Model saved to {checkpoint_path}")
        
        with open("models/class_names.txt", "w") as f:
            for name in class_names:
                f.write(f"{name}\n")
        print("Class names saved to models/class_names.txt")

    except Exception as e:
        print(f"Training failed: {e}")
        print("Tip: Ensure you have enough images per class (at least 10-20 to start).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model for TinyML")
    parser.add_argument("--base_model", type=str, default="MobileNetV2", choices=["MobileNet", "MobileNetV2", "MobileNetV3Large", "MobileNetV3Small"], help="Base model architecture")
    parser.add_argument("--width", type=int, default=160, help="Image width")
    parser.add_argument("--height", type=int, default=120, help="Image height")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--splits_dir", type=str, default="data/splits", help="Directory holding the train/val partitions produced by split_dataset.py")
    
    args = parser.parse_args()
    main(args.width, args.height, args.batch_size, args.epochs, args.learning_rate, args.base_model, args.splits_dir)
