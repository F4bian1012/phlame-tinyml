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


def residual_block(x, filters, stride=1, name_prefix="res"):
    """
    Basic residual block of ResNet18/34 (BasicBlock).

    Structure:
        Conv 3x3 -> BN -> ReLU -> Conv 3x3 -> BN -> (+shortcut) -> ReLU

    If stride > 1 or the number of channels changes, the shortcut applies
    a 1x1 projection (Conv + BN) to match dimensions.

    Args:
        x: input tensor
        filters: number of filters for the 3x3 convolutions
        stride: stride of the first convolution (controls downsampling)
        name_prefix: prefix used to name all layers of the block
    Returns:
        output tensor of the residual block
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
    Builds a ResNet adapted for TinyML / custom classification.

    ResNet18 architecture:
        Stem (Conv7x7/2 -> MaxPool3x3/2 -> BN -> ReLU; or Conv3x3/1 -> BN -> ReLU if H <= 64)
        -> Layer1: 2x BasicBlock(64,  stride=1)
        -> Layer2: BasicBlock(128, stride=2) + BasicBlock(128, stride=1)
        -> Layer3: BasicBlock(256, stride=2) + BasicBlock(256, stride=1)
        -> Layer4: BasicBlock(512, stride=2) + BasicBlock(512, stride=1)
        -> GlobalAveragePooling -> Dropout -> Dense(num_classes, Softmax/Sigmoid)

    ResNet8 architecture:
        Stem (Conv7x7/2 -> MaxPool3x3/2 -> BN -> ReLU; or Conv3x3/1 -> BN -> ReLU if H <= 64)
        -> Layer1: 1x BasicBlock(64,  stride=1)
        -> Layer2: 1x BasicBlock(128, stride=2)
        -> Layer3: 1x BasicBlock(256, stride=2)
        -> GlobalAveragePooling -> Dropout -> Dense(num_classes, Softmax/Sigmoid)

    In-graph preprocessing and small-input adaptation:
      - If input height (H) <= 64px, the stem uses Conv 3x3/1 without MaxPool
        instead of Conv 7x7/2 + MaxPool 3x3/2 to avoid collapsing the feature map.
      - Supports grayscale inputs (C=1), expanded to 3 channels inside the graph.
      - Pixel normalization [0, 255] -> [-1, 1] and Data Augmentation built into the model.

    Args:
        input_shape: tuple (H, W, C)
        num_classes: number of output classes
        dropout_rate: dropout rate before the classifier layer
        base_model: "ResNet8" or "ResNet18"
    Returns:
        (compiled model, loss function)
    """
    inputs = tf.keras.Input(shape=input_shape, name=f"input_{base_model.lower()}")

    # If the image is grayscale, expand it to 3 channels
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

    # ---------- Stem ----------
    # For small input heights (H <= 64px) use Conv 3x3/1 without MaxPool to preserve spatial resolution
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

    # ---------- Residual layers ----------
    if base_model == "ResNet8":
        # Layer 1: 1 BasicBlock, 64 filters, no downsampling
        x = residual_block(x, filters=64,  stride=1, name_prefix="layer1_block1")

        # Layer 2: 1 BasicBlock, 128 filters, downsampling (stride=2)
        x = residual_block(x, filters=128, stride=2, name_prefix="layer2_block1")

        # Layer 3: 1 BasicBlock, 256 filters, downsampling (stride=2)
        x = residual_block(x, filters=256, stride=2, name_prefix="layer3_block1")
    else: # ResNet18
        # Layer 1: 2 BasicBlocks, 64 filters, no downsampling
        x = residual_block(x, filters=64,  stride=1, name_prefix="layer1_block1")
        x = residual_block(x, filters=64,  stride=1, name_prefix="layer1_block2")

        # Layer 2: 2 BasicBlocks, 128 filters, downsampling (stride=2)
        x = residual_block(x, filters=128, stride=2, name_prefix="layer2_block1")
        x = residual_block(x, filters=128, stride=1, name_prefix="layer2_block2")

        # Layer 3: 2 BasicBlocks, 256 filters, downsampling (stride=2)
        x = residual_block(x, filters=256, stride=2, name_prefix="layer3_block1")
        x = residual_block(x, filters=256, stride=1, name_prefix="layer3_block2")

        # Layer 4: 2 BasicBlocks, 512 filters, downsampling (stride=2)
        x = residual_block(x, filters=512, stride=2, name_prefix="layer4_block1")
        x = residual_block(x, filters=512, stride=1, name_prefix="layer4_block2")

    # ---------- Classification head ----------
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)

    if num_classes == 1:
        outputs = layers.Dense(1, activation='sigmoid', name="output_sigmoid")(x)
        loss_fn = 'binary_crossentropy'
        print("Binary configuration (Sigmoid) for a single class.")
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
    Wrapper that builds, compiles and returns the ResNet model.

    Args:
        num_classes: number of classes detected in the dataset
        img_shape: tuple (H, W, C)
        learning_rate: learning rate for Adam
        dropout_rate: dropout rate before the classifier layer
        base_model: base architecture ("ResNet8" or "ResNet18")
    Returns:
        compiled model
    """
    print(f"Building {base_model} for {num_classes} class(es)...")
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
    plt.title(f'Training and Validation Accuracy - {base_model}')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title(f'Training and Validation Loss - {base_model}')

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Training plots saved to: {plot_path}")


def main(img_width, img_height, batch_size, epochs, learning_rate, dropout_rate, base_model, splits_dir):
    model_name = base_model
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

    print(f"\nStarting training of {base_model}...")
    try:
        history = model.fit(
            train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=callbacks
        )

        plot_history(history, plot_path, base_model)
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
        description="ResNet training for TinyML (grayscale images)"
    )
    parser.add_argument(
        "--base_model", type=str, default="ResNet18", choices=["ResNet8", "ResNet18"],
        help="Base architecture (ResNet8 or ResNet18) (default: ResNet18)"
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
        args.width, args.height, args.batch_size, args.epochs, 
        args.learning_rate, args.dropout_rate, 
        args.base_model, args.splits_dir
    )
