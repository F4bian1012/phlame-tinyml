import os
import sys
import datetime

import artifacts
import cli
import train_common


def parse_args():
    parser = cli.ArgumentParser(
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
    parser.add_argument(
        "--tag", type=artifacts.valid_tag, default=None,
        help="Optional run name added after the architecture (e.g. the dataset: ResNet8_VWW+...), "
             "so runs on different datasets do not share file names"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace the checkpoint and the training curve if they already exist"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed of the initial weights, the shuffling and the augmentation (default: 42). On a CPU it repeats a run bit for bit on the same machine and software; on a GPU two runs with the same seed still differ, because some GPU kernels are not deterministic"
    )

    train_common.add_recipe_arguments(parser)
    return parser.parse_args()


if __name__ == "__main__":
    # The command line is read before TensorFlow is imported, which takes
    # several seconds: --help and a wrong option answer at once, and nothing is
    # created in the working directory.
    ARGS = parse_args()

import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402
from tensorflow.keras import layers  # noqa: E402
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

LOG_DIR = "tensorboard_logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

os.makedirs("models/checkpoints", exist_ok=True)
os.makedirs("tensorboard_logs", exist_ok=True)
os.makedirs("results/training_curves", exist_ok=True)


# Augmentation of the default recipe, as recorded in the training record.
STANDARD_AUGMENTATION = {"horizontal_flip": True, "rotation_fraction_of_turn": 0.15, "zoom_fraction": 0.1}


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


def build_resnet(input_shape, num_classes, dropout_rate=0.5, base_model="ResNet18", augmentation="standard"):
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
        augmentation: "standard" (rotation of 0.15 of a turn) or "light" (see train_common.py)
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

    # Data Augmentation (only active during training). The 'light' one is
    # applied to the batches by train_common.py, not by layers of the model.
    if augmentation != 'light':
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


def create_resnet_model(num_classes, img_shape, learning_rate, dropout_rate=0.5, base_model="ResNet18",
                        augmentation="standard"):
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

    model, loss_fn = build_resnet(img_shape, num_classes, dropout_rate, base_model, augmentation)

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


def main(img_width, img_height, batch_size, epochs, learning_rate, dropout_rate, base_model, splits_dir,
         tag=None, overwrite=False, seed=42, lr_schedule="constant", augmentation="standard"):
    model_name = base_model
    img_shape = (img_height, img_width, 1)
    plot_path = "results/training_curves/" + artifacts.training_curve_name(
        model_name, batch_size, epochs, learning_rate, img_width, img_height, tag)
    checkpoint_path = "models/checkpoints/" + artifacts.model_stem(
        model_name, batch_size, epochs, learning_rate, img_width, img_height, tag) + ".keras"

    # Checked before any work: a run on another dataset must not replace the
    # checkpoint or the training curve of a previous one.
    artifacts.refuse_to_overwrite(
        [checkpoint_path, plot_path], overwrite,
        hint="pass --tag NAME to give this run its own name (e.g. the dataset)")

    # One seed for Python, NumPy and TensorFlow, set before the data pipeline and
    # the model exist: same initial weights, same shuffling and same augmentation
    # draws on every run.
    keras.utils.set_random_seed(seed)

    train_ds, val_ds, class_names, data_info = train_common.load_splits(
        splits_dir, img_width, img_height, batch_size, seed, augmentation)

    num_classes = len(class_names)

    model = create_resnet_model(num_classes, img_shape, learning_rate, dropout_rate, base_model,
                                augmentation=augmentation)
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir=LOG_DIR,
            histogram_freq=1
        )
    ] + train_common.schedule_callbacks(lr_schedule, learning_rate, epochs, plateau=True)

    print(f"\nStarting training of {base_model}...")
    try:
        history = model.fit(
            train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=callbacks,
            verbose=train_common.fit_verbosity()
        )

        plot_history(history, plot_path, base_model)
        print(f"Model saved to: {checkpoint_path}")

        # The label order travels with the checkpoint, so runs on different
        # datasets do not share (and overwrite) a single class-name file.
        classes_path = artifacts.write_class_names(checkpoint_path, class_names)
        print(f"Class names saved to {classes_path}")

        record_path = artifacts.write_training_record(
            checkpoint_path, script=os.path.basename(__file__), architecture=base_model,
            tag=tag, seed=seed,
            settings={"width": img_width, "height": img_height, "batch_size": batch_size,
                      "epochs": epochs, "learning_rate": learning_rate,
                      "dropout_rate": dropout_rate,
                      **train_common.recipe_settings(lr_schedule, learning_rate, epochs, augmentation,
                                                     STANDARD_AUGMENTATION, plateau=True)},
            splits_dir=splits_dir, class_names=class_names,
            history=history.history, software=train_common.software_versions(), data=data_info)
        print(f"Training record (seed {seed}) saved to {record_path}")

    except Exception as e:
        print(f"Error during training: {e}")
        print("Tip: Make sure you have enough images per class (at least 10-20 to start).")
        sys.exit(1)


if __name__ == "__main__":
    args = ARGS

    main(
        args.width, args.height, args.batch_size, args.epochs,
        args.learning_rate, args.dropout_rate,
        args.base_model, args.splits_dir,
        tag=args.tag, overwrite=args.overwrite, seed=args.seed,
        lr_schedule=args.lr_schedule, augmentation=args.augmentation
    )
