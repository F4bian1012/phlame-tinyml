import os
import sys
import datetime

import artifacts
import cli
import train_common


def parse_args():
    parser = cli.ArgumentParser(
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
    parser.add_argument(
        "--tag", type=artifacts.valid_tag, default=None,
        help="Optional run name added after the architecture (e.g. the dataset: SqueezeNet_VWW+...), "
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
    parser.add_argument(
        "--kernel_initializer", choices=["glorot_uniform", "he_normal"], default="glorot_uniform",
        help="Starting weights of the convolutions. glorot_uniform (default) is what the published models "
             "used. he_normal is scaled for ReLU layers: try it when the network stays at a constant "
             "output (accuracy 1/classes); the layers do not change"
    )
    parser.add_argument(
        "--batch_norm", action="store_true",
        help="Batch normalisation between every convolution of the backbone and its ReLU. A different "
             "network to train than the published one (the converter folds it into the convolutions)"
    )
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


def conv_relu(x, filters, kernel, name, kernel_initializer="glorot_uniform", batch_norm=False, **kwargs):
    """Convolution followed by ReLU; with batch_norm, Conv -> BatchNormalization -> ReLU."""
    if not batch_norm:
        return layers.Conv2D(filters, kernel, activation='relu', padding='same',
                             kernel_initializer=kernel_initializer, name=name, **kwargs)(x)
    x = layers.Conv2D(filters, kernel, padding='same', use_bias=False,
                      kernel_initializer=kernel_initializer, name=name, **kwargs)(x)
    x = layers.BatchNormalization(name=name + "_bn")(x)
    return layers.ReLU(name=name + "_relu")(x)


def fire_module(x, squeeze_filters, expand_filters, name_prefix,
                kernel_initializer="glorot_uniform", batch_norm=False):
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
    x = conv_relu(x, squeeze_filters, (1, 1), f"{name_prefix}_squeeze", kernel_initializer, batch_norm)
    expand_1x1 = conv_relu(x, expand_filters, (1, 1), f"{name_prefix}_expand1x1", kernel_initializer, batch_norm)
    expand_3x3 = conv_relu(x, expand_filters, (3, 3), f"{name_prefix}_expand3x3", kernel_initializer, batch_norm)

    x = layers.Concatenate(axis=-1, name=f"{name_prefix}_concat")([expand_1x1, expand_3x3])
    return x


def build_squeezenet(input_shape, num_classes, dropout_rate=0.5, augmentation="standard",
                     kernel_initializer="glorot_uniform", batch_norm=False):
    """
    Builds SqueezeNet 1.1 adapted for TinyML / custom classification.

    Differences from SqueezeNet 1.0:
      - Fewer parameters (50% fewer MACs than v1.0)
      - Compatible with small inputs (e.g. 96x96)

    Architecture:
        Conv2D -> MaxPool -> Fire(2) -> Fire(3) -> MaxPool -> Fire(4) -> Fire(5) ->
        MaxPool -> Fire(6) -> Fire(7) -> Fire(8) -> Fire(9) -> Dropout -> Conv2D -> GAP -> Softmax/Sigmoid

    The defaults build the network of the published models, which has no
    normalisation layer. On some datasets it does not start to learn: on COCO
    person detection it stayed at a constant output (accuracy 0.500) at every
    learning rate tried. Two options are offered for that case (the
    measurements are in docs/scripts.md):

      kernel_initializer="he_normal"   starting weights scaled for ReLU layers;
                                       the layers are the same
      batch_norm=True                  batch normalisation between every
                                       convolution of the backbone and its
                                       ReLU: a different network to train

    Args:
        input_shape: tuple (H, W, C), accepts grayscale (C=1) or RGB (C=3)
        num_classes: number of output classes
        dropout_rate: dropout rate before the final classification layer
        augmentation: "standard" (rotation of 0.15 of a turn) or "light" (see train_common.py)
        kernel_initializer: initialiser of every convolution
        batch_norm: Conv -> BatchNormalization -> ReLU in the backbone
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

    # Data Augmentation (only active during training). The 'light' one is
    # applied to the batches by train_common.py, not by layers of the model.
    if augmentation != 'light':
        x = layers.RandomFlip('horizontal', name="aug_flip")(x)
        x = layers.RandomRotation(0.15, name="aug_rotation")(x)
        x = layers.RandomZoom(0.1, name="aug_zoom")(x)

    # ---------- SqueezeNet 1.1 backbone ----------

    # Initial block
    x = conv_relu(x, 64, (3, 3), "conv1", kernel_initializer, batch_norm, strides=(2, 2))
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name="pool1")(x)

    # Fire modules 2-3
    x = fire_module(x, squeeze_filters=16, expand_filters=64, name_prefix="fire2",
                    kernel_initializer=kernel_initializer, batch_norm=batch_norm)
    x = fire_module(x, squeeze_filters=16, expand_filters=64, name_prefix="fire3",
                    kernel_initializer=kernel_initializer, batch_norm=batch_norm)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name="pool3")(x)

    # Fire modules 4-5
    x = fire_module(x, squeeze_filters=32, expand_filters=128, name_prefix="fire4",
                    kernel_initializer=kernel_initializer, batch_norm=batch_norm)
    x = fire_module(x, squeeze_filters=32, expand_filters=128, name_prefix="fire5",
                    kernel_initializer=kernel_initializer, batch_norm=batch_norm)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name="pool5")(x)

    # Fire modules 6-9
    x = fire_module(x, squeeze_filters=48, expand_filters=192, name_prefix="fire6",
                    kernel_initializer=kernel_initializer, batch_norm=batch_norm)
    x = fire_module(x, squeeze_filters=48, expand_filters=192, name_prefix="fire7",
                    kernel_initializer=kernel_initializer, batch_norm=batch_norm)
    x = fire_module(x, squeeze_filters=64, expand_filters=256, name_prefix="fire8",
                    kernel_initializer=kernel_initializer, batch_norm=batch_norm)
    x = fire_module(x, squeeze_filters=64, expand_filters=256, name_prefix="fire9",
                    kernel_initializer=kernel_initializer, batch_norm=batch_norm)

    # ---------- Classification head ----------
    x = layers.Dropout(dropout_rate, name="dropout")(x)

    # Final conv -> Global Average Pooling (replaces Flatten + Dense, ~0 extra params)
    if num_classes == 1:
        x = layers.Conv2D(1, (1, 1), activation='relu', padding='same',
                          kernel_initializer=kernel_initializer, name="conv10")(x)
        x = layers.GlobalAveragePooling2D(name="gap")(x)
        outputs = layers.Activation('sigmoid', name="output_sigmoid")(x)
        loss_fn = 'binary_crossentropy'
        print("Binary configuration (Sigmoid) for a single class.")
    else:
        x = layers.Conv2D(num_classes, (1, 1), activation='relu', padding='same',
                          kernel_initializer=kernel_initializer, name="conv10")(x)
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


def create_squeezenet_model(num_classes, img_shape, learning_rate, dropout_rate=0.5, augmentation="standard",
                            kernel_initializer="glorot_uniform", batch_norm=False):
    """
    Wrapper that builds, compiles and returns the SqueezeNet model.

    Args:
        num_classes: number of classes detected in the dataset
        img_shape: tuple (H, W, C)
        learning_rate: learning rate for Adam
        dropout_rate: dropout rate
        kernel_initializer, batch_norm: see build_squeezenet
    Returns:
        compiled model
    """
    print(f"Building SqueezeNet 1.1 for {num_classes} class(es)...")
    print(f"  Input shape : {img_shape}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Dropout rate: {dropout_rate}")
    print(f"  Initialiser : {kernel_initializer}")
    print(f"  Batch norm  : {'yes' if batch_norm else 'no'}")

    model, loss_fn = build_squeezenet(img_shape, num_classes, dropout_rate, augmentation,
                                      kernel_initializer=kernel_initializer, batch_norm=batch_norm)

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


def main(img_width, img_height, batch_size, epochs, learning_rate, dropout_rate, splits_dir,
         tag=None, overwrite=False, seed=42, lr_schedule="constant", augmentation="standard",
         kernel_initializer="glorot_uniform", batch_norm=False):
    model_name = "SqueezeNet"
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

    model = create_squeezenet_model(num_classes, img_shape, learning_rate, dropout_rate,
                                    augmentation=augmentation,
                                    kernel_initializer=kernel_initializer, batch_norm=batch_norm)
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

    print("\nStarting training of SqueezeNet...")
    try:
        history = model.fit(
            train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=callbacks,
            verbose=train_common.fit_verbosity()
        )

        plot_history(history, plot_path)
        print(f"Model saved to: {checkpoint_path}")

        # The label order travels with the checkpoint, so runs on different
        # datasets do not share (and overwrite) a single class-name file.
        classes_path = artifacts.write_class_names(checkpoint_path, class_names)
        print(f"Class names saved to {classes_path}")

        record_path = artifacts.write_training_record(
            checkpoint_path, script=os.path.basename(__file__), architecture=model_name,
            tag=tag, seed=seed,
            settings={"width": img_width, "height": img_height, "batch_size": batch_size,
                      "epochs": epochs, "learning_rate": learning_rate,
                      "dropout_rate": dropout_rate,
                      "kernel_initializer": kernel_initializer,
                      "batch_norm": bool(batch_norm),
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
        args.width, args.height,
        args.batch_size, args.epochs,
        args.learning_rate,
        args.dropout_rate, args.splits_dir,
        tag=args.tag, overwrite=args.overwrite, seed=args.seed,
        lr_schedule=args.lr_schedule, augmentation=args.augmentation,
        kernel_initializer=args.kernel_initializer, batch_norm=args.batch_norm
    )
