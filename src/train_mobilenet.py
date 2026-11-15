import os
import sys
import datetime

import artifacts
import cli
import train_common


def parse_args():
    parser = cli.ArgumentParser(description="Train model for TinyML")
    parser.add_argument("--base_model", type=str, default="MobileNetV2", choices=["MobileNet", "MobileNetV2", "MobileNetV3Large", "MobileNetV3Small"], help="Base model architecture")
    parser.add_argument("--width", type=int, default=160, help="Image width")
    parser.add_argument("--height", type=int, default=120, help="Image height")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--splits_dir", type=str, default="data/splits", help="Directory holding the train/val partitions produced by split_dataset.py")
    parser.add_argument("--tag", type=artifacts.valid_tag, default=None, help="Optional run name added after the architecture (e.g. the dataset: MobileNetV2_VWW+...), so runs on different datasets do not share file names")
    parser.add_argument("--overwrite", action="store_true", help="Replace the checkpoint and the training curve if they already exist")
    parser.add_argument("--seed", type=int, default=42, help="Seed of the initial weights, the shuffling and the augmentation (default: 42). On a CPU it repeats a run bit for bit on the same machine and software; on a GPU two runs with the same seed still differ, because some GPU kernels are not deterministic")

    train_common.add_recipe_arguments(parser)
    parser.add_argument("--trainable_backbone", action="store_true", help="Train every layer, with the ImageNet weights as the starting point (default: the backbone is frozen and only the new classifier is trained)")
    parser.add_argument("--weights", choices=["imagenet", "none"], default="imagenet", help="Starting weights of the backbone: imagenet (default) or none, which trains every layer from random weights")
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
STANDARD_AUGMENTATION = {"horizontal_flip": True, "rotation_fraction_of_turn": 0.2, "zoom_fraction": 0.1}


def create_mobilenet_model(num_classes, img_shape, learning_rate, base_model_name,
                           weights='imagenet', trainable_backbone=False, augmentation='standard',
                           update_bn_statistics=None):
    """
    Creates a model based on the selected base_model.

    Default: transfer learning, i.e. the ImageNet backbone is frozen and only
    the new classifier is trained. With trainable_backbone every layer is
    trained from the ImageNet weights, and with weights=None from random
    weights (every layer is trained then, whatever trainable_backbone says).

    update_bn_statistics says whether the batch-normalisation layers of the
    backbone recompute their moving averages on the new data. None picks the
    default of each case: no for a frozen backbone, yes otherwise.
    """
    print(f"Building {base_model_name} model for {num_classes} classes...")

    # The 'light' augmentation is applied to the batches by train_common.py, so
    # the model holds augmentation layers only with the standard one.
    data_augmentation = None if augmentation == 'light' else keras.Sequential([
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
        raise ValueError(f"Unsupported base model: {base_model_name}")

    base_input_shape = list(img_shape)
    if base_input_shape[-1] == 1:
        base_input_shape[-1] = 3
    base_input_shape = tuple(base_input_shape)

    base_model = base_model_class(
        input_shape=base_input_shape,
        include_top=False,
        weights=weights,
        alpha=alpha
    )
    train_all = bool(trainable_backbone) or weights is None
    base_model.trainable = train_all
    if update_bn_statistics is None:
        update_bn_statistics = train_all
    print(f"  Backbone: weights={weights}, trainable={train_all}, "
          f"batch-norm statistics {'updated' if update_bn_statistics else 'frozen'}")

    inputs = tf.keras.Input(shape=img_shape)
    x = data_augmentation(inputs) if data_augmentation is not None else inputs

    if img_shape[-1] == 1:
        x = layers.Concatenate(axis=-1)([x, x, x])

    x = preprocess_fn(x)
    # training=False keeps the batch-normalisation layers on their ImageNet
    # statistics; leaving the argument out lets them follow the new data.
    x = base_model(x) if update_bn_statistics else base_model(x, training=False)
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

def main(img_width, img_height, batch_size, epochs, learning_rate, base_model_name, splits_dir,
         tag=None, overwrite=False, seed=42, lr_schedule="constant", augmentation="standard",
         weights="imagenet", trainable_backbone=False):
    img_shape = (img_height, img_width, 1)
    plot_path = "results/training_curves/" + artifacts.training_curve_name(
        base_model_name, batch_size, epochs, learning_rate, img_width, img_height, tag)
    checkpoint_path = "models/checkpoints/" + artifacts.model_stem(
        base_model_name, batch_size, epochs, learning_rate, img_width, img_height, tag) + ".keras"

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

    model = create_mobilenet_model(num_classes, img_shape, learning_rate, base_model_name,
                                   weights=weights, trainable_backbone=trainable_backbone,
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
    ] + train_common.schedule_callbacks(lr_schedule, learning_rate, epochs)

    print("Starting training...")
    try:
        history = model.fit(
            train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=callbacks,
            verbose=train_common.fit_verbosity()
        )

        plot_history(history, plot_path)
        print(f"Model saved to {checkpoint_path}")

        # The label order travels with the checkpoint, so runs on different
        # datasets do not share (and overwrite) a single class-name file.
        classes_path = artifacts.write_class_names(checkpoint_path, class_names)
        print(f"Class names saved to {classes_path}")

        record_path = artifacts.write_training_record(
            checkpoint_path, script=os.path.basename(__file__), architecture=base_model_name,
            tag=tag, seed=seed,
            settings={"width": img_width, "height": img_height, "batch_size": batch_size,
                      "epochs": epochs, "learning_rate": learning_rate,
                      "weights": weights,
                      "trainable_backbone": bool(trainable_backbone) or weights is None,
                      "backbone_batch_norm_statistics": (
                          "updated" if (bool(trainable_backbone) or weights is None) else "frozen (ImageNet)"),
                      **train_common.recipe_settings(lr_schedule, learning_rate, epochs, augmentation,
                                                     STANDARD_AUGMENTATION)},
            splits_dir=splits_dir, class_names=class_names,
            history=history.history, software=train_common.software_versions(), data=data_info)
        print(f"Training record (seed {seed}) saved to {record_path}")

    except Exception as e:
        print(f"Training failed: {e}")
        print("Tip: Ensure you have enough images per class (at least 10-20 to start).")
        sys.exit(1)

if __name__ == "__main__":
    args = ARGS
    main(args.width, args.height, args.batch_size, args.epochs, args.learning_rate, args.base_model, args.splits_dir,
         tag=args.tag, overwrite=args.overwrite, seed=args.seed,
         lr_schedule=args.lr_schedule, augmentation=args.augmentation,
         weights=None if args.weights == "none" else args.weights,
         trainable_backbone=args.trainable_backbone)
