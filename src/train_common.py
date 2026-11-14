#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_common.py - What the three training scripts share: the data and the recipe.

Data. The train and val partitions are read once with images.py, the loader of
MIL, SIL and the PIL bench, and kept in memory as uint8 (a 160x120 grayscale
image is 19,200 bytes, so 100,000 of them take 1.9 GB). Training therefore
sees the pixels the evaluation levels see, and an epoch is bound by the model
and not by JPEG decoding. The training images are shuffled again at every
epoch, image by image.

Recipe. Without options the scripts train as they always did. Three options
add what the best published results on person detection over COCO (Visual
Wake Words) have in common:

  --lr_schedule stepped     full-length training with the learning rate halved
                            at 40 % and at 60 % of the epochs (MLPerf Tiny
                            trains its reference model with 20 epochs at 1e-3,
                            10 at 5e-4 and 20 at 2.5e-4), instead of a constant
                            rate with early stopping
  --augmentation light      horizontal flip, rotation of 10 degrees, shifts of
                            5 % and zoom of 10 % (MLPerf Tiny), instead of the
                            stronger rotation of each script. It is applied to
                            the batches before they reach the model, as one
                            affine transform per image, so the saved model
                            holds no augmentation layers
  --trainable_backbone      (MobileNets) every layer is trained, with the
                            ImageNet weights as the starting point, instead of
                            a frozen backbone with a new classifier on top

Every choice is written to the training record (<model>.training.json).
"""

import os
import sys

import numpy as np
import PIL

import images

LR_SCHEDULES = ("constant", "stepped")
AUGMENTATIONS = ("standard", "light")

# MLPerf Tiny, benchmark/training/visual_wake_words/train_vww.py
LIGHT_AUGMENTATION = {"horizontal_flip": True, "rotation_degrees": 10, "shift_fraction": 0.05,
                      "zoom_fraction": 0.1}
# The learning rate is multiplied by these factors from these fifths of the epochs on
# (0, 2/5 and 3/5 of the run, rounded up to a whole epoch).
STEPPED_SCHEDULE = ((0, 1.0), (2, 0.5), (3, 0.25))


def _tf():
    """TensorFlow, imported on first use: the helpers that only describe the
    recipe (and their tests) do not need it."""
    import tensorflow
    return tensorflow


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def add_recipe_arguments(parser):
    """Recipe options shared by the three training scripts."""
    parser.add_argument(
        "--lr_schedule", choices=LR_SCHEDULES, default="constant",
        help="constant (default): --learning_rate for the whole run, with early stopping. "
             "stepped: every epoch is run, with the rate halved at 40%% and at 60%% of them "
             "(for 50 epochs: 20 at the rate, 10 at half, 20 at a quarter), no early stopping")
    parser.add_argument(
        "--augmentation", choices=AUGMENTATIONS, default="standard",
        help="standard (default): the augmentation the script always used. light: horizontal flip, "
             "rotation of 10 degrees, shifts of 5%% and zoom of 10%%")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def light_augment(batch, seed):
    """The 'light' augmentation of a float32 batch (N, H, W, C), with a seed of two integers.

    Flip, rotation, zoom and shift are drawn per image and applied together as
    one affine transform (bilinear interpolation, edge pixels repeated outside
    the image), which is how the Keras ImageDataGenerator of the MLPerf Tiny
    script works: one interpolation instead of one per change.
    """
    tf = _tf()
    a = LIGHT_AUGMENTATION
    n = tf.shape(batch)[0]
    h = tf.cast(tf.shape(batch)[1], tf.float32)
    w = tf.cast(tf.shape(batch)[2], tf.float32)
    u = tf.random.stateless_uniform([6, n], seed=seed, minval=-1.0, maxval=1.0)
    theta = u[0] * (a["rotation_degrees"] * np.pi / 180.0)
    zoom_x = 1.0 + u[1] * a["zoom_fraction"]
    zoom_y = 1.0 + u[2] * a["zoom_fraction"]
    shift_x = u[3] * a["shift_fraction"] * w
    shift_y = u[4] * a["shift_fraction"] * h
    flip = tf.where(u[5] < 0.0, -1.0, 1.0) if a["horizontal_flip"] else tf.ones_like(u[5])
    cx, cy = (w - 1.0) / 2.0, (h - 1.0) / 2.0
    # Each output pixel (x, y) takes its value from (a0 x + a1 y + a2, b0 x + b1 y + b2).
    a0 = zoom_x * tf.cos(theta) * flip
    a1 = -zoom_x * tf.sin(theta)
    b0 = zoom_y * tf.sin(theta) * flip
    b1 = zoom_y * tf.cos(theta)
    a2 = cx + shift_x - a0 * cx - a1 * cy
    b2 = cy + shift_y - b0 * cx - b1 * cy
    zeros = tf.zeros_like(a0)
    transforms = tf.stack([a0, a1, a2, b0, b1, b2, zeros, zeros], axis=1)
    return tf.raw_ops.ImageProjectiveTransformV3(
        images=batch, transforms=transforms, output_shape=tf.shape(batch)[1:3],
        interpolation="BILINEAR", fill_mode="NEAREST", fill_value=0.0)


def _dataset(x, y, batch_size, shuffle, seed, augmentation="standard"):
    """tf.data pipeline over arrays held in memory.

    Only the indices go through the shuffle buffer; each batch is then gathered
    from the uint8 array, so memory stays at the size of the array. The 'light'
    augmentation is applied here; the 'standard' one is made of layers inside
    the model.
    """
    tf = _tf()
    with tf.device("/CPU:0"):
        x_t = tf.constant(x)
        y_t = tf.constant(y)
    grayscale = (x.ndim == 3)

    def take(idx):
        batch = tf.cast(tf.gather(x_t, idx), tf.float32)
        if grayscale:
            batch = batch[..., tf.newaxis]
        return batch, tf.gather(y_t, idx)

    ds = tf.data.Dataset.range(len(x))
    if shuffle:
        ds = ds.shuffle(len(x), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).map(take, num_parallel_calls=tf.data.AUTOTUNE)
    if augmentation == "light":
        # One pair of integers per batch, different at every epoch, drawn from the seed:
        # the augmentation does not depend on how many threads run the map.
        seeds = tf.data.Dataset.random(seed=seed, rerandomize_each_iteration=True).batch(2, drop_remainder=True)
        ds = tf.data.Dataset.zip((ds, seeds)).map(
            lambda batch, s: (light_augment(batch[0], s), batch[1]), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def load_splits(splits_dir, img_width, img_height, batch_size, seed=None, augmentation="standard"):
    """Loads the train/val partitions written to disk by split_dataset.py.

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

    Returns (train_ds, val_ds, class_names, data_info); data_info goes to the
    training record.
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
    class_names = images.list_class_dirs(train_dir)
    if not class_names:
        print(f"Data Loading Error: no class subfolders in {train_dir}")
        print(f"Ensure each split has a subfolder per class, e.g., '{train_dir}/class_a/image.jpg'.")
        sys.exit(1)
    print(f"Found classes: {class_names}")

    # Both splits must expose the same label order, or the argmax the firmware
    # returns would not mean the same thing in training and evaluation.
    val_classes = images.list_class_dirs(val_dir)
    if val_classes != class_names:
        print("ERROR: class order differs between the train and val splits.")
        print(f"  train: {class_names}")
        print(f"  val  : {val_classes}")
        sys.exit(1)

    if len(class_names) == 1:
        print("\nWARNING: Only 1 class found ('{}').".format(class_names[0]))
        print("   Training a classifier with ONLY positive samples will result in a model")
        print("   that predicts this class for EVERYTHING (Accuracy will be trivial).")
        print("   You should add a 'background' or 'negative' class folder with random images")
        print("   to teach the model what is NOT a '{}'.\n".format(class_names[0]))

    x_train, y_train, _, _ = images.load_folder(train_dir, img_width, img_height, class_names)
    x_val, y_val, _, _ = images.load_folder(val_dir, img_width, img_height, class_names)
    if len(x_train) == 0 or len(x_val) == 0:
        print(f"Data Loading Error: no images found under {train_dir} or {val_dir}")
        sys.exit(1)
    print(f"Found {len(x_train)} training and {len(x_val)} validation images "
          f"({x_train.nbytes / 2**20:.0f} MiB in memory).")

    data_info = {
        "loader": "images.py (Pillow %s), raw pixels in memory" % PIL.__version__,
        "shuffle": "every epoch, image by image",
        "n_train": int(len(x_train)),
        "n_val": int(len(x_val)),
        "train_per_class": {name: int((y_train == i).sum()) for i, name in enumerate(class_names)},
        "val_per_class": {name: int((y_val == i).sum()) for i, name in enumerate(class_names)},
    }
    train_ds = _dataset(x_train, y_train, batch_size, shuffle=True, seed=seed, augmentation=augmentation)
    val_ds = _dataset(x_val, y_val, batch_size, shuffle=False, seed=None)
    return train_ds, val_ds, class_names, data_info


# --------------------------------------------------------------------------- #
# Recipe
# --------------------------------------------------------------------------- #
def learning_rates(lr_schedule, learning_rate, epochs):
    """The learning rate of every epoch for a schedule (constant: the starting one)."""
    if lr_schedule != "stepped":
        return [learning_rate] * epochs
    rates = []
    for epoch in range(epochs):
        factor = 1.0
        for fifths, f in STEPPED_SCHEDULE:
            if epoch >= -(-fifths * epochs // 5):
                factor = f
        rates.append(learning_rate * factor)
    return rates


def schedule_callbacks(lr_schedule, learning_rate, epochs, plateau=False):
    """Callbacks that decide the learning rate and the length of the run.

    constant: early stopping on the validation loss (and, with plateau=True,
    the rate is halved when that loss stops improving), as the scripts always
    did. stepped: the run has the length asked for and the rate follows
    learning_rates(); nothing stops it early.
    """
    keras = _tf().keras
    if lr_schedule == "stepped":
        rates = learning_rates(lr_schedule, learning_rate, epochs)
        return [keras.callbacks.LearningRateScheduler(lambda epoch, lr: rates[epoch], verbose=0)]
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                               restore_best_weights=True, verbose=1)]
    if plateau:
        callbacks.append(keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3,
                                                           min_lr=1e-7, verbose=1))
    return callbacks


def recipe_settings(lr_schedule, learning_rate, epochs, augmentation, standard_augmentation, plateau=False):
    """The recipe fields of the training record."""
    stepped = (lr_schedule == "stepped")
    return {
        "lr_schedule": lr_schedule,
        "learning_rates": sorted(set(learning_rates(lr_schedule, learning_rate, epochs)), reverse=True),
        "early_stopping": None if stepped else "val_loss, patience 5, best weights restored",
        "reduce_lr_on_plateau": ("val_loss, factor 0.5, patience 3" if plateau and not stepped else None),
        "checkpoint": "epoch with the best val_accuracy",
        "augmentation": augmentation,
        "augmentation_parameters": LIGHT_AUGMENTATION if augmentation == "light" else standard_augmentation,
    }


def fit_verbosity():
    """One line per epoch when the output goes to a file, the progress bar on a terminal."""
    return 1 if sys.stdout.isatty() else 2


def software_versions():
    """Software behind a run, written to the training record."""
    import platform
    tf = _tf()
    return {
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "keras": tf.keras.__version__,
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "platform": platform.platform(),
        "gpus": [device.name for device in tf.config.list_physical_devices("GPU")],
    }
