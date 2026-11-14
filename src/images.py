#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
images.py - The one place where an image file becomes the array a model receives.

Training, the calibration of the quantizer, MIL, SIL and the PIL bench all read
their images through this module, so the same file gives the same bytes at
every level of the ladder. Before it existed, MIL and SIL decoded each JPEG
with TensorFlow and the PIL bench with Pillow, and the two decoders do not give
the same pixels: part of what looked like a difference between the PC and the
chip was a difference between decoders.

What a model receives is the raw pixel, uint8 in [0, 255], in HWC order:

    Pillow opens the file
    -> convert('L') for grayscale, convert('RGB') for colour
    -> LANCZOS resize, only when the file is not already width x height
    -> numpy uint8 array of shape (H, W) or (H, W, 3)

The HIL bench does not use it: its images are the frames the camera captures.

Standard library, Pillow and numpy only (no TensorFlow), so the board bench and
the tests can import it without the heavy packages.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

# Extensions read from a dataset folder, compared in lower case.
IMAGE_EXTENSIONS = ('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')
DEFAULT_WORKERS = min(32, (os.cpu_count() or 4))


def load_image(path, width, height, grayscale=True):
    """The uint8 array the model receives for one image file.

    Shape (height, width) for grayscale and (height, width, 3) for colour. The
    file is resized with LANCZOS only when its size is not width x height.
    """
    with Image.open(path) as img:
        img = img.convert('L' if grayscale else 'RGB')
        if img.size != (width, height):
            img = img.resize((width, height), Image.LANCZOS)
        return np.array(img, dtype=np.uint8)


def list_class_dirs(folder):
    """Class subfolders of a dataset folder, sorted; hidden folders are skipped."""
    return sorted(d for d in os.listdir(folder)
                  if os.path.isdir(os.path.join(folder, d)) and not d.startswith('.'))


def list_images(folder, class_names=None):
    """Image files of a class-per-folder directory, in a fixed order.

    Returns (class_names, items) where items is a list of
    (relative path with '/', label). The classes follow class_names (the label
    is the position in that list) or, when it is not given, the sorted
    subfolders. Inside a class the files are sorted by their relative path, so
    the order does not depend on the file system or on the operating system.
    """
    if class_names is None:
        class_names = list_class_dirs(folder)
    class_names = list(class_names)
    items = []
    for label, name in enumerate(class_names):
        class_dir = os.path.join(folder, name)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError("missing subfolder for class '%s' in %s" % (name, folder))
        found = []
        for root, dirs, files in os.walk(class_dir):
            dirs[:] = sorted(d for d in dirs if not d.startswith('.'))
            for f in files:
                if f.lower().endswith(IMAGE_EXTENSIONS) and not f.startswith('.'):
                    rel = os.path.relpath(os.path.join(root, f), folder).replace(os.sep, '/')
                    found.append(rel)
        items.extend((rel, label) for rel in sorted(found))
    return class_names, items


def load_paths(paths, width, height, grayscale=True, workers=DEFAULT_WORKERS):
    """Loads a list of image files into one uint8 array, keeping their order.

    Shape (N, height, width) for grayscale and (N, height, width, 3) for
    colour. Files are read by a thread pool (decoding releases the GIL); the
    result does not depend on the number of threads.
    """
    shape = (len(paths), height, width) if grayscale else (len(paths), height, width, 3)
    out = np.empty(shape, dtype=np.uint8)

    def read(i):
        out[i] = load_image(paths[i], width, height, grayscale)

    if workers and workers > 1 and len(paths) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # list() so that an exception raised while reading a file surfaces here
            list(pool.map(read, range(len(paths))))
    else:
        for i in range(len(paths)):
            read(i)
    return out


def load_folder(folder, width, height, class_names=None, grayscale=True, workers=DEFAULT_WORKERS):
    """Loads a class-per-folder directory.

    Returns (images, labels, rel_paths, class_names):
      images      uint8, (N, height, width) or (N, height, width, 3)
      labels      int64, (N,), position of the class in class_names
      rel_paths   paths relative to folder, with '/' on every operating system
      class_names the classes, in label order
    in the fixed order of list_images().
    """
    class_names, items = list_images(folder, class_names)
    rel_paths = [rel for rel, _ in items]
    labels = np.array([label for _, label in items], dtype=np.int64)
    images = load_paths([os.path.join(folder, *rel.split('/')) for rel in rel_paths],
                        width, height, grayscale, workers)
    return images, labels, rel_paths, class_names


def as_model_input(images):
    """uint8 images as the float32 batch a Keras or TFLite model takes:
    values still in [0, 255] and a trailing channel axis for grayscale."""
    batch = images.astype(np.float32)
    if batch.ndim == 3:
        batch = batch[..., np.newaxis]
    return batch
