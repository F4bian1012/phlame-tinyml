#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
artifacts.py - Naming and protection of the files a run produces.

Shared by the training, quantization and evaluation scripts so that a run on a
new dataset neither collides with nor silently replaces the artifacts of
another run (in particular the published results tracked in results/).

Three rules live here:

1. Run name. Every artifact of a model derives from one stem,
       {arch}[_{tag}]+{batch}+{epochs}+{lr}+{W}+{H}
   The optional tag (--tag) tells runs on different datasets apart, e.g.
   MobileNetV2_VWW+32+20+0.0001+160+120.

2. No silent overwrite. A script stops before replacing an existing output
   unless --overwrite is given.

3. Class names travel with the model. Training writes the label order next to
   the checkpoint (<model>.classes.txt) and quantization copies it next to the
   .tflite, so the evaluators do not depend on a single global file.

Training also leaves a record of what produced the checkpoint
(<model>.training.json: seed, settings, per-epoch history, software versions).

Standard library only: importing it must stay cheap for --help.
"""

import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone

LEGACY_CLASS_NAMES = os.path.join("models", "class_names.txt")
CLASSES_SUFFIX = ".classes.txt"
TRAINING_SUFFIX = ".training.json"
_TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


# --------------------------------------------------------------------------- #
# Run name
# --------------------------------------------------------------------------- #
def valid_tag(value):
    """argparse type for --tag: letters, digits, '_' and '-' only.

    '+' separates the fields of the run name and the quantizer reads the input
    geometry from them, so it cannot appear inside a tag.
    """
    if not _TAG_RE.match(value):
        raise argparse.ArgumentTypeError(
            "invalid tag %r: use letters, digits, '_' or '-' (no spaces, no '+')" % value)
    return value


def model_stem(arch, batch_size, epochs, learning_rate, width, height, tag=None):
    """Run name shared by the checkpoint, the .tflite and every result file."""
    name = "%s_%s" % (arch, tag) if tag else arch
    return "%s+%s+%s+%s+%s+%s" % (name, batch_size, epochs, learning_rate, width, height)


def training_curve_name(arch, batch_size, epochs, learning_rate, width, height, tag=None):
    """File name of the training-curve plot of a run."""
    name = "%s_%s" % (arch, tag) if tag else arch
    return "%s_training_history+%s+%s+%s+%s+%s.png" % (
        name, batch_size, epochs, learning_rate, width, height)


# --------------------------------------------------------------------------- #
# Overwrite protection
# --------------------------------------------------------------------------- #
def refuse_to_overwrite(paths, overwrite, hint=None):
    """Stops the script (exit code 1) if any output exists and --overwrite was not given."""
    existing = [p for p in paths if os.path.exists(p)]
    if not existing or overwrite:
        return
    print("ERROR: this run would replace existing files:")
    for p in existing:
        print("  %s" % p)
    print("Nothing was written. Either:")
    if hint:
        print("  - %s" % hint)
    print("  - pass --overwrite to replace them.")
    sys.exit(1)


# --------------------------------------------------------------------------- #
# Class names
# --------------------------------------------------------------------------- #
def classes_path_for(model_path):
    """Path of the class-name file that travels with a model."""
    return os.path.splitext(model_path)[0] + CLASSES_SUFFIX


def write_class_names(model_path, class_names):
    """Writes the label order (one class per line) next to the model."""
    path = classes_path_for(model_path)
    with open(path, "w", encoding="utf-8") as fh:
        for name in class_names:
            fh.write("%s\n" % name)
    return path


def copy_class_names(src_model_path, dst_model_path):
    """Copies the class-name file of a model next to a derived one (e.g. its .tflite)."""
    src = classes_path_for(src_model_path)
    if not os.path.isfile(src):
        return None
    dst = classes_path_for(dst_model_path)
    shutil.copyfile(src, dst)
    return dst


def resolve_class_names_path(model_path, explicit_path=None):
    """Where an evaluator reads the label order from.

    Order: the path given on the command line, the file next to the model, and
    last the repository-wide models/class_names.txt (models trained before the
    per-model file existed, such as the published ones).
    """
    if explicit_path:
        return explicit_path
    beside = classes_path_for(model_path)
    if os.path.isfile(beside):
        return beside
    return LEGACY_CLASS_NAMES


# --------------------------------------------------------------------------- #
# Training record
# --------------------------------------------------------------------------- #
def training_record_path_for(model_path):
    """Path of the training record that travels with a checkpoint."""
    return os.path.splitext(model_path)[0] + TRAINING_SUFFIX


def write_training_record(model_path, script, architecture, tag, seed, settings,
                          splits_dir, class_names, history, software, data=None):
    """Writes next to the checkpoint what produced it.

    The seed fixes the initial weights, the shuffling and the augmentation, but
    two runs only match bit for bit on a CPU with the same software (on a GPU
    they do not), so the record also keeps the software versions and the
    per-epoch history: a later run can be compared with this one instead of
    being assumed equal to it.
    """
    record = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": script,
        "architecture": architecture,
        "tag": tag,
        "seed": seed,
        "settings": settings,
        "splits_dir": splits_dir.replace(os.sep, "/"),
        "class_names": list(class_names),
        "epochs_run": len(next(iter(history.values()), [])),
        "history": {name: [float(v) for v in values] for name, values in history.items()},
        "software": software,
    }
    if data is not None:
        # How the images were read and how many there were (see train_common.py).
        record["data"] = data
    path = training_record_path_for(model_path)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, ensure_ascii=False)
    return path
