#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation.py - What the two PC levels of the ladder do after they have predictions.

mil_benchmark.py (float .keras model) and sil_benchmark.py (INT8 .tflite model)
differ in how a model is loaded and run. Everything else is the same and lives
here, so that the two levels cannot drift apart:

  * where the label order comes from, and the check against the folders of the
    test split;
  * the predicted label of every image from the model output;
  * accuracy, weighted precision / recall / F1, macro-F1 and the per-class report;
  * the confusion matrix and its plot;
  * the three result files, named after the model:
        confusion_matrix_<model>.png
        predictions_<model>.csv      one row per image, the schema of every level
        metrics_<model>.json         documented in docs/results_schema.md

A level passes what is its own (level name, the keys of its `model` and
`dataset` blocks, the `software` block, the texts of the report) and gets the
same files as before this module existed.

Imports numpy, matplotlib, scikit-learn and seaborn, but not TensorFlow: the
scripts import it after they have read their command line.
"""

import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import matplotlib.pyplot as plt

import artifacts

# Additional dependencies required for the metrics and the confusion matrix:
# pip install scikit-learn seaborn
try:
    import seaborn as sns
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
except ImportError:
    print("Please install scikit-learn and seaborn to compute the metrics.")
    print("Run: pip install scikit-learn seaborn")
    sys.exit(1)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def class_name(idx, class_names):
    """Class name from the predicted or ground-truth index."""
    if idx is None or idx < 0:
        return ""
    if idx < len(class_names):
        return class_names[idx]
    return "Invalid_%d" % idx


def sha256_file(path, chunk=1024 * 1024):
    """SHA-256 of the model file, read in chunks.

    In chunks because a model file can weigh tens of MB and there is no need
    to hold it entirely in memory just to hash it.

    Returns None if the path is not a file (e.g. a SavedModel, which is a
    folder): a JSON without a checksum is preferable to losing the whole
    evaluation on the last line, after having run the full inference.
    """
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def json_safe(o):
    """Converts numpy types to native Python types.

    classification_report(output_dict=True) returns the "support" field as
    np.int64 and the metrics as np.float64, and json.dump cannot serialize
    them: without this the dump fails with TypeError after having run the
    full inference.
    """
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError("Cannot serialize to JSON: %r (%s)" % (o, type(o).__name__))


def model_name(model_path):
    """Name every result file of a model carries: its file name without the extension."""
    return os.path.splitext(os.path.basename(model_path))[0]


# --------------------------------------------------------------------------- #
# Before the model runs
# --------------------------------------------------------------------------- #
def refuse_to_replace_results(output_dir, model_path, overwrite):
    """Stops the script if the three result files of this model exist and --overwrite was not given.

    Checked before loading anything: a run must not replace the results of
    another one (the published ones are tracked in results/).
    """
    stem = model_name(model_path)
    artifacts.refuse_to_overwrite(
        [os.path.join(output_dir, name) for name in
         (f"confusion_matrix_{stem}.png", f"predictions_{stem}.csv", f"metrics_{stem}.json")],
        overwrite,
        hint="write somewhere else with --output_dir, or train with --tag NAME")


def stop_without_test_split(data_dir, width, height, splits_dir):
    """Says how to create the test split that is missing, and stops the script (exit code 1)."""
    print(f"ERROR: test split not found at {data_dir}")
    print("Run the dataset split first:")
    print(f"  python src/split_dataset.py"
          f" --input_dir data/processed/{width}x{height}"
          f" --output_dir {splits_dir}")
    sys.exit(1)


def read_class_names(model_path, explicit_path=None):
    """The label order that travels with the model.

    Returns (class_names, path it was looked for in); the list is empty when
    there is no such file, and the names are then taken from the folders of
    the test split.
    """
    class_names = []
    class_names_path = artifacts.resolve_class_names_path(model_path, explicit_path)
    if os.path.exists(class_names_path):
        with open(class_names_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines() if line.strip()]
        print(f"Class names loaded from {class_names_path}: {class_names}")
    return class_names, class_names_path


def class_names_of_this_split(class_names, class_names_path, data_dir):
    """The class names, or an empty list when they are not the folders of the test split.

    A class-name file of another dataset would label every image wrongly, so
    the names are then inferred from the folders, with a warning.
    """
    if class_names:
        subdirs = sorted(
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith(".")
        )
        if subdirs and set(subdirs) != set(class_names):
            print(
                f"[WARN] The subfolders of {data_dir} ({subdirs}) do not match "
                f"{class_names_path} ({class_names}); the class names will be inferred from the directory."
            )
            class_names = []
    return class_names


def describe_test_split(rel_paths, class_names, inferred):
    """Prints what the loader found; `inferred` says the class names come from the folders."""
    if inferred:
        print(f"Class names inferred from the directory: {class_names}")
    print(f"Found {len(rel_paths)} files belonging to {len(class_names)} classes.")


# --------------------------------------------------------------------------- #
# After the model has run
# --------------------------------------------------------------------------- #
def predicted_labels(predictions, num_classes):
    """The predicted label of every image, from the scores the model returns."""
    if num_classes == 1:
        # Binary classification with a single defined class (sigmoid or other output)
        return (predictions > 0.5).astype(int).reshape(-1)
    if num_classes == 2 and predictions.shape[1] == 1:
        # Binary classification (sigmoid output)
        return (predictions > 0.5).astype(int).reshape(-1)
    # Multiclass classification (softmax output)
    return np.argmax(predictions, axis=1)


def write_level_results(level, model_path, output_dir, y_true, y_pred, rel_paths, class_names,
                        dataset, image_size, software, report_title, plot_title, model_extra=None):
    """Prints the metrics report of a level and writes its three result files.

    level         "MIL" or "SIL", written to the metrics file
    dataset       the keys the level puts first in the `dataset` block (the paths
                  it read); the number of images, the class names and the image
                  size are added here
    image_size    (width, height) of the model input
    software      the `software` block (artifacts.software_versions)
    report_title  heading of the report on the console
    plot_title    title of the confusion-matrix plot
    model_extra   keys the level adds to the `model` block after the checksum
                  (SIL: the TFLite kernels)

    Returns the path of the metrics file.
    """
    print("\n" + "=" * 50)
    print(report_title)
    print("=" * 50)

    # Compute the global metrics (weighted works well if there are imbalanced classes)
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    # 'macro' averages the classes without weighting by support. With an
    # imbalanced dataset, the 'weighted' F1 is dominated by the majority class
    # and can hide that the minority one is doing badly; macro weights them
    # equally. The two figures together make that difference visible instead
    # of averaging it away.
    _, _, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)

    print(f"Accuracy:             {accuracy:.4f}")
    print(f"Precision:            {precision:.4f}")
    print(f"Recall:               {recall:.4f}")
    print(f"F1-Score (weighted):  {f1:.4f}")
    print(f"F1-Score (macro):     {f1_macro:.4f}")

    # Detailed per-class report
    print("\nDetailed Classification Report:")
    # Requested twice: the text for the console and the dict for the JSON. They
    # are the same call with output_dict, so the figures cannot diverge.
    print(classification_report(y_true, y_pred, target_names=class_names, labels=range(len(class_names)), zero_division=0))
    report_dict = classification_report(y_true, y_pred, target_names=class_names, labels=range(len(class_names)), zero_division=0, output_dict=True)

    print("\nGenerating Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(plot_title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    # The matrices of a level live alongside the rest of the ladder results
    # (results/mil, results/sil, results/pil, results/hil), not next to the
    # model file. The file names derive from the model name, as in
    # pil_benchmark.py, so runs with different models do not overwrite each other.
    out_dir = output_dir
    os.makedirs(out_dir, exist_ok=True)

    model_name_without_ext = model_name(model_path)
    cm_plot_path = os.path.join(out_dir, f"confusion_matrix_{model_name_without_ext}.png")

    plt.savefig(cm_plot_path)
    print(f"Confusion matrix plot saved to {cm_plot_path}")

    # CSV with the per-image prediction, with the same schema as pil_benchmark.py
    # (Image;RelPath;TrueLabel;TrueClass;PredLabel;PredClass;Correct) so that
    # MIL, SIL, PIL and the HIL subset (hil_subset==1) can be joined image by image.
    pred_csv_path = os.path.join(out_dir, f"predictions_{model_name_without_ext}.csv")
    with open(pred_csv_path, "w", newline="", encoding="utf-8") as pred_file:
        pred_writer = csv.writer(pred_file, delimiter=";")
        pred_writer.writerow(["Image", "RelPath", "TrueLabel", "TrueClass",
                              "PredLabel", "PredClass", "Correct"])
        # RelPath is written with '/' on every operating system
        for rel, t_lbl, p_lbl in zip(rel_paths, y_true, y_pred):
            t_int = int(t_lbl)
            p_int = int(p_lbl)
            pred_writer.writerow([
                rel.split("/")[-1],
                rel,
                t_int,
                class_name(t_int, class_names),
                p_int,
                class_name(p_int, class_names),
                1 if t_int == p_int else 0,
            ])
    print(f"Per-image predictions saved to {pred_csv_path}")

    # Besides the PNG, a JSON with the same figures is saved. The image is for
    # looking at, but not for comparing levels of the ladder or for another
    # script to read the numbers; the JSON is. Same naming criterion as
    # confusion_matrix_*.png: derived from the model, so that two runs do not
    # overwrite each other.
    metrics_path = os.path.join(out_dir, f"metrics_{model_name_without_ext}.json")

    # The checksum ties these metrics to the exact artifact that produced them.
    # If later on a run does not reproduce these figures, the hash says right
    # away whether the model changed or something else did (the data, the
    # preprocessing, the split). Without it, a regenerated model file with the
    # same name is indistinguishable from the original.
    model_sha256 = sha256_file(model_path)
    model_bytes = os.path.getsize(model_path) if os.path.isfile(model_path) else None
    print(f"Model SHA-256:        {model_sha256}")

    model_block = {
        "path": model_path,
        "name": model_name_without_ext,
        "sha256": model_sha256,
        "bytes": model_bytes,
    }
    model_block.update(model_extra or {})
    dataset_block = dict(dataset)
    dataset_block.update({
        "n_samples": int(len(y_true)),
        "class_names": list(class_names),
        "image_size": {"width": image_size[0], "height": image_size[1]},
    })

    results = {
        "level": level,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_block,
        "dataset": dataset_block,
        "global_metrics": {
            "accuracy": float(accuracy),
            "precision_weighted": float(precision),
            "recall_weighted": float(recall),
            "f1_weighted": float(f1),
            "f1_macro": float(f1_macro),
        },
        "per_class_metrics": report_dict,
        "confusion_matrix": {
            "labels": list(class_names),
            "matrix": cm.tolist(),
            "order": "rows = true label, columns = predicted label",
        },
        "software": software,
        "artifacts": {
            "confusion_matrix_png": cm_plot_path,
            "predictions_csv": pred_csv_path,
        },
    }

    with open(metrics_path, "w", encoding="utf-8") as fh:
        # Output schema documented in docs/results_schema.md - update it if you add, rename or remove keys.
        json.dump(results, fh, indent=2, ensure_ascii=False, default=json_safe)
    print(f"JSON metrics saved to {metrics_path}")
    return metrics_path
