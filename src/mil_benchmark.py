import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import artifacts
import cli

# Additional dependencies required for the metrics and the confusion matrix:
# pip install scikit-learn seaborn
try:
    import seaborn as sns
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
except ImportError:
    print("Please install scikit-learn and seaborn to compute the metrics.")
    print("Run: pip install scikit-learn seaborn")
    sys.exit(1)

BATCH_SIZE = 32


def _class_name(idx, class_names):
    """Class name from the predicted or ground-truth index."""
    if idx is None or idx < 0:
        return ""
    if idx < len(class_names):
        return class_names[idx]
    return "Invalid_%d" % idx


def _sha256_file(path, chunk=1024 * 1024):
    """SHA-256 of the model file, read in chunks.

    In chunks because a .keras file can weigh tens of MB and there is no need
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


def _json_safe(o):
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


def parse_args():
    parser = cli.ArgumentParser(description="MIL level: evaluate a .keras model on the test split and compute its metrics")
    parser.add_argument('--width', type=int, default=160, help="Image width")
    parser.add_argument('--height', type=int, default=120, help="Image height")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the .keras model to evaluate (required)")
    parser.add_argument('--splits_dir', type=str, default="data/splits",
                        help="Directory holding the splits produced by split_dataset.py "
                             "(evaluation uses ONLY <splits_dir>/test)")
    parser.add_argument('--data_dir', type=str, default=None, help="Explicit path to the test split; defaults to <splits_dir>/test")
    parser.add_argument('--class_names_path', type=str, default=None, help="Path to the txt file with the class names (default: the .classes.txt next to the model, else models/class_names.txt)")
    parser.add_argument('--output_dir', type=str, default=os.path.join("results", "mil"), help="Output directory for the matrix, JSON and CSV (default: results/mil)")
    parser.add_argument('--overwrite', action='store_true', help="Replace the matrix, JSON and CSV if they already exist")

    args = parser.parse_args()

    # MIL evaluates the test split held out by split_dataset.py, not the full
    # directory: otherwise the reported accuracy would include the images the
    # model was trained on.
    if args.data_dir is None:
        args.data_dir = os.path.join(args.splits_dir, "test")

    return args

def main():
    args = parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: model not found at {args.model_path}")
        sys.exit(1)

    if not os.path.exists(args.data_dir):
        print(f"ERROR: test split not found at {args.data_dir}")
        print("Run the dataset split first:")
        print(f"  python src/split_dataset.py"
              f" --input_dir data/processed/{args.width}x{args.height}"
              f" --output_dir {args.splits_dir}")
        sys.exit(1)

    # Checked before loading anything: a run must not replace the results of
    # another one (the published ones are tracked in results/).
    stem = os.path.splitext(os.path.basename(args.model_path))[0]
    artifacts.refuse_to_overwrite(
        [os.path.join(args.output_dir, name) for name in
         (f"confusion_matrix_{stem}.png", f"predictions_{stem}.csv", f"metrics_{stem}.json")],
        args.overwrite,
        hint="write somewhere else with --output_dir, or train with --tag NAME")

    class_names = []
    class_names_path = artifacts.resolve_class_names_path(args.model_path, args.class_names_path)
    if os.path.exists(class_names_path):
        with open(class_names_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines() if line.strip()]
        print(f"Class names loaded from {class_names_path}: {class_names}")

    if class_names:
        subdirs = sorted(
            d for d in os.listdir(args.data_dir)
            if os.path.isdir(os.path.join(args.data_dir, d)) and not d.startswith(".")
        )
        if subdirs and set(subdirs) != set(class_names):
            print(
                f"[WARN] The subfolders of {args.data_dir} ({subdirs}) do not match "
                f"{class_names_path} ({class_names}); the class names will be inferred from the directory."
            )
            class_names = []

    print(f"\nLoading the model from {args.model_path}...")
    model = keras.models.load_model(args.model_path, safe_mode=False)

    print(f"Loading the test dataset from {args.data_dir}...")
    # We use shuffle=False to keep the image order and align y_true with y_pred
    test_ds = tf.keras.utils.image_dataset_from_directory(
        args.data_dir,
        labels='inferred',
        label_mode='int',
        class_names=class_names if class_names else None,
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        image_size=(args.height, args.width),
        shuffle=False
    )

    if not class_names:
        class_names = test_ds.class_names
        print(f"Class names inferred from the directory: {class_names}")

    num_classes = len(class_names)

    print("Extracting ground-truth labels...")
    y_true = np.concatenate([y.numpy() for x, y in test_ds], axis=0)

    print("Generating predictions with the model (this may take a few seconds)...")
    predictions = model.predict(test_ds)

    if num_classes == 1:
        # Binary classification with a single defined class (sigmoid or other output)
        y_pred = (predictions > 0.5).astype(int).reshape(-1)
    elif num_classes == 2 and predictions.shape[1] == 1:
        # Binary classification (sigmoid output)
        y_pred = (predictions > 0.5).astype(int).reshape(-1)
    else:
        # Multiclass classification (softmax output)
        y_pred = np.argmax(predictions, axis=1)

    print("\n" + "="*50)
    print("                 METRICS REPORT")
    print("="*50)

    # Compute the global metrics (weighted works well if there are imbalanced classes)
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    # 'macro' averages the classes without weighting by support. With the
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
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    # The MIL-level matrices live alongside the rest of the ladder results
    # (results/pil, results/hil), not mixed in with the checkpoints.
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    model_basename = os.path.basename(args.model_path)
    model_name_without_ext = os.path.splitext(model_basename)[0]
    cm_plot_name = f"confusion_matrix_{model_name_without_ext}.png"
    cm_plot_path = os.path.join(out_dir, cm_plot_name)

    plt.savefig(cm_plot_path)
    print(f"Confusion matrix plot saved to {cm_plot_path}")

    # CSV with the per-image prediction, with the same schema as pil_benchmark.py
    # (Image;RelPath;TrueLabel;TrueClass;PredLabel;PredClass;Correct) so that
    # MIL, SIL, PIL and the HIL subset (hil_subset==1) can be joined image by image.
    pred_csv_path = os.path.join(out_dir, f"predictions_{model_name_without_ext}.csv")
    file_paths = list(test_ds.file_paths)
    with open(pred_csv_path, "w", newline="", encoding="utf-8") as pred_file:
        pred_writer = csv.writer(pred_file, delimiter=";")
        pred_writer.writerow(["Image", "RelPath", "TrueLabel", "TrueClass",
                              "PredLabel", "PredClass", "Correct"])
        for fp, t_lbl, p_lbl in zip(file_paths, y_true, y_pred):
            t_int = int(t_lbl)
            p_int = int(p_lbl)
            pred_writer.writerow([
                os.path.basename(fp),
                os.path.relpath(fp, args.data_dir),
                t_int,
                _class_name(t_int, class_names),
                p_int,
                _class_name(p_int, class_names),
                1 if t_int == p_int else 0,
            ])
    print(f"Per-image predictions saved to {pred_csv_path}")

    # Besides the PNG, a JSON with the same figures is saved. The image is for
    # looking at, but not for comparing levels of the ladder or for another
    # script to read the numbers; the JSON is. Same naming criterion as
    # confusion_matrix_*.png: derived from the model, so that two runs do not overwrite
    # each other.
    metrics_path = os.path.join(out_dir, f"metrics_{model_name_without_ext}.json")

    # The checksum ties these metrics to the exact artifact that produced them.
    # If later on a run does not reproduce these figures, the hash says right
    # away whether the model changed or something else did (the data, the
    # preprocessing, the split). Without it, a regenerated .keras with the
    # same name is indistinguishable from the original.
    model_sha256 = _sha256_file(args.model_path)
    model_bytes = os.path.getsize(args.model_path) if os.path.isfile(args.model_path) else None
    print(f"Model SHA-256:        {model_sha256}")

    results = {
        "level": "MIL",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "path": args.model_path,
            "name": model_name_without_ext,
            "sha256": model_sha256,
            "bytes": model_bytes,
        },
        "dataset": {
            "data_dir": args.data_dir,
            "n_samples": int(len(y_true)),
            "class_names": list(class_names),
            "image_size": {"width": args.width, "height": args.height},
        },
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
        "artifacts": {
            "confusion_matrix_png": cm_plot_path,
            "predictions_csv": pred_csv_path,
        },
    }

    with open(metrics_path, "w", encoding="utf-8") as fh:
        # Output schema documented in docs/results_schema.md - update it if you add, rename or remove keys.
        json.dump(results, fh, indent=2, ensure_ascii=False, default=_json_safe)
    print(f"JSON metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
