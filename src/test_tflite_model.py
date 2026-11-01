import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone

try:
    import seaborn as sns
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
except ImportError:
    print("Please install scikit-learn and seaborn to compute the metrics.")
    print("Run: pip install scikit-learn seaborn")
    exit(1)


def _nombre_clase(idx, class_names):
    """Class name from the predicted or ground-truth index."""
    if idx is None or idx < 0:
        return ""
    if idx < len(class_names):
        return class_names[idx]
    return "Invalida_%d" % idx


def _sha256_archivo(path, chunk=1024 * 1024):
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
        for bloque in iter(lambda: fh.read(chunk), b""):
            h.update(bloque)
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
    parser = argparse.ArgumentParser(description="Test TFLite Model and Calculate Metrics exactly like test_model.py")
    parser.add_argument('--width', type=int, default=160, help="Image width")
    parser.add_argument('--height', type=int, default=120, help="Image height")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the .tflite model to evaluate (required)")
    parser.add_argument('--splits_dir', type=str, default="data/splits",
                        help="Directory holding the splits produced by split_dataset.py "
                             "(evaluation uses ONLY <splits_dir>/test)")
    parser.add_argument('--data_dir', type=str, default=None, help="Explicit path to the test split; defaults to <splits_dir>/test")
    parser.add_argument('--class_names_path', type=str, default="models/class_names.txt", help="Path to the txt file with the class names")
    parser.add_argument('--output_dir', type=str, default=os.path.join("results", "sil"), help="Output directory for the matrix, JSON and CSV (default: results/sil)")

    args = parser.parse_args()

    # SIL evaluates EXCLUSIVELY the test split, the exact same split consumed
    # by MIL, PIL, and HIL, so that jumps across ladder levels are attributable
    # to the deployment level rather than to a different image sample. Never
    # train (which calibrated INT8 quantization in quantize_int8_basic.py) and
    # never val (already consumed by early stopping and model selection).
    if args.data_dir is None:
        args.data_dir = os.path.join(args.splits_dir, "test")

    return args

def quantize_input(data, input_details):
    """
    If the TFLite model expects integer inputs instead of float32 (e.g. a
    full-INT8 model), scale and offset the input data manually according to
    its quantization parameters before invoking inference.
    """
    if input_details['dtype'] == np.int8:
        scale, zero_point = input_details['quantization']

        # TFLite returns iterable lists or numpy arrays
        s = scale[0] if isinstance(scale, (list, np.ndarray)) and len(scale) > 0 else scale
        z = zero_point[0] if isinstance(zero_point, (list, np.ndarray)) and len(zero_point) > 0 else zero_point

        if s > 0:
            # Map floating-point values into the quantized INT8 domain
            q_data = (data / s) + z
            q_data = np.clip(q_data, -128, 127).astype(np.int8)
            return q_data
        return data.astype(np.int8)
    # Return as float32 for non-INT8 input tensors
    return data.astype(np.float32)

def dequantize_output(data, output_details):
    """
    If the output tensor is quantized in INT8, restore it to floating-point
    scores by subtracting zero_point and multiplying by scale.
    """
    if output_details['dtype'] == np.int8:
        scale, zero_point = output_details['quantization']

        s = scale[0] if isinstance(scale, (list, np.ndarray)) and len(scale) > 0 else scale
        z = zero_point[0] if isinstance(zero_point, (list, np.ndarray)) and len(zero_point) > 0 else zero_point

        if s > 0:
            dq_data = (data.astype(np.float32) - z) * s
            return dq_data
    return data.astype(np.float32)

def main():
    args = parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: You must provide a valid TFLite model. Not found: {args.model_path}")
        return

    class_names = []
    if os.path.exists(args.class_names_path):
        with open(args.class_names_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines() if line.strip()]
        print(f"Class names loaded: {class_names}")

    print(f"\nInitializing TFLite Interpreter with {args.model_path}...")
    interpreter = tf.lite.Interpreter(model_path=args.model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    _, expected_height, expected_width, expected_channels = input_details['shape']

    # Strip surrounding whitespace or stray quotes from the path string
    if args.data_dir:
        args.data_dir = args.data_dir.strip().strip("'").strip('"')

    # If the path encodes dimensions in its name (legacy paths such as
    # data/processed/160x120), update them automatically to match the model
    if f"{args.width}x{args.height}" in args.data_dir and (args.width != expected_width or args.height != expected_height):
        args.data_dir = args.data_dir.replace(f"{args.width}x{args.height}", f"{expected_width}x{expected_height}")
        print(f"[INFO] Updating path resolution to {expected_width}x{expected_height} (required by the model).")

    # Input geometry is always dictated by the model's input tensor: the split
    # path no longer encodes resolution, so updating the path alone is not enough.
    if args.width != expected_width or args.height != expected_height:
        print(f"[INFO] Resizing inputs to {expected_width}x{expected_height} (required by the model).")
        args.width = expected_width
        args.height = expected_height

    # Fall back to <cwd>/<splits_dir>/test if the given path is missing
    if not os.path.exists(args.data_dir):
        fallback_path = os.path.join(os.getcwd(), args.splits_dir, "test")
        if os.path.exists(fallback_path):
            args.data_dir = fallback_path

    if not os.path.exists(args.data_dir):
        print(f"ERROR: test split not found at {args.data_dir}")
        print("Run the dataset split first:")
        print(f"  python src/split_dataset.py"
              f" --input_dir data/processed/{expected_width}x{expected_height}"
              f" --output_dir {args.splits_dir}")
        exit(1)

    print(f"Loading the test dataset from {args.data_dir}...")
    if class_names:
        subdirs = sorted(
            d for d in os.listdir(args.data_dir)
            if os.path.isdir(os.path.join(args.data_dir, d)) and not d.startswith(".")
        )
        if subdirs and set(subdirs) != set(class_names):
            print(
                f"[WARN] The subfolders of {args.data_dir} ({subdirs}) do not match "
                f"{args.class_names_path} ({class_names}); the class names will be inferred from the directory."
            )
            class_names = []
    # Use batch_size=1 because the TFLite interpreter typically runs single-frame
    # inference, and shuffle=False to keep inputs aligned with filenames/labels.
    test_ds = tf.keras.utils.image_dataset_from_directory(
        args.data_dir,
        labels='inferred',
        label_mode='int',
        class_names=class_names if class_names else None,
        color_mode='grayscale',
        batch_size=1,
        image_size=(args.height, args.width),
        shuffle=False
    )

    if not class_names:
        class_names = test_ds.class_names
        print(f"Class names inferred from the directory: {class_names}")

    num_classes = len(class_names)

    y_true = []
    predictions = []

    print("Generating frame-by-frame predictions with the TFLite model (sequential inference)...")

    for x, y in test_ds:
        y_true.append(y.numpy()[0])

        # x is originally in [0..255] float32; convert as required by the TFLite input tensor.
        input_data = x.numpy()

        # Reshape if necessary to match the exact expected input tensor shape
        if tuple(input_data.shape) != tuple(input_details['shape']):
            input_data = np.reshape(input_data, input_details['shape'])

        input_data = quantize_input(input_data, input_details)
        interpreter.set_tensor(input_details['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details['index'])
        output_data = dequantize_output(output_data, output_details)
        predictions.append(output_data[0])

    y_true = np.array(y_true)
    predictions = np.array(predictions)

    if num_classes == 1:
        # Single-output binary classification
        y_pred = (predictions > 0.5).astype(int).reshape(-1)
    elif num_classes == 2 and predictions.shape[1] == 1:
        # Sigmoid binary classification
        y_pred = (predictions > 0.5).astype(int).reshape(-1)
    else:
        # Multiclass argmax (softmax output)
        y_pred = np.argmax(predictions, axis=1)

    print("\n" + "="*50)
    print("              METRICS REPORT (TFLite)")
    print("="*50)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    # 'macro' averages classes without weighting by support. With an imbalanced
    # dataset, 'weighted' F1 is dominated by the majority class and can hide
    # poor minority-class performance; macro weights them equally. Reporting
    # both makes that gap visible instead of averaging it away.
    _, _, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)

    print(f"Accuracy:             {accuracy:.4f}")
    print(f"Precision:            {precision:.4f}")
    print(f"Recall:               {recall:.4f}")
    print(f"F1-Score (weighted):  {f1:.4f}")
    print(f"F1-Score (macro):     {f1_macro:.4f}")

    print("\nDetailed Classification Report:")
    # Requested twice: text for the console and dict for the JSON. Both come
    # from the exact same call with output_dict, so the figures cannot diverge.
    print(classification_report(y_true, y_pred, target_names=class_names, labels=range(len(class_names)), zero_division=0))
    report_dict = classification_report(y_true, y_pred, target_names=class_names, labels=range(len(class_names)), zero_division=0, output_dict=True)

    print("\nGenerating Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'SIL Confusion Matrix - {os.path.splitext(os.path.basename(args.model_path))[0]}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    # SIL-level matrices live alongside the rest of the ladder outputs
    # (results/mil, results/pil, results/hil), not next to the .tflite file.
    # Filenames derive from the model name, just as in test_model.py and
    # pil_benchmark.py, so runs with different models do not overwrite each other.
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    model_basename = os.path.basename(args.model_path)
    model_name_without_ext = os.path.splitext(model_basename)[0]
    cm_plot_name = f"Matriz_{model_name_without_ext}.png"
    cm_plot_path = os.path.join(out_dir, cm_plot_name)

    plt.savefig(cm_plot_path)
    print(f"\nConfusion matrix plot saved to {cm_plot_path}")

    # Per-image predictions CSV using the exact same schema as pil_benchmark.py
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
                _nombre_clase(t_int, class_names),
                p_int,
                _nombre_clase(p_int, class_names),
                1 if t_int == p_int else 0,
            ])
    print(f"Per-image predictions saved to {pred_csv_path}")

    # Alongside the PNG, a JSON with the exact same figures is written. The PNG
    # is for visual inspection, whereas the JSON enables automated comparisons
    # across ladder levels. Same model-derived naming convention as Matriz_*.png.
    metrics_path = os.path.join(out_dir, f"metrics_{model_name_without_ext}.json")

    # The checksum binds these metrics to the exact model artifact that produced
    # them. If a later run diverges, the hash immediately tells whether the
    # model binary changed or something else did (data, preprocessing, split).
    modelo_sha256 = _sha256_archivo(args.model_path)
    modelo_bytes = os.path.getsize(args.model_path) if os.path.isfile(args.model_path) else None
    print(f"Model SHA-256:        {modelo_sha256}")

    resultados = {
        "nivel": "SIL",
        "generado_utc": datetime.now(timezone.utc).isoformat(),
        "modelo": {
            "path": args.model_path,
            "nombre": model_name_without_ext,
            "sha256": modelo_sha256,
            "bytes": modelo_bytes,
        },
        "dataset": {
            "splits_dir": args.splits_dir,
            "data_dir": args.data_dir,
            "n_muestras": int(len(y_true)),
            "class_names": list(class_names),
            "image_size": {"width": args.width, "height": args.height},
        },
        "metricas_globales": {
            "accuracy": float(accuracy),
            "precision_weighted": float(precision),
            "recall_weighted": float(recall),
            "f1_weighted": float(f1),
            "f1_macro": float(f1_macro),
        },
        "metricas_por_clase": report_dict,
        "matriz_confusion": {
            "labels": list(class_names),
            "matriz": cm.tolist(),
            "orden": "filas = etiqueta real, columnas = etiqueta predicha",
        },
        "artefactos": {
            "matriz_png": cm_plot_path,
            "predicciones_csv": pred_csv_path,
        },
    }

    with open(metrics_path, "w", encoding="utf-8") as fh:
        # Output schema documented in docs/results_schema.md - update it if you add, rename or remove keys.
        json.dump(resultados, fh, indent=2, ensure_ascii=False, default=_json_safe)
    print(f"JSON metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
