r"""
pil_benchmark.py
================
Sends preprocessed images to the Arduino Portenta H7 over the serial port
for Processor-in-the-Loop (PIL) evaluation.

Protocol:
  - Start marker : '#' (0x23)
  - End marker   : '@' (0x40)
  - Payload bytes that match '#' (0x23), '@' (0x40) or ESC (0x1B)
    are escaped by sending: ESC (0x1B) followed by the byte XOR 0x20.

Basic usage (single image):
  python src/pil_benchmark.py --port COM3 --model_path models/tflite/model_int8.tflite --image path/image.png

Basic usage (held-out test split):
  python src/pil_benchmark.py --port COM3 --model_path models/tflite/model_int8.tflite --width 160 --height 120

The PIL level evaluates the test split held out by split_dataset.py, the same
one consumed by MIL and SIL, so that the gap between levels is attributable to
the level and not to a different sampling of images.

Options:
  --port        Serial port (default: COM9; e.g. COM3, /dev/ttyUSB0)
  --model_path  Path to the flashed .tflite model (required; hashed & used to tag outputs)
  --image       Path to a specific image to send
  --splits_dir  Directory with the splits produced by split_dataset.py
                (default: data/splits; the benchmark uses ONLY <splits_dir>/test)
  --folder      Explicit folder to send, instead of <splits_dir>/test
  --baud        Baud rate (default: 115200)
  --width       Target tensor width (default: 320)
  --height      Target tensor height (default: 320)
  --color       Send in RGB color (default: grayscale)
  --delay       Seconds to wait before the first transmission (default: 1.0)
  --gap         Seconds to wait between consecutive images (default: 0.5)
  --count       Number of random images to send from the folder (default: all)
"""

import argparse
from collections import Counter
import glob
import os
import random
import time
import sys
import csv
import hashlib
import json
import statistics
from datetime import datetime, timezone

try:
    import serial
except ImportError:
    print("ERROR: pyserial is not installed. Run: pip install pyserial")
    sys.exit(1)

try:
    from PIL import Image
    import numpy as np
except ImportError:
    print("ERROR: Pillow and/or numpy are not installed.")
    print("Run: pip install Pillow numpy")
    sys.exit(1)


MARKER_START = b'#'   # 0x23
MARKER_END   = b'@'   # 0x40
ESCAPE_BYTE  = 0x1B   # ESC
BYTES_TO_ESCAPE = {0x23, 0x40, 0x1B}  # '#', '@', ESC


def _json_safe(o):
    """Convert to native types whatever json.dump cannot serialize.

    classification_report(output_dict=True) returns numpy scalars (np.int64 in
    "support" and np.float64 in the metrics), which json.dump rejects with
    TypeError. Using the .tolist()/.item() duck-typing protocol handles both
    numpy arrays and numpy scalars cleanly.
    """
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    raise TypeError("Cannot serialize to JSON: %r (%s)" % (o, type(o).__name__))


def _sha256_archivo(path, chunk=1024 * 1024):
    """SHA-256 of the file, read in chunks.

    Returns None if the path is not a file: a JSON without a checksum is
    preferable to losing the whole benchmark on the last line, after having
    sent every image to the board.
    """
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for bloque in iter(lambda: fh.read(chunk), b""):
            h.update(bloque)
    return h.hexdigest()


def _resumen_latencias(filas, campos):
    """Per-field statistics from the CYC_*/US_*/TEMP_C lines.

    Values arrive as text over the serial port and may be missing (an image
    with no response leaves the field empty), so non-numeric ones are
    discarded instead of aborting the whole summary. A field with no valid
    measurement at all is left as None, which is information: it says the
    firmware did not report it.

    p95 by nearest rank over the sorted sample (no interpolation): with few
    images this is more honest than inventing an intermediate value.
    """
    resumen = {}
    for campo in campos:
        vals = []
        for fila in filas:
            try:
                vals.append(float(fila.get(campo, "")))
            except (TypeError, ValueError):
                continue
        if not vals:
            resumen[campo] = None
            continue
        ordenados = sorted(vals)
        idx95 = int(round(0.95 * (len(ordenados) - 1)))
        resumen[campo] = {
            "n": len(vals),
            "media": statistics.fmean(vals),
            "mediana": statistics.median(vals),
            "desv_std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "min": ordenados[0],
            "max": ordenados[-1],
            "p95": ordenados[idx95],
        }
    return resumen


def _nombre_clase(idx, class_names):
    """Class name from the index returned by the board.

    The firmware returns a raw integer. If it does not correspond to any known
    class (firmware flashed with another model, or a different dataset) it is
    marked as Invalida_<n> instead of blowing up the row: the CSV must record
    what happened, including an incoherent response.
    """
    if idx is None or idx < 0:
        return ""
    if idx < len(class_names):
        return class_names[idx]
    return "Invalida_%d" % idx


def escape_payload(raw_bytes: bytes) -> bytes:
    """
    Applies the escape scheme to the payload before sending it.
    If a byte is '#', '@' or ESC, it is sent as: ESC, (byte XOR 0x20).
    """
    escaped = bytearray()
    for b in raw_bytes:
        if b in BYTES_TO_ESCAPE:
            escaped.append(ESCAPE_BYTE)
            escaped.append(b ^ 0x20)
        else:
            escaped.append(b)
    return bytes(escaped)


def preprocess_image(image_path: str, width: int, height: int, grayscale: bool) -> bytes:
    """
    Loads, resizes and converts the image to the format expected by the tensor.
    Returns the RAW bytes (uint8, 0-255) in HWC order.
    """
    img = Image.open(image_path)

    if grayscale:
        img = img.convert('L')
    else:
        img = img.convert('RGB')

    img = img.resize((width, height), Image.LANCZOS)

    arr = np.array(img, dtype=np.uint8)  # shape: (H, W) or (H, W, 3)

    # Flatten in C order (row-major) -> tensor bytes
    raw = arr.flatten().tobytes()

    print(f"[INFO] Image preprocessed: {arr.shape}  -> {len(raw)} payload bytes")
    return raw


def build_packet(payload_raw: bytes) -> bytes:
    """
    Builds the full packet: MARKER_START + escaped_payload + MARKER_END
    """
    payload_escaped = escape_payload(payload_raw)
    packet = MARKER_START + payload_escaped + MARKER_END
    return packet


def send_image(port: str, baud: int, image_path: str,
               width: int, height: int, grayscale: bool,
               pre_delay: float, model_tag: str):
    """
    Opens the serial port and sends the image with the #...@ protocol
    """
    raw_payload = preprocess_image(image_path, width, height, grayscale)
    packet = build_packet(raw_payload)

    print(f"[INFO] Connecting to {port} @ {baud} baud ...")
    try:
        ser = serial.Serial(port, baud, timeout=10)
    except serial.SerialException as e:
        print(f"ERROR opening the port: {e}")
        sys.exit(1)

    print(f"[INFO] Port open. Waiting {pre_delay}s before sending...")
    time.sleep(pre_delay)  # wait for the Arduino to finish its setup

    ser.reset_input_buffer()

    print(f"[INFO] Sending packet: {len(packet)} bytes total "
          f"(raw payload={len(raw_payload)}, escaped={len(packet)-2})")

    ser.write(packet)
    ser.flush()
    print("[OK]   Packet sent. Waiting for the Arduino's response...\n")

    # Read the response until the "Listo" (ready) line is received
    timeout = time.time() + 30  # at most 30 seconds waiting for a response

    csv_path = os.path.join("results", "pil",
                            f"latency_metrics_single_{model_tag}.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    # Check if file exists to write header or not
    file_exists = os.path.isfile(csv_path)
    csv_file = open(csv_path, "a", newline="")
    csv_writer = csv.writer(csv_file, delimiter=";")
    metrics_headers = ["TEMP_C", "CYC_PRE", "CYC_INF", "CYC_POST", "CYC_TOTAL", "US_PRE", "US_INF", "US_POST", "US_TOTAL"]
    if not file_exists:
        csv_writer.writerow(["Image"] + metrics_headers)

    current_metrics = {}
    while time.time() < timeout:
        if ser.in_waiting:
            line = ser.readline().decode('utf-8', errors='replace').rstrip()
            print(f"  Arduino >>> {line}")

            if line.startswith("CYC_") or line.startswith("US_") or line.startswith("TEMP_C:"):
                parts = line.split(":")
                if len(parts) >= 2:
                    metric = parts[0].strip()
                    value = ":".join(parts[1:]).strip()
                    current_metrics[metric] = value

            # The firmware prints its end-of-inference markers in Spanish
            # ("siguiente imagen" = "next image", "listo" = "ready"), so these
            # checks must keep the Spanish words the board actually sends.
            if "siguiente imagen" in line.lower() or "listo" in line.lower():
                break
        else:
            time.sleep(0.05)

    row = [os.path.basename(image_path)]
    for h in metrics_headers:
        row.append(current_metrics.get(h, ""))
    csv_writer.writerow(row)
    csv_file.flush()

    csv_file.close()
    print(f"[INFO] Metrics saved to {csv_path}")

    ser.close()
    print("\n[INFO] Port closed. End of transmission.")


def send_folder(port: str, baud: int, folder: str,
                width: int, height: int, grayscale: bool,
                pre_delay: float, gap: float,
                model_tag: str,
                model_path: str = None,
                n_samples: int = None):
    """
    Sends the images in `folder` to the Arduino in sequence.

    If `n_samples` is None, every image found is sent.
    If `n_samples` is an integer, that number of images is selected
    at random (without replacement) before sending them.

    The Arduino returns the predicted class index along with on-device telemetry
    lines (TEMP_C, CYC_*, US_*). On completion, a prediction summary is printed
    and artifacts are saved under results/pil/ (latency_metrics_*.csv,
    predictions_*.csv, Matriz_*.png, and metrics_*.json).

    Parameters
    ----------
    port       : Serial port (e.g. 'COM9' or '/dev/ttyUSB0')
    baud       : Baud rate
    folder     : Folder containing the images
    width      : Input tensor width
    height     : Input tensor height
    grayscale  : True -> grayscale, False -> RGB
    pre_delay  : Seconds to wait before sending the first image
    gap        : Seconds to wait between consecutive images
    model_tag  : Model stem used to name output files under results/pil/
    model_path : Optional path to the flashed .tflite file (hashed in metrics JSON)
    n_samples  : Number of images to select at random.
                 If None (or >= total available), all of them are sent.
    """
    extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')

    subdirs = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
    subdirs.sort()

    image_data = [] # List of tuples: (image_path, true_label_int)
    class_names = []

    if not subdirs:
        print(f"[INFO] No subfolders found in {folder}. No confusion matrix will be generated.")
        for root, _, files in os.walk(folder):
            for f in files:
                if f.lower().endswith(extensions):
                    image_data.append((os.path.join(root, f), -1))
    else:
        class_names = subdirs
        print(f"[INFO] Ground-truth classes detected: {class_names}")
        for label_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(folder, class_name)
            for f in os.listdir(class_dir):
                if f.lower().endswith(extensions):
                    image_data.append((os.path.join(class_dir, f), label_idx))

    image_data = sorted(list(set(image_data)))

    if not image_data:
        print(f"[ERROR] No valid images found in: {folder}")
        sys.exit(1)

    available = len(image_data)
    if n_samples is not None and n_samples < available:
        image_data = random.sample(image_data, n_samples)
        print(f"[INFO] Selected {n_samples} images at random "
              f"out of {available} available in '{folder}'")
    else:
        print(f"[INFO] {available} images found in '{folder}' (sending all)")

    print(f"[INFO] Connecting to {port} @ {baud} baud ...")
    try:
        ser = serial.Serial(port, baud, timeout=10)
    except serial.SerialException as e:
        print(f"ERROR opening the port: {e}")
        sys.exit(1)

    print(f"[INFO] Port open. Waiting {pre_delay}s before sending...")
    time.sleep(pre_delay)

    total = len(image_data)

    csv_path = os.path.join("results", "pil",
                            f"latency_metrics_{model_tag}.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file, delimiter=";")
    metrics_headers = ["TEMP_C", "CYC_PRE", "CYC_INF", "CYC_POST", "CYC_TOTAL", "US_PRE", "US_INF", "US_POST", "US_TOTAL"]
    csv_writer.writerow(["Image"] + metrics_headers)

    # Separate CSV with the per-image prediction. Kept apart from the latency
    # one so as not to alter that file's schema, which already has finished
    # runs; both share the Image column and can be joined by name.
    # RelPath disambiguates: two classes may have files with the same name.
    pred_csv_path = os.path.join("results", "pil",
                                 f"predictions_{model_tag}.csv")
    pred_file = open(pred_csv_path, "w", newline="", encoding="utf-8")
    pred_writer = csv.writer(pred_file, delimiter=";")
    pred_writer.writerow(["Image", "RelPath", "TrueLabel", "TrueClass",
                          "PredLabel", "PredClass", "Correct"])

    class_counts: Counter = Counter()
    errors = 0
    y_true = []
    y_pred = []
    # Accumulated in addition to the CSV: the JSON summarizes per phase, and
    # re-reading the just-written CSV only for that would be a detour.
    latencias = []

    for idx, (image_path, true_label) in enumerate(image_data, start=1):
        print(f"\n[INFO] ({idx}/{total}) Processing: {os.path.basename(image_path)}")

        raw_payload = preprocess_image(image_path, width, height, grayscale)
        packet = build_packet(raw_payload)

        ser.reset_input_buffer()
        print(f"[INFO] Sending packet: {len(packet)} bytes total "
              f"(raw payload={len(raw_payload)}, escaped={len(packet)-2})")

        ser.write(packet)
        ser.flush()
        print("[OK]   Packet sent. Waiting for the predicted class...")

        # Read the response: the Arduino sends only the class number, then the latencies
        predicted_class = None
        current_metrics = {}
        timeout = time.time() + 30
        while time.time() < timeout:
            if ser.in_waiting:
                line = ser.readline().decode('utf-8', errors='replace').rstrip()
                print(f"  Arduino >>> {line}")

                if line.startswith("CYC_") or line.startswith("US_") or line.startswith("TEMP_C:"):
                    parts = line.split(":")
                    if len(parts) >= 2:
                        metric = parts[0].strip()
                        value = ":".join(parts[1:]).strip()
                        current_metrics[metric] = value

                # The firmware prints its end-of-inference markers in Spanish
                # ("siguiente imagen" = "next image", "listo" = "ready"), so these
                # checks must keep the Spanish words the board actually sends.
                if "siguiente imagen" in line.lower() or "listo" in line.lower():
                    break

                try:
                    # If the line is just an integer, assume it is the predicted class
                    parsed_int = int(line.strip())
                    predicted_class = parsed_int
                except ValueError:
                    pass
            else:
                time.sleep(0.05)

        row = [os.path.basename(image_path)]
        for h in metrics_headers:
            row.append(current_metrics.get(h, ""))
        csv_writer.writerow(row)
        csv_file.flush()
        latencias.append(dict(current_metrics))

        if predicted_class is not None:
            class_counts[predicted_class] += 1
            print(f"  -> Predicted class: {predicted_class}")
            if true_label != -1:
                y_true.append(true_label)
                y_pred.append(predicted_class)
        else:
            errors += 1
            print("  [WARN] No predicted class received for this image.")

        # One row per image, whether or not the label is known and whether or
        # not the board answered: the empty cells distinguish "there was no
        # label" from "there was no response", and both matter when
        # interpreting the benchmark.
        etiqueta_conocida = (true_label != -1)
        hubo_respuesta = (predicted_class is not None)
        if etiqueta_conocida and hubo_respuesta:
            correcto = 1 if true_label == predicted_class else 0
        else:
            correcto = ""
        pred_writer.writerow([
            os.path.basename(image_path),
            os.path.relpath(image_path, folder),
            true_label if etiqueta_conocida else "",
            _nombre_clase(true_label, class_names) if etiqueta_conocida else "",
            predicted_class if hubo_respuesta else "",
            _nombre_clase(predicted_class, class_names) if hubo_respuesta else "",
            correcto,
        ])
        # flush per row, like the latency CSV: a serial benchmark can be
        # interrupted and what has already been measured should not be lost.
        pred_file.flush()

        if idx < total:
            print(f"[INFO] Waiting {gap}s before the next image...")
            time.sleep(gap)

    csv_file.close()
    print(f"\n[INFO] Latency metrics saved to {csv_path}")
    pred_file.close()
    print(f"[INFO] Per-image predictions saved to {pred_csv_path}")

    ser.close()
    print("\n[INFO] Port closed. All images sent.")

    respondidas = total - errors
    print("\n" + "=" * 45)
    print("  PREDICTION SUMMARY")
    print(f"  Folder  : {folder}")
    print(f"  Total   : {total} images  |  Answered: {respondidas}  |  No response: {errors}")
    print("=" * 45)
    if class_counts:
        print(f"  {'Class':>6}  {'Count':>9}  {'Percentage':>11}")
        print("  " + "-" * 32)
        for cls in sorted(class_counts):
            cnt = class_counts[cls]
            pct = cnt / respondidas * 100 if respondidas else 0
            print(f"  {cls:>6}  {cnt:>9}  {pct:>10.1f}%")
    print("=" * 45)

    # Defaults: if there are no ground-truth labels or sklearn is missing, the
    # JSON is written anyway with these fields as null and the latency intact.
    metricas_globales = None
    report_dict = None
    matriz_confusion = None
    cm_plot_path = None

    # Generate the confusion matrix and metrics if we have ground-truth labels
    if y_true and len(y_true) > 0:
        try:
            import seaborn as sns
            import matplotlib.pyplot as plt
            from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

            print("\n" + "="*50)
            print("                 METRICS REPORT")
            print("="*50)

            accuracy = accuracy_score(y_true, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

            print(f"Accuracy:             {accuracy:.4f}")
            print(f"Precision:            {precision:.4f}")
            print(f"Recall:               {recall:.4f}")
            # macro in addition to weighted: same criterion as in MIL and SIL, so
            # that the columns of the ladder are comparable with each other.
            _, _, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
            print(f"F1-Score (weighted):  {f1:.4f}")
            print(f"F1-Score (macro):     {f1_macro:.4f}")

            labels_present = sorted(list(set(y_true + y_pred)))
            target_names = []
            for lab in labels_present:
                if lab < len(class_names):
                    target_names.append(class_names[lab])
                else:
                    target_names.append(f"Invalida_{lab}")

            print("\nDetailed Classification Report:")
            print(classification_report(y_true, y_pred, labels=labels_present, target_names=target_names, zero_division=0))

            print("\nGenerating Confusion Matrix...")
            cm = confusion_matrix(y_true, y_pred, labels=labels_present)

            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
            plt.title(f'PIL Confusion Matrix - {model_tag}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()

            # Same criterion as test_model.py (MIL): the name is derived from the
            # evaluated model, so that two runs with different models do not
            # overwrite each other's matrix.
            output_dir = os.path.join("results", "pil")
            os.makedirs(output_dir, exist_ok=True)
            cm_plot_name = f"Matriz_{model_tag}.png"
            cm_plot_path = os.path.join(output_dir, cm_plot_name)
            plt.savefig(cm_plot_path)
            print(f"Confusion matrix plot saved to: {cm_plot_path}")

            # Collected here, inside the try, so that the JSON further below
            # finds them already computed; if sklearn is missing they stay None.
            metricas_globales = {
                "accuracy": float(accuracy),
                "precision_weighted": float(precision),
                "recall_weighted": float(recall),
                "f1_weighted": float(f1),
                "f1_macro": float(f1_macro),
            }
            report_dict = classification_report(y_true, y_pred, labels=labels_present,
                                                target_names=target_names,
                                                zero_division=0, output_dict=True)
            matriz_confusion = {
                "labels": target_names,
                "matriz": cm.tolist(),
                "orden": "filas = etiqueta real, columnas = etiqueta predicha",
            }


        except ImportError:
            print("\n[WARN] Missing libraries for the confusion matrix (seaborn, sklearn, matplotlib).")
            print("Install them with: pip install scikit-learn seaborn matplotlib")

    # ------------------------------------------------------------------
    # Results JSON for the PIL level.
    #
    # It is ALWAYS written, even if there were no ground-truth labels or if
    # sklearn is missing: what this level contributes exclusively is the
    # latency measured on the board, and losing it because a confusion
    # matrix could not be computed would be absurd. The classification
    # fields are left as null when they could not be computed, which is
    # also information.
    # ------------------------------------------------------------------
    json_dir = os.path.join("results", "pil")
    os.makedirs(json_dir, exist_ok=True)
    metrics_path = os.path.join(json_dir, f"metrics_{model_tag}.json")

    resultados = {
        "nivel": "PIL",
        "generado_utc": datetime.now(timezone.utc).isoformat(),
        "modelo": {
            "path": model_path,
            "nombre": model_tag,
            "sha256": _sha256_archivo(model_path),
            "bytes": os.path.getsize(model_path) if model_path and os.path.isfile(model_path) else None,
            "nota": ("checksum del .tflite indicado con --model_path. El banco NO lo "
                     "carga ni lo compara contra la placa: acredita que este archivo "
                     "es el mismo que evaluo SIL, no que sea el que esta flasheado."),
        },
        "corrida": {
            "puerto": port,
            "baud": baud,
            "input_shape": {"width": width, "height": height,
                            "channels": 1 if grayscale else 3},
            "pre_delay_s": pre_delay,
            "gap_s": gap,
            "n_samples_pedidas": n_samples,
        },
        "dataset": {
            "folder": folder,
            "class_names": list(class_names),
            "n_imagenes": total,
            "n_respondidas": respondidas,
            "n_sin_respuesta": errors,
        },
        "metricas_globales": metricas_globales,
        "metricas_por_clase": report_dict,
        "matriz_confusion": matriz_confusion,
        "latencia": {
            "fuente": ("contador de ciclos DWT->CYCCNT del Cortex-M7, reportado por "
                       "el firmware en las lineas CYC_*/US_*"),
            "n_imagenes_con_medida": len(latencias),
            "metodo_p95": "rango mas cercano sobre la muestra ordenada, sin interpolar",
            "por_campo": _resumen_latencias(latencias, metrics_headers),
        },
        "conteo_predicciones": {str(k): v for k, v in sorted(class_counts.items())},
        "artefactos": {
            "matriz_png": cm_plot_path,
            "latencias_csv": csv_path,
            "predicciones_csv": pred_csv_path,
        },
    }

    with open(metrics_path, "w", encoding="utf-8") as fh:
        # Output schema documented in docs/results_schema.md - update it if you add, rename or remove keys.
        json.dump(resultados, fh, indent=2, ensure_ascii=False, default=_json_safe)
    print(f"[INFO] JSON metrics saved to {metrics_path}")



def parse_args():
    parser = argparse.ArgumentParser(
        description="Sends image(s) to the Arduino over serial (#...@ protocol)"
    )
    parser.add_argument('--port',   default="COM9", required=False,
                        help="Serial port (e.g. COM9 or /dev/ttyUSB0)")

    source = parser.add_mutually_exclusive_group()
    source.add_argument('--image',  default=None,
                        help="Path to a specific image to send")
    source.add_argument('--folder', default=None,
                        help="Explicit folder to send, instead of the test "
                             "split (default: <splits_dir>/test)")

    parser.add_argument('--splits_dir', default="data/splits",
                        help="Directory with the splits produced by split_dataset.py "
                             "(the PIL benchmark uses ONLY <splits_dir>/test)")

    parser.add_argument('--model_path', required=True,
                        help="Path to the .tflite flashed on the board. It is not "
                             "executed on the PC: its SHA-256/size are recorded in "
                             "metrics_*.json and its stem names the results/pil/ "
                             "artifacts (required)")

    parser.add_argument('--baud',   type=int, default=115200,
                        help="Baud rate (default: 115200)")
    parser.add_argument('--width',  type=int, default=320,
                        help="Input tensor width (default: 320)")
    parser.add_argument('--height', type=int, default=320,
                        help="Input tensor height (default: 320)")
    parser.add_argument('--color',  action='store_true',
                        help="Send in RGB color (default: grayscale)")
    parser.add_argument('--delay',  type=float, default=1.0,
                        help="Seconds to wait before sending (default: 1.0)")
    parser.add_argument('--gap',    type=float, default=0.5,
                        help="Seconds between consecutive images (default: 0.5)")
    parser.add_argument('--count',  type=int, default=None, metavar='N',
                        help="Number of random images to send from the folder. "
                             "If not specified, all of them are sent.")
    return parser.parse_args()


def normalize_port(port: str) -> str:
    """
    Normalizes the port name according to the platform:
      - Native Windows : COM9+ -> \\\\.\\COMx
      - WSL (Linux)    : COMx  -> /dev/ttySx  (COM1->ttyS1, COM7->ttyS7, etc.)
      - Native Linux   : used as is (/dev/ttyUSB0, etc.)
    """
    import re
    m = re.match(r'^COM(\d+)$', port.strip().upper())
    if m:
        n = int(m.group(1))
        # Detect WSL: sys.platform is 'linux' but /proc/version exists and contains 'microsoft'
        is_wsl = False
        if sys.platform != 'win32':
            try:
                with open('/proc/version', 'r') as f:
                    is_wsl = 'microsoft' in f.read().lower()
            except OSError:
                pass

        if is_wsl:
            port = f'/dev/ttyS{n}'
            print(f"[INFO] Port normalized for WSL: {port}")
        elif sys.platform == 'win32' and n >= 9:
            port = f'\\\\.\\COM{n}'
            print(f"[INFO] Port normalized for Windows: {port}")
    return port


def resolve_folder(args):
    """An explicit --folder wins; otherwise, the test split from split_dataset.py.

    PIL must never read the full dataset: it would include the images the
    model was trained on, and the SIL->PIL gap would no longer be
    attributable to the hardware.
    """
    if args.folder:
        return args.folder

    folder = os.path.join(args.splits_dir, 'test')
    if not os.path.isdir(folder):
        print(f"ERROR: test split not found at {folder}")
        print("Run the dataset split first:")
        print(f"  python src/split_dataset.py"
              f" --input_dir data/processed/{args.width}x{args.height}"
              f" --output_dir {args.splits_dir}")
        sys.exit(1)
    return folder


if __name__ == '__main__':
    args = parse_args()
    port = normalize_port(args.port)

    # Run tag: the model name without extension, the same criterion that
    # test_model.py uses to name its confusion matrix.
    model_tag = os.path.splitext(os.path.basename(args.model_path))[0]

    if args.image:
        send_image(
            port=port,
            baud=args.baud,
            image_path=args.image,
            width=args.width,
            height=args.height,
            grayscale=not args.color,
            pre_delay=args.delay,
            model_tag=model_tag,
        )
    else:
        folder = resolve_folder(args)
        send_folder(
            port=port,
            baud=args.baud,
            folder=folder,
            width=args.width,
            height=args.height,
            grayscale=not args.color,
            pre_delay=args.delay,
            gap=args.gap,
            model_tag=model_tag,
            model_path=args.model_path,
            n_samples=args.count,
        )
