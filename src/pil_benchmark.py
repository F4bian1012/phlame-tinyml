r"""
send_multiple_images_serial.py
================================
Envía imágenes preprocesadas al Arduino Portenta H7 por puerto serial.

Protocolo:
  - Marcador de inicio : '#' (0x23)
  - Marcador de fin    : '@' (0x40)
  - Los bytes del payload que coincidan con '#' (0x23), '@' (0x40) o ESC (0x1B)
    se escapan enviando: ESC (0x1B) seguido del byte XOR 0x20.

Uso básico (una imagen):
  python send_multiple_images_serial.py --port COM3 --image ruta/imagen.png

Uso básico (carpeta completa):
  python pil_benchmark.py --port COM3

El nivel PIL evalúa la partición de test reservada por split_dataset.py, la misma
que consumen MIL y SIL, para que la brecha entre niveles sea atribuible al nivel
y no a un muestreo distinto de imágenes.

Opciones:
  --port        Puerto serial (ej. COM3, /dev/ttyUSB0)  [requerido]
  --image       Ruta a una imagen concreta a enviar
  --splits_dir  Directorio de particiones de split_dataset.py
                (default: data\splits; el banco usa SOLO <splits_dir>\test)
  --folder      Carpeta explícita a enviar, en vez de <splits_dir>\test
  --baud    Tasa de baudios (default: 115200)
  --width   Ancho objetivo del tensor (default: 120)
  --height  Alto objetivo del tensor (default: 160)
  --color   Enviar en color RGB (default: escala de grises)
  --delay   Segundos de espera antes de la primera transmisión (default: 1.0)
  --gap     Segundos de espera entre imágenes consecutivas (default: 0.5)
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
    print("ERROR: pyserial no está instalado. Ejecuta: pip install pyserial")
    sys.exit(1)

try:
    from PIL import Image
    import numpy as np
except ImportError:
    print("ERROR: Pillow y/o numpy no están instalados.")
    print("Ejecuta: pip install Pillow numpy")
    sys.exit(1)


MARKER_START = b'#'   # 0x23
MARKER_END   = b'@'   # 0x40
ESCAPE_BYTE  = 0x1B   # ESC
BYTES_TO_ESCAPE = {0x23, 0x40, 0x1B}  # '#', '@', ESC


def _json_safe(o):
    """Convierte a tipos nativos lo que json.dump no sabe serializar.

    classification_report(output_dict=True) devuelve numpy: np.int64 en
    "support" y np.float64 en las metricas, y json.dump falla con TypeError.
    No se importa numpy aqui porque en este script solo entra dentro de las
    funciones; se usa el protocolo .tolist()/.item(), que es justo lo que
    exponen tanto los escalares como los arrays de numpy.
    """
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    raise TypeError("No se puede serializar a JSON: %r (%s)" % (o, type(o).__name__))


def _sha256_archivo(path, chunk=1024 * 1024):
    """SHA-256 del archivo, leido por bloques.

    Devuelve None si la ruta no es un archivo: es preferible un JSON sin
    checksum que perder el banco entero en la ultima linea, despues de haber
    enviado todas las imagenes a la placa.
    """
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for bloque in iter(lambda: fh.read(chunk), b""):
            h.update(bloque)
    return h.hexdigest()


def _resumen_latencias(filas, campos):
    """Estadisticos por campo a partir de las lineas CYC_*/US_*/TEMP_C.

    Los valores llegan como texto por el puerto serie y pueden faltar (una
    imagen sin respuesta deja el campo vacio), asi que se descartan los no
    numericos en vez de abortar el resumen entero. Un campo sin ninguna
    medida valida queda en None, que es informacion: dice que el firmware no
    lo reporto.

    p95 por rango mas cercano sobre la muestra ordenada (sin interpolar): con
    pocas imagenes es mas honesto que inventar un valor intermedio.
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
    """Nombre de la clase a partir del indice que devuelve la placa.

    El firmware devuelve un entero crudo. Si no corresponde a ninguna clase
    conocida (firmware flasheado con otro modelo, o dataset distinto) se marca
    como Invalida_<n> en vez de reventar la fila: el CSV debe registrar lo que
    paso, incluida una respuesta incoherente.
    """
    if idx is None or idx < 0:
        return ""
    if idx < len(class_names):
        return class_names[idx]
    return "Invalida_%d" % idx


def escape_payload(raw_bytes: bytes) -> bytes:
    """
    Aplica el esquema de escape al payload antes de enviarlo.
    Si un byte es '#', '@' o ESC, se envía como: ESC, (byte XOR 0x20).
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
    Carga, redimensiona y convierte la imagen al formato esperado por el tensor.
    Retorna los bytes RAW (uint8, 0-255) en orden HWC.
    """
    img = Image.open(image_path)

    if grayscale:
        img = img.convert('L')
    else:
        img = img.convert('RGB')

    img = img.resize((width, height), Image.LANCZOS)

    arr = np.array(img, dtype=np.uint8)  # shape: (H, W) o (H, W, 3)

    # Aplanar en orden C (row-major) → bytes del tensor
    raw = arr.flatten().tobytes()

    print(f"[INFO] Imagen preprocesada: {arr.shape}  -> {len(raw)} bytes de payload")
    return raw


def build_packet(payload_raw: bytes) -> bytes:
    """
    Construye el paquete completo: MARKER_START + payload_escapado + MARKER_END
    """
    payload_escaped = escape_payload(payload_raw)
    packet = MARKER_START + payload_escaped + MARKER_END
    return packet


def send_image(port: str, baud: int, image_path: str,
               width: int, height: int, grayscale: bool,
               pre_delay: float, model_tag: str):
    """
    Abre el puerto serial y envía la imagen con el protocolo #...@
    """
    raw_payload = preprocess_image(image_path, width, height, grayscale)
    packet = build_packet(raw_payload)

    print(f"[INFO] Conectando a {port} @ {baud} baud ...")
    try:
        ser = serial.Serial(port, baud, timeout=10)
    except serial.SerialException as e:
        print(f"ERROR al abrir el puerto: {e}")
        sys.exit(1)

    time.sleep(pre_delay)  # esperar a que el Arduino termine su setup
    print(f"[INFO] Puerto abierto. Esperando {pre_delay}s antes de enviar...")
    time.sleep(pre_delay)

    ser.reset_input_buffer()

    print(f"[INFO] Enviando paquete: {len(packet)} bytes totales "
          f"(payload raw={len(raw_payload)}, escapado={len(packet)-2})")

    ser.write(packet)
    ser.flush()
    print(f"[OK]   Paquete enviado. Esperando respuesta del Arduino...\n")

    # Leer la respuesta hasta que se reciba la línea de "Listo"
    timeout = time.time() + 30  # máximo 30 segundos esperando respuesta

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
    print(f"[INFO] Métricas guardadas en {csv_path}")

    ser.close()
    print("\n[INFO] Puerto cerrado. Fin de transmisión.")


def send_folder(port: str, baud: int, folder: str,
                width: int, height: int, grayscale: bool,
                pre_delay: float, gap: float,
                model_tag: str,
                model_path: str = None,
                n_samples: int = None):
    """
    Envía imágenes de `folder` al Arduino en secuencia.

    Si `n_samples` es None envía todas las imágenes encontradas.
    Si `n_samples` es un entero, selecciona ese número de imágenes
    aleatoriamente (sin reemplazo) antes de enviarlas.

    El Arduino retorna únicamente el número entero de la clase predicha.
    Al finalizar, imprime un resumen con el conteo y porcentaje por clase.

    Parámetros
    ----------
    port      : Puerto serial (ej. 'COM9' o '/dev/ttyUSB0')
    baud      : Baudrate
    folder    : Carpeta que contiene las imágenes
    width     : Ancho del tensor de entrada
    height    : Alto del tensor de entrada
    grayscale : True → escala de grises, False → RGB
    pre_delay : Segundos de espera antes de enviar la primera imagen
    gap       : Segundos de espera entre imágenes consecutivas
    n_samples : Número de imágenes a seleccionar aleatoriamente.
                Si es None (o >= total disponibles), se envían todas.
    """
    extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')

    subdirs = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
    subdirs.sort()

    image_data = [] # Lista de tuplas: (ruta_imagen, etiqueta_real_int)
    class_names = []

    if not subdirs:
        print(f"[INFO] No se encontraron subcarpetas en {folder}. No se generará matriz de confusión.")
        for root, _, files in os.walk(folder):
            for f in files:
                if f.lower().endswith(extensions):
                    image_data.append((os.path.join(root, f), -1))
    else:
        class_names = subdirs
        print(f"[INFO] Clases reales detectadas: {class_names}")
        for label_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(folder, class_name)
            for f in os.listdir(class_dir):
                if f.lower().endswith(extensions):
                    image_data.append((os.path.join(class_dir, f), label_idx))

    image_data = sorted(list(set(image_data)))

    if not image_data:
        print(f"[ERROR] No se encontraron imágenes válidas en: {folder}")
        sys.exit(1)

    available = len(image_data)
    if n_samples is not None and n_samples < available:
        image_data = random.sample(image_data, n_samples)
        print(f"[INFO] Seleccionadas {n_samples} imágenes aleatoriamente "
              f"de {available} disponibles en '{folder}'")
    else:
        print(f"[INFO] {available} imágenes encontradas en '{folder}' (enviando todas)")

    print(f"[INFO] Conectando a {port} @ {baud} baud ...")
    try:
        ser = serial.Serial(port, baud, timeout=10)
    except serial.SerialException as e:
        print(f"ERROR al abrir el puerto: {e}")
        sys.exit(1)

    print(f"[INFO] Puerto abierto. Esperando {pre_delay}s antes de enviar...")
    time.sleep(pre_delay)

    total = len(image_data)

    csv_path = os.path.join("results", "pil",
                            f"latency_metrics_{model_tag}.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file, delimiter=";")
    metrics_headers = ["TEMP_C", "CYC_PRE", "CYC_INF", "CYC_POST", "CYC_TOTAL", "US_PRE", "US_INF", "US_POST", "US_TOTAL"]
    csv_writer.writerow(["Image"] + metrics_headers)

    # CSV aparte con la prediccion por imagen. Separado del de latencias
    # para no alterar el esquema de aquel, que ya tiene corridas hechas;
    # ambos comparten la columna Image y se pueden cruzar por nombre.
    # RelPath desambigua: dos clases pueden tener archivos homonimos.
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
    # Se acumulan ademas del CSV: el JSON resume por fase y releer el CSV
    # recien escrito solo para eso seria dar un rodeo.
    latencias = []

    for idx, (image_path, true_label) in enumerate(image_data, start=1):
        print(f"\n[INFO] ({idx}/{total}) Procesando: {os.path.basename(image_path)}")

        raw_payload = preprocess_image(image_path, width, height, grayscale)
        packet = build_packet(raw_payload)

        ser.reset_input_buffer()
        print(f"[INFO] Enviando paquete: {len(packet)} bytes totales "
              f"(payload raw={len(raw_payload)}, escapado={len(packet)-2})")

        ser.write(packet)
        ser.flush()
        print(f"[OK]   Paquete enviado. Esperando clase predicha...")

        # Leer la respuesta: el Arduino envía solo el número de clase y luego latencias
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

                if "siguiente imagen" in line.lower() or "listo" in line.lower():
                    break

                try:
                    # Si la linea es solo un numero entero, asumimos que es la clase predicha
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
            print(f"  → Clase predicha: {predicted_class}")
            if true_label != -1:
                y_true.append(true_label)
                y_pred.append(predicted_class)
        else:
            errors += 1
            print(f"  [WARN] No se recibió clase predicha para esta imagen.")

        # Una fila por imagen, se sepa o no la etiqueta y responda o no la
        # placa: las celdas vacias distinguen "no habia etiqueta" de "no
        # hubo respuesta", y ambas cosas importan al interpretar el banco.
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
        # flush por fila, igual que el CSV de latencias: un banco por serie
        # puede interrumpirse y lo ya medido no deberia perderse.
        pred_file.flush()

        if idx < total:
            print(f"[INFO] Esperando {gap}s antes de la siguiente imagen...")
            time.sleep(gap)

    csv_file.close()
    print(f"\n[INFO] Métricas de latencia guardadas en {csv_path}")
    pred_file.close()
    print(f"[INFO] Predicciones por imagen guardadas en {pred_csv_path}")

    ser.close()
    print("\n[INFO] Puerto cerrado. Todas las imágenes enviadas.")

    respondidas = total - errors
    print("\n" + "=" * 45)
    print(f"  RESUMEN DE PREDICCIONES")
    print(f"  Carpeta : {folder}")
    print(f"  Total   : {total} imágenes  |  Respondidas: {respondidas}  |  Sin respuesta: {errors}")
    print("=" * 45)
    if class_counts:
        print(f"  {'Clase':>6}  {'Cantidad':>9}  {'Porcentaje':>11}")
        print("  " + "-" * 32)
        for cls in sorted(class_counts):
            cnt = class_counts[cls]
            pct = cnt / respondidas * 100 if respondidas else 0
            print(f"  {cls:>6}  {cnt:>9}  {pct:>10.1f}%")
    print("=" * 45)

    # Valores por defecto: si no hay etiquetas reales o falta sklearn, el JSON
    # se escribe igual con estos campos en null y la latencia intacta.
    metricas_globales = None
    report_dict = None
    matriz_confusion = None
    cm_plot_path = None

    # Generación de la matriz de confusión y métricas si tenemos verdaderas etiquetas
    if y_true and len(y_true) > 0:
        try:
            import seaborn as sns
            import matplotlib.pyplot as plt
            from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

            print("\n" + "="*50)
            print("                 REPORTE DE MÉTRICAS")
            print("="*50)

            accuracy = accuracy_score(y_true, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

            print(f"Accuracy (Exactitud): {accuracy:.4f}")
            print(f"Precision:            {precision:.4f}")
            print(f"Recall (Exhaustividad):{recall:.4f}")
            # macro ademas de weighted: mismo criterio que en MIL y SIL, para que
            # las columnas de la escalera sean comparables entre si.
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

            print("\nReporte de Clasificación Detallado:")
            print(classification_report(y_true, y_pred, labels=labels_present, target_names=target_names, zero_division=0))

            print("\nGenerando Matriz de Confusión...")
            cm = confusion_matrix(y_true, y_pred, labels=labels_present)

            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
            plt.title(f'PIL Confusion Matrix - {model_tag}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()

            # Mismo criterio que test_model.py (MIL): el nombre se deriva del
            # modelo evaluado, para que dos corridas con modelos distintos no se
            # pisen la matriz la una a la otra.
            output_dir = os.path.join("results", "pil")
            os.makedirs(output_dir, exist_ok=True)
            cm_plot_name = f"Matriz_{model_tag}.png"
            cm_plot_path = os.path.join(output_dir, cm_plot_name)
            plt.savefig(cm_plot_path)
            print(f"Gráfico de la matriz de confusión guardado en: {cm_plot_path}")

            # Se recogen aqui, dentro del try, para que el JSON de mas abajo
            # los encuentre ya calculados; si sklearn falta quedan en None.
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
            print("\n[WARN] Faltan librerías para matriz de confusión (seaborn, sklearn, matplotlib).")
            print("Instálalas con: pip install scikit-learn seaborn matplotlib")

    # ------------------------------------------------------------------
    # JSON de resultados del nivel PIL.
    #
    # Se escribe SIEMPRE, incluso si no hubo etiquetas reales o si falta
    # sklearn: lo que este nivel aporta de forma exclusiva es la latencia
    # medida en la placa, y perderla porque no se pudo calcular una matriz de
    # confusion seria absurdo. Los campos de clasificacion quedan en null
    # cuando no se pudieron calcular, que tambien es informacion.
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
        json.dump(resultados, fh, indent=2, ensure_ascii=False, default=_json_safe)
    print(f"[INFO] Métricas en JSON guardadas en {metrics_path}")



def parse_args():
    parser = argparse.ArgumentParser(
        description="Envía imagen(es) al Arduino por serial (protocolo #...@)"
    )
    parser.add_argument('--port',   default="COM9", required=False,
                        help="Puerto serial (ej. COM9 o /dev/ttyUSB0)")

    source = parser.add_mutually_exclusive_group()
    source.add_argument('--image',  default=None,
                        help="Ruta a una imagen concreta a enviar")
    source.add_argument('--folder', default=None,
                        help="Carpeta explicita a enviar, en vez de la particion "
                             "de test (default: <splits_dir>/test)")

    parser.add_argument('--splits_dir', default="data/splits",
                        help="Directorio con las particiones de split_dataset.py "
                             "(el banco PIL usa SOLO <splits_dir>/test)")

    parser.add_argument('--model_path', required=True,
                        help="Ruta al .tflite que esta flasheado en la placa. No se "
                             "carga: solo identifica la corrida, igual que en "
                             "test_model.py, para nombrar la matriz y el CSV de "
                             "latencias (obligatorio)")

    parser.add_argument('--baud',   type=int, default=115200,
                        help="Baudrate (default: 115200)")
    parser.add_argument('--width',  type=int, default=320,
                        help="Ancho del tensor de entrada (default: 320)")
    parser.add_argument('--height', type=int, default=320,
                        help="Alto del tensor de entrada (default: 320)")
    parser.add_argument('--color',  action='store_true',
                        help="Enviar en color RGB (default: escala de grises)")
    parser.add_argument('--delay',  type=float, default=1.0,
                        help="Segundos de espera antes de enviar (default: 1.0)")
    parser.add_argument('--gap',    type=float, default=0.5,
                        help="Segundos entre imágenes consecutivas (default: 0.5)")
    parser.add_argument('--count',  type=int, default=None, metavar='N',
                        help="Número de imágenes aleatorias a enviar de la carpeta. "
                             "Si no se especifica, se envían todas.")
    return parser.parse_args()


def normalize_port(port: str) -> str:
    """
    Normaliza el nombre del puerto según la plataforma:
      - Windows nativo : COM9+ → \\\\.\\COMx
      - WSL (Linux)    : COMx  → /dev/ttySx  (COM1→ttyS1, COM7→ttyS7, etc.)
      - Linux nativo   : se usa tal cual (/dev/ttyUSB0, etc.)
    """
    import re
    m = re.match(r'^COM(\d+)$', port.strip().upper())
    if m:
        n = int(m.group(1))
        # Detectar WSL: sys.platform es 'linux' pero existe /proc/version con 'microsoft'
        is_wsl = False
        if sys.platform != 'win32':
            try:
                with open('/proc/version', 'r') as f:
                    is_wsl = 'microsoft' in f.read().lower()
            except OSError:
                pass

        if is_wsl:
            port = f'/dev/ttyS{n}'
            print(f"[INFO] Puerto normalizado para WSL: {port}")
        elif sys.platform == 'win32' and n >= 9:
            port = f'\\\\.\\COM{n}'
            print(f"[INFO] Puerto normalizado para Windows: {port}")
    return port


def resolve_folder(args):
    """--folder explicito manda; si no, la particion de test de split_dataset.py.

    PIL nunca debe leer el dataset completo: incluiria las imagenes con las que
    se entreno el modelo, y la brecha SIL->PIL dejaria de ser atribuible al
    hardware.
    """
    if args.folder:
        return args.folder

    folder = os.path.join(args.splits_dir, 'test')
    if not os.path.isdir(folder):
        print(f"ERROR: no se encontro la particion de prueba en {folder}")
        print("Ejecuta primero la particion del dataset:")
        print(f"  python src/split_dataset.py"
              f" --input_dir data/processed/{args.width}x{args.height}"
              f" --output_dir {args.splits_dir}")
        sys.exit(1)
    return folder


if __name__ == '__main__':
    args = parse_args()
    port = normalize_port(args.port)

    # Etiqueta de la corrida: el nombre del modelo sin extension, el mismo
    # criterio que usa test_model.py para nombrar su matriz de confusion.
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
