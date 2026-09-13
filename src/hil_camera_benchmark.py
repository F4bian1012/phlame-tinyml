r"""
hil_camera_benchmark.py
=======================
Banco HIL REAL (camera-in-the-loop) para PHLAME — Fase 1C, camino 2.

A diferencia de `pil_benchmark.py` (banco PIL: la imagen viaja por serial),
este script cierra el lazo con el sensor fisico: la camara HM01B0 del
Portenta Vision Shield captura la escena real. El host orquesta el rig de
estimulo controlado (opcion (a) del checklist: monitor mostrando las
imagenes del dataset en secuencia temporizada), de modo que el ground truth
de cada inferencia es conocido por construccion:

    host                                    Portenta H7 + Vision Shield
    ----                                    ---------------------------
    1. muestra estimulo (clase conocida)
       en pantalla completa
    2. espera --settle s (estabilizacion
       de pantalla + auto-exposicion)
    3. envia trigger 'T'          ------>   4. captura frame HM01B0 (CAPTURE)
                                            5. resize+cuantiza (PRE)
                                            6. Invoke()          (INF)
                                            7. argmax            (POST)
    8. lee clase + telemetria     <------      TS_MS / clase / CYC_* / US_*
    9. empareja prediccion <-> etiqueta
       del estimulo presentado

Salidas (en --output-dir, default results/hil/):
  - HIL_Confusion_Matrix_<tag>.png : matriz de confusion del lazo sensor->prediccion
  - hil_latencies_<tag>.csv        : una fila por inferencia con latencias por fase
  - metrics_<tag>.json             : metricas (mismo esquema que MIL/SIL/PIL)
                               (CAPTURE/PRE/INF/POST/TOTAL), TEMP_C y ground truth
  - hil_conditions_<tag>.json      : condiciones del rig + huella del modelo
                               — el protocolo ambiental que pide un revisor

Firmware companion: deployment/hil_camera_firmware/hil_camera_firmware.ino
(sketch DEDICADO al banco HIL; el firmware PIL pil_firmware.ino y su banco
pil_benchmark.py quedan intactos y separados).

Protocolo serie:
  READY_HIL  <- handshake de arranque (la camara ya esta inicializada)
  'T'   -> dispara una captura + inferencia
  'F1'/'F0' -> activa/desactiva el frame-dump (validacion cruzada SIL)

Validacion cruzada (opcional, --sil-model / --mil-model): el firmware
devuelve cada frame capturado (comando F1) y el host ejecuta sobre ESE
MISMO frame los niveles inferiores de la escalera de fidelidad:
  MIL  (--mil-model, .keras float en PC)   <- test_model.py equivalente
  SIL  (--sil-model, .tflite INT8 en PC)   <- test_tflite_model.py equiv.
  HIL  (TFLM INT8 en el Cortex-M7)         <- este banco
Con los tres niveles sobre bytes identicos del sensor, la brecha se
descompone en componentes atribuibles:
  MIL vs SIL  = perdida por CUANTIZACION INT8 (mismo hardware, mismo frame)
  SIL vs HIL  = divergencia de EJECUCION chip/PC (TFLM vs TFLite, kernels)
  PIL vs HIL  = degradacion del FRONTEND fisico (pantalla/optica/sensor)

Uso tipico (escalera completa sobre el frame real):
  python src/hil_camera_benchmark.py --port COM9 \
      --folder data/processed/160x120 --count 100 \
      --settle 1.5 --lux 320 --distance-cm 25 \
      --sil-model models/tflite/model_int8.tflite \
      --mil-model models/checkpoints/best_model.keras

Requisitos: pyserial, opencv-python, numpy (+ scikit-learn, seaborn,
matplotlib para la matriz de confusion).
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
import hashlib


# ---------------------------------------------------------------------------
# Identidad del modelo (actividad 43)
#
# El firmware imprime al arrancar, ANTES de READY_HIL:
#     MODEL_BYTES:<g_model_len>      tamanio del array g_model[] en flash
#     MODEL_FNV1A:<hex32>            FNV-1a de 32 bits sobre esos bytes
#     ARENA_USED:<bytes>             interpreter->arena_used_bytes()
#
# g_model[] es el .tflite byte a byte (tflite_to_c.py), asi que la misma
# FNV-1a calculada en el PC sobre el .tflite debe coincidir EXACTAMENTE con
# la del chip. Si no coincide, la placa NO lleva el modelo que el operador
# cree, y el banco aborta antes de gastar la sesion de rig.
#
# FNV-1a y no SHA-256 porque el firmware no tiene mbedtls a mano y FNV son
# cuatro lineas de C; 32 bits bastan para distinguir 7 modelos.
# ---------------------------------------------------------------------------
FNV1A32_OFFSET = 0x811C9DC5
FNV1A32_PRIME = 0x01000193


def fnv1a32_bytes(data: bytes) -> int:
    h = FNV1A32_OFFSET
    for b in data:
        h ^= b
        h = (h * FNV1A32_PRIME) & 0xFFFFFFFF
    return h


def fnv1a32_file(path: str, chunk: int = 1 << 20) -> int:
    h = FNV1A32_OFFSET
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(chunk), b''):
            for b in block:
                h ^= b
                h = (h * FNV1A32_PRIME) & 0xFFFFFFFF
    return h


def sha256_file(path: str, chunk: int = 1 << 20):
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def parse_firmware_fingerprint(lines):
    """Extrae MODEL_BYTES / MODEL_FNV1A / ARENA_USED de las lineas de arranque.
    Devuelve dict con None en lo que el firmware no haya impreso (firmware viejo)."""
    fw = {'model_bytes': None, 'model_fnv1a': None, 'arena_used_bytes': None}
    for l in lines:
        if l.startswith("MODEL_BYTES:"):
            fw['model_bytes'] = int(l.split(":", 1)[1].strip())
        elif l.startswith("MODEL_FNV1A:"):
            fw['model_fnv1a'] = l.split(":", 1)[1].strip().lower().replace("0x", "")
        elif l.startswith("ARENA_USED:"):
            fw['arena_used_bytes'] = int(l.split(":", 1)[1].strip())
    return fw


def expected_model_fingerprint(tflite_path: str):
    """Huella del .tflite que el operador DICE haber flasheado."""
    if not tflite_path or not os.path.isfile(tflite_path):
        return None
    return {
        'path': os.path.abspath(tflite_path),
        'bytes': os.path.getsize(tflite_path),
        'fnv1a': f"{fnv1a32_file(tflite_path):08x}",
        'sha256': sha256_file(tflite_path),
    }


def previous_runs_fingerprints(output_dir: str):
    """Lee los hil_conditions_*.json ya existentes: (model_tag, fnv1a)."""
    out = []
    if not os.path.isdir(output_dir):
        return out
    for name in sorted(os.listdir(output_dir)):
        if name.startswith("hil_conditions_") and name.endswith(".json"):
            try:
                with open(os.path.join(output_dir, name), encoding='utf-8') as f:
                    c = json.load(f)
                fw = (c.get('firmware_model') or {})
                out.append((c.get('model_tag'), fw.get('model_fnv1a'), name))
            except Exception:
                continue
    return out


def verify_model_identity(fw, expected, model_tag, output_dir, allow_mismatch=False):
    """
    Tres comprobaciones, en este orden:
      1. el firmware imprimio huella (si no: firmware viejo, se avisa);
      2. la huella del chip == la del .tflite indicado (si no: ABORTA);
      3. ninguna corrida PREVIA en output_dir tiene la misma huella bajo
         OTRA etiqueta (si la hay: ABORTA — es exactamente el fallo
         'ResNet8 etiquetado como ResNet18').
    Devuelve True si la identidad quedo verificada positivamente.
    """
    ok = False
    if fw['model_fnv1a'] is None:
        print("[WARN] El firmware NO imprimio MODEL_FNV1A: es un firmware sin "
              "huella. La etiqueta de esta corrida NO esta verificada contra "
              "la placa. Reflashea hil_camera_firmware.ino actualizado.")
    elif expected is None:
        print("[WARN] Sin --tflite: no puedo comparar la huella del chip "
              f"({fw['model_fnv1a']}, {fw['model_bytes']} B) contra nada. "
              "Pasa --tflite <ruta> para verificar la identidad.")
    else:
        same = (fw['model_fnv1a'] == expected['fnv1a']
                and fw['model_bytes'] == expected['bytes'])
        print(f"[INFO] Huella chip   : fnv1a={fw['model_fnv1a']} bytes={fw['model_bytes']}")
        print(f"[INFO] Huella .tflite: fnv1a={expected['fnv1a']} bytes={expected['bytes']}  "
              f"({os.path.basename(expected['path'])})")
        if same:
            print("[OK]   IDENTIDAD VERIFICADA: la placa ejecuta exactamente el "
                  ".tflite indicado.")
            ok = True
        else:
            print("[ERROR] LA PLACA NO LLEVA ESE MODELO. La huella del chip no "
                  "coincide con el .tflite indicado. Regenera model.h con "
                  "tflite_to_c.py, recompila, reflashea y reintenta.")
            if not allow_mismatch:
                sys.exit(2)
            print("[WARN] --allow-model-mismatch: continuo bajo tu responsabilidad.")

    # 3) colision con corridas previas bajo otra etiqueta
    if fw['model_fnv1a'] is not None:
        for prev_tag, prev_fnv, fname in previous_runs_fingerprints(output_dir):
            if prev_fnv == fw['model_fnv1a'] and prev_tag and prev_tag != model_tag:
                print(f"[ERROR] La huella {fw['model_fnv1a']} ya existe en {fname} "
                      f"bajo la etiqueta '{prev_tag}'. Estas a punto de medir EL "
                      f"MISMO binario con la etiqueta '{model_tag}'. Reflashea.")
                if not allow_mismatch:
                    sys.exit(2)
    return ok


def check_required_conditions(args):
    """Actividad 50: sin condiciones fisicas la corrida no es reportable."""
    missing = [n for n, v in (("--lux", args.lux),
                              ("--distance-cm", args.distance_cm),
                              ("--ambient-temp", args.ambient_temp)) if v is None]
    if missing and not args.allow_missing_conditions:
        print("[ERROR] Faltan condiciones del rig: " + ", ".join(missing) + ".")
        print("        Dos corridas del mismo modelo difirieron 0,074 de F1-macro "
              "sin registro de la causa: sin estas variables el nivel HIL no es "
              "reportable. Pasalas, o usa --allow-missing-conditions para una "
              "prueba de humo.")
        sys.exit(2)
    if missing:
        print("[WARN] Condiciones incompletas (" + ", ".join(missing) +
              "): corrida marcada como NO reportable en hil_conditions.")
    if not args.notes.strip():
        print("[WARN] --notes vacio: anota al menos monitor, brillo y soporte.")

try:
    import serial
except ImportError:
    print("ERROR: pyserial no está instalado. Ejecuta: pip install pyserial")
    sys.exit(1)

try:
    import numpy as np
    import cv2
except ImportError:
    print("ERROR: OpenCV y/o numpy no están instalados.")
    print("Ejecuta: pip install opencv-python numpy")
    sys.exit(1)


# ---- Comandos del protocolo (ver hil_camera_firmware.ino) ----
CMD_TRIGGER   = b'T'
CMD_DUMP_ON   = b'F1'
CMD_DUMP_OFF  = b'F0'

# Escape del protocolo #...@ (identico a pil_benchmark.py / firmware)
MARKER_START = 0x23  # '#'
MARKER_END   = 0x40  # '@'
ESCAPE_BYTE  = 0x1B  # ESC

# Prefijos de telemetria que emite el firmware
TELEMETRY_RE = re.compile(r'^([A-Z][A-Z0-9_]*):(-?[\d.]+)$')
# Campos que exportamos al CSV (en este orden)
TELEMETRY_FIELDS = [
    'TS_MS', 'TEMP_C',
    'CYC_CAPTURE', 'CYC_PRE', 'CYC_INF', 'CYC_POST', 'CYC_TOTAL',
    'US_CAPTURE', 'US_PRE', 'US_INF', 'US_POST', 'US_TOTAL',
]
END_MARKERS = ("siguiente trigger", "siguiente imagen", "Listo")

WINDOW_NAME = "PHLAME HIL Stimulus"


def normalize_port(port: str) -> str:
    """
    Normaliza el nombre del puerto segun la plataforma (identico a
    pil_benchmark.py):
      - Windows nativo : COM9+ -> \\\\.\\COMx
      - WSL (Linux)    : COMx  -> /dev/ttySx
      - Linux nativo   : se usa tal cual (/dev/ttyUSB0, etc.)
    """
    m = re.match(r'^COM(\d+)$', port.strip().upper())
    if m:
        n = int(m.group(1))
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


def collect_from_manifest(manifest_path: str, use_all_test=False, seed=42):
    """
    Enumera (ruta_imagen, etiqueta_int) desde el manifiesto de la particion de
    test generado por src/split_dataset.py.

    Por defecto usa SOLO las filas con hil_subset == 1 (el subconjunto marcado
    para el rig fisico). Con use_all_test=True recorre las 1128 imagenes.

    Por que el manifiesto y no la carpeta:
      - El manifiesto es el UNICO lugar donde vive la marca hil_subset. Sin ese
        filtro la sesion de laboratorio pasa de ~200 a 1128 estimulos.
      - El orden de las clases se toma del propio manifiesto (label -> class_name
        ordenado por label entero), de modo que el indice coincide con el argmax
        que devuelve el firmware. Enumerar subcarpetas puede dar otro orden.
      - Deja constancia auditable de QUE imagenes se evaluaron: el CSV esta
        versionado en git.

    Retorna (image_data, class_names).
    """
    if not os.path.isfile(manifest_path):
        print(f"[ERROR] No existe el manifiesto: {manifest_path}")
        print("        Genera la particion primero:")
        print("          python src/split_dataset.py")
        print("        O apunta al manifiesto con --manifest RUTA.")
        sys.exit(1)

    with open(manifest_path, newline='', encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print(f"[ERROR] El manifiesto esta vacio: {manifest_path}")
        sys.exit(1)

    requeridas = {'class_name', 'label', 'src_path', 'dst_path'}
    faltan = requeridas - set(rows[0].keys())
    if faltan:
        print(f"[ERROR] El manifiesto no tiene las columnas {sorted(faltan)}.")
        print(f"        Columnas encontradas: {sorted(rows[0].keys())}")
        sys.exit(1)

    # Orden de clases desde el manifiesto: label entero -> nombre
    label_a_nombre = {}
    for r in rows:
        label_a_nombre[int(r['label'])] = r['class_name']
    class_names = [label_a_nombre[k] for k in sorted(label_a_nombre)]
    print(f"[INFO] Clases desde el manifiesto (orden = indice del firmware): "
          f"{class_names}")

    # Filtro del subconjunto marcado para el rig
    if use_all_test:
        sel = rows
        print(f"[INFO] --all-test: se usan las {len(sel)} imagenes de test.")
    else:
        if 'hil_subset' not in rows[0]:
            print("[ERROR] El manifiesto no tiene columna 'hil_subset'.")
            print("        Regenera la particion o usa --all-test.")
            sys.exit(1)
        sel = [r for r in rows
               if str(r['hil_subset']).strip() in ('1', 'True', 'true')]
        if not sel:
            print("[ERROR] Ninguna fila tiene hil_subset == 1.")
            print("        Regenera la particion o usa --all-test.")
            sys.exit(1)
        print(f"[INFO] Subconjunto HIL: {len(sel)} de {len(rows)} imagenes "
              f"(hil_subset == 1).")

    # Resolucion de rutas: dst_path (particion materializada) y si no, src_path
    image_data = []
    ausentes = []
    for r in sel:
        ruta = None
        for cand in (r['dst_path'], r['src_path']):
            if cand and os.path.isfile(cand):
                ruta = cand
                break
        if ruta is None:
            ausentes.append(r['dst_path'] or r['src_path'])
            continue
        image_data.append((ruta, int(r['label'])))

    if ausentes:
        print(f"[ERROR] {len(ausentes)} imagenes del manifiesto no existen en "
              f"disco. Primeras 5:")
        for p in ausentes[:5]:
            print(f"          {p}")
        print("        Materializa la particion con:")
        print("          python src/split_dataset.py")
        sys.exit(1)

    reparto = Counter(class_names[lbl] for _, lbl in image_data)
    print(f"[INFO] Reparto por clase: {dict(reparto)}")

    # Barajar SIEMPRE con semilla fija. El manifiesto viene agrupado por clase:
    # presentar en ese orden mostraria una clase con el chip frio y la otra
    # caliente, confundiendo la deriva termica con el efecto de clase y dejando
    # la matriz de confusion sin interpretacion posible.
    random.seed(seed)
    random.shuffle(image_data)
    print(f"[INFO] Orden de presentacion barajado (semilla {seed}).")
    return image_data, class_names


def collect_dataset(folder: str, n_samples=None, seed=None):
    """
    Enumera (ruta_imagen, etiqueta_int) desde subcarpetas por clase,
    igual que pil_benchmark.py. Retorna (image_data, class_names).
    """
    extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    subdirs = sorted(d for d in os.listdir(folder)
                     if os.path.isdir(os.path.join(folder, d)))
    if not subdirs:
        print(f"[ERROR] Se requieren subcarpetas por clase en {folder} "
              f"(sin ground truth no hay matriz de confusion HIL valida).")
        sys.exit(1)

    class_names = subdirs
    print(f"[INFO] Clases reales detectadas: {class_names}")

    image_data = []
    for label_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(folder, class_name)
        for f in os.listdir(class_dir):
            if f.lower().endswith(extensions):
                image_data.append((os.path.join(class_dir, f), label_idx))

    image_data = sorted(set(image_data))
    if not image_data:
        print(f"[ERROR] No se encontraron imágenes válidas en: {folder}")
        sys.exit(1)

    available = len(image_data)
    if seed is not None:
        random.seed(seed)
    if n_samples is not None and n_samples < available:
        image_data = random.sample(image_data, n_samples)
        print(f"[INFO] Seleccionadas {n_samples} imágenes aleatoriamente "
              f"de {available} disponibles")
    else:
        print(f"[INFO] {available} imágenes encontradas (usando todas)")

    # Barajar el orden de presentacion: evita bloques por clase, que
    # confundirian deriva temporal (temperatura, luz) con efecto de clase.
    random.shuffle(image_data)
    return image_data, class_names


# --------------------------------------------------------------------------
# Comunicacion serie
# --------------------------------------------------------------------------

def read_lines_until(ser, predicate, timeout_s=30.0, echo=True):
    """
    Lee lineas del serial hasta que predicate(line) sea True o expire el
    timeout. Retorna (lista_de_lineas, linea_que_cumplio | None).
    """
    lines = []
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if ser.in_waiting:
            line = ser.readline().decode('utf-8', errors='replace').rstrip()
            if not line:
                continue
            if echo:
                print(f"  Arduino >>> {line}")
            lines.append(line)
            if predicate(line):
                return lines, line
        else:
            time.sleep(0.02)
    return lines, None


def wait_ready(ser, timeout_s=30.0):
    """
    Espera el handshake de arranque del firmware HIL dedicado
    (hil_camera_firmware.ino): CAM_INIT:OK ... READY_HIL.
    Aborta si la camara fallo o si no responde (¿firmware PIL flasheado?).
    """
    lines, hit = read_lines_until(
        ser, lambda l: l == "READY_HIL" or l == "CAM_INIT:FAIL",
        timeout_s=timeout_s)
    if hit == "READY_HIL":
        print("[OK]   Firmware HIL listo (camara HM01B0 inicializada).")
        return parse_firmware_fingerprint(lines)
    if hit == "CAM_INIT:FAIL":
        print("[ERROR] La camara HM01B0 no inicializo (CAM_INIT:FAIL). "
              "¿Vision Shield conectado?")
    else:
        print("[ERROR] No llego READY_HIL. ¿Flasheaste "
              "deployment/hil_camera_firmware/hil_camera_firmware.ino? "
              "(el firmware PIL no responde a este banco). Si el firmware ya "
              "estaba corriendo, pulsa RESET en la placa y reintenta.")
    ser.close()
    sys.exit(1)


def parse_inference(lines):
    """
    Extrae (clase_predicha, telemetria_dict) de las lineas de una inferencia.
    La clase es el unico entero 'a pelo' en su propia linea (compat con
    pil_benchmark.py); la telemetria son lineas 'PREFIJO:valor'.
    """
    predicted = None
    telemetry = {}
    for line in lines:
        m = TELEMETRY_RE.match(line.strip())
        if m:
            key, value = m.group(1), m.group(2)
            if key in TELEMETRY_FIELDS:
                telemetry[key] = float(value) if '.' in value else int(value)
            continue
        try:
            predicted = int(line.strip())
        except ValueError:
            pass
    return predicted, telemetry


def trigger_inference(ser, timeout_s=30.0):
    """
    Dispara una captura HIL ('T') y parsea la respuesta completa.
    Retorna (clase_predicha | None, telemetria_dict).
    """
    ser.reset_input_buffer()
    ser.write(CMD_TRIGGER)
    ser.flush()
    lines, hit = read_lines_until(
        ser, lambda l: any(mk in l for mk in END_MARKERS),
        timeout_s=timeout_s)
    if hit is None:
        return None, {}
    return parse_inference(lines)


def read_frame(ser, timeout_s=20.0):
    """
    Lee el frame-dump que el firmware envia tras la inferencia HIL
    (FRAME_BEGIN:<n> + paquete #...@ escapado + FRAME_END).
    Retorna np.ndarray (120, 160) uint8, o None si fallo/timeout.
    Se llama DESPUES del end-marker de trigger_inference; el frame llega
    fuera de la region cronometrada, asi que no afecta las latencias.
    """
    # 1) Esperar la linea FRAME_BEGIN:<n>
    lines, hit = read_lines_until(
        ser, lambda l: l.startswith("FRAME_BEGIN:"),
        timeout_s=timeout_s, echo=False)
    if hit is None:
        print("  [WARN] No llego FRAME_BEGIN (¿frame-dump activado?).")
        return None
    n_expected = int(hit.split(':', 1)[1])

    # 2) Leer bytes crudos: '#' ... payload escapado ... '@'
    #    Maquina de estados byte a byte; el flag escape_pending sobrevive
    #    a los limites de chunk (un ESC puede llegar al final de un read()
    #    y su byte escapado en el siguiente).
    payload = bytearray()
    in_packet = False
    escape_pending = False
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        chunk = ser.read(max(1, ser.in_waiting or 1))
        if not chunk:
            continue
        for b in chunk:
            if not in_packet:
                if b == MARKER_START:
                    in_packet = True
                continue
            if escape_pending:
                payload.append(b ^ 0x20)
                escape_pending = False
            elif b == ESCAPE_BYTE:
                escape_pending = True
            elif b == MARKER_END:
                # 3) Consumir la linea FRAME_END que sigue
                read_lines_until(ser, lambda l: l == "FRAME_END",
                                 timeout_s=3, echo=False)
                if len(payload) != n_expected:
                    print(f"  [WARN] Frame incompleto: {len(payload)}/"
                          f"{n_expected} bytes.")
                    return None
                arr = np.frombuffer(bytes(payload), dtype=np.uint8)
                return arr.reshape(120, 160)  # HM01B0: 160x120 gris
            else:
                payload.append(b)
    print("  [WARN] Timeout leyendo el frame.")
    return None


# --------------------------------------------------------------------------
# Validacion cruzada: SIL en PC sobre el frame REAL capturado por la HM01B0
# --------------------------------------------------------------------------

class SilModel:
    """
    Ejecuta el MISMO modelo .tflite INT8 en el PC sobre el frame devuelto por
    el firmware (frame-dump). Replica exactamente el preprocesado del
    firmware (cameraFrameToImageBuffer + loadImageToInputTensor):
      - resize NEAREST desde 160x120 a la geometria del modelo
      - gris replicado a 3 canales si el modelo es RGB
      - pixel crudo [0,255] cuantizado con scale/zero_point del tensor
    Asi, cualquier desacuerdo HIL(chip) vs SIL(PC) sobre el mismo frame es
    atribuible a la EJECUCION (TFLM vs TFLite, kernels, redondeo), y el
    desacuerdo SIL(frame capturado) vs SIL/PIL(imagen limpia) es atribuible
    al FRONTEND fisico (pantalla, optica, sensor, iluminacion).
    """

    def __init__(self, model_path):
        try:
            from tensorflow import lite as tflite_mod  # noqa
            self._interp = tflite_mod.Interpreter(model_path=model_path)
        except ImportError:
            try:
                import tflite_runtime.interpreter as tflite_rt
                self._interp = tflite_rt.Interpreter(model_path=model_path)
            except ImportError:
                print("ERROR: se requiere tensorflow o tflite-runtime para "
                      "--sil-model. Ejecuta: pip install tensorflow")
                sys.exit(1)
        self._interp.allocate_tensors()
        self._in = self._interp.get_input_details()[0]
        self._out = self._interp.get_output_details()[0]
        shape = self._in['shape']  # [1, H, W, C]
        self.h, self.w, self.c = int(shape[1]), int(shape[2]), int(shape[3])
        print(f"[INFO] SIL-en-PC listo: {model_path} "
              f"(entrada {self.h}x{self.w}x{self.c})")

    def predict(self, frame_gray):
        """frame_gray: np.ndarray (120,160) uint8 -> clase predicha (int)."""
        # Resize NEAREST (identico al nearest-neighbor del firmware)
        img = cv2.resize(frame_gray, (self.w, self.h),
                         interpolation=cv2.INTER_NEAREST)
        if self.c == 3:
            img = np.stack([img] * 3, axis=-1)
        else:
            img = img[..., np.newaxis]
        x = img[np.newaxis].astype(np.float32)  # pixel crudo [0,255]

        if self._in['dtype'] == np.int8:
            scale, zp = self._in['quantization']
            q = np.round(x / scale) + zp
            x = np.clip(q, -128, 127).astype(np.int8)
        self._interp.set_tensor(self._in['index'], x)
        self._interp.invoke()
        y = self._interp.get_tensor(self._out['index'])
        return int(np.argmax(y))


class MilModel:
    """
    Nivel MIL de la escalera: el modelo .keras FLOAT ejecutado en el PC
    sobre el frame real capturado por la HM01B0 (equivalente en fidelidad a
    test_model.py, pero con datos del sensor en vez del dataset limpio).
    Mismo contrato de preprocesado que el firmware y SilModel: resize
    NEAREST, gris replicado si el modelo es RGB, pixel crudo [0,255]
    (el modelo se entreno SIN capa Rescaling; ver hil_camera_firmware.ino).
    """

    def __init__(self, model_path):
        try:
            import tensorflow as tf
        except ImportError:
            print("ERROR: se requiere tensorflow para --mil-model. "
                  "Ejecuta: pip install tensorflow")
            sys.exit(1)
        self._model = tf.keras.models.load_model(model_path, compile=False)
        shape = self._model.input_shape  # (None, H, W, C)
        self.h, self.w, self.c = int(shape[1]), int(shape[2]), int(shape[3])
        print(f"[INFO] MIL-en-PC listo: {model_path} "
              f"(entrada {self.h}x{self.w}x{self.c}, float32)")

    def predict(self, frame_gray):
        """frame_gray: np.ndarray (120,160) uint8 -> clase predicha (int)."""
        img = cv2.resize(frame_gray, (self.w, self.h),
                         interpolation=cv2.INTER_NEAREST)
        if self.c == 3:
            img = np.stack([img] * 3, axis=-1)
        else:
            img = img[..., np.newaxis]
        x = img[np.newaxis].astype(np.float32)  # pixel crudo [0,255]
        y = self._model.predict(x, verbose=0)
        return int(np.argmax(y))


# --------------------------------------------------------------------------
# Rig de estimulo: monitor controlado por el host
# --------------------------------------------------------------------------

def open_stimulus_window(fullscreen=True):
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    if fullscreen:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)


def show_stimulus(image_path, settle_s):
    """
    Muestra el estimulo centrado sobre fondo negro y espera settle_s
    (estabilizacion de la pantalla + auto-exposicion de la HM01B0).
    El escalado usa interpolacion NEAREST para no introducir contenido
    nuevo: la fidelidad la aporta la camara, no el monitor.
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return False
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    try:
        _, _, win_w, win_h = cv2.getWindowImageRect(WINDOW_NAME)
    except cv2.error:
        win_w, win_h = 1920, 1080
    if win_w <= 0 or win_h <= 0:
        win_w, win_h = 1920, 1080

    h, w = img.shape[:2]
    scale = min(win_w / w, win_h / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    canvas = np.zeros((max(win_h, new_h), max(win_w, new_w), 3), dtype=np.uint8)
    y0 = (canvas.shape[0] - new_h) // 2
    x0 = (canvas.shape[1] - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized

    cv2.imshow(WINDOW_NAME, canvas)
    # waitKey procesa el event-loop de la GUI; imprescindible para que la
    # imagen llegue realmente a la pantalla antes del trigger.
    cv2.waitKey(max(1, int(settle_s * 1000)))
    return True


# --------------------------------------------------------------------------
# Reporte
# --------------------------------------------------------------------------

def _json_safe(o):
    """numpy -> tipos nativos (classification_report devuelve np.int64/np.float64)."""
    try:
        import numpy as _np
        if isinstance(o, _np.integer):
            return int(o)
        if isinstance(o, _np.floating):
            return float(o)
        if isinstance(o, _np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    raise TypeError(f"no serializable: {type(o).__name__}")


def _latency_summary(rows):
    """Resumen por campo de telemetria, mismo formato que pil_benchmark.py."""
    out = {}
    for field in TELEMETRY_FIELDS:
        key = field.lower()
        vals = [r[key] for r in rows if r.get(key) is not None]
        if not vals:
            continue
        arr = np.sort(np.array(vals, dtype=float))
        k = max(0, min(len(arr) - 1, int(round(0.95 * (len(arr) - 1)))))
        out[field] = {'n': int(len(arr)), 'media': float(arr.mean()),
                      'mediana': float(np.median(arr)), 'desv_std': float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                      'min': float(arr.min()), 'max': float(arr.max()), 'p95': float(arr[k])}
    return out


def write_outputs(rows, class_names, conditions, output_dir, model_tag):
    """
    Actividad 55: TODAS las salidas llevan el model_tag en el nombre. Antes eran
    fijas (HIL_Confusion_Matrix.png, hil_latencies.csv, hil_conditions.json) y
    cada corrida borraba la anterior: la primera corrida de MobileNet se perdio asi.
    Actividad 63: ademas se escribe metrics_<tag>.json con el MISMO esquema que
    MIL/SIL/PIL (f1_macro, matriz cruda con orientacion declarada, sha256).
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- CSV de latencias sensor->prediccion ---
    csv_path = os.path.join(output_dir, f"hil_latencies_{model_tag}.csv")
    fieldnames = (['idx', 'utc_iso', 'image', 'true_label', 'true_class',
                   'pred_label', 'pred_class', 'sil_label', 'sil_class',
                   'mil_label', 'mil_class', 'frame_file']
                  + [f.lower() for f in TELEMETRY_FIELDS])
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK]   Latencias por fase guardadas en: {csv_path}")

    # --- Condiciones del rig (protocolo ambiental) ---
    cond_path = os.path.join(output_dir, f"hil_conditions_{model_tag}.json")
    with open(cond_path, 'w', encoding='utf-8') as f:
        json.dump(conditions, f, indent=2, ensure_ascii=False)
    print(f"[OK]   Condiciones del rig guardadas en: {cond_path}")

    # --- Metricas + matriz de confusion ---
    y_true = [r['true_label'] for r in rows if r['pred_label'] is not None]
    y_pred = [r['pred_label'] for r in rows if r['pred_label'] is not None]
    if not y_true:
        print("[WARN] Sin predicciones validas: no se genera matriz de confusion.")
        return

    try:
        import seaborn as sns
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.metrics import (accuracy_score, classification_report,
                                     confusion_matrix,
                                     precision_recall_fscore_support)

        print("\n" + "=" * 50)
        print("        REPORTE DE MÉTRICAS — HIL (camara en el lazo)")
        print("=" * 50)
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0)
        print(f"Accuracy (Exactitud):  {accuracy:.4f}")
        print(f"Precision:             {precision:.4f}")
        print(f"Recall (Exhaustividad):{recall:.4f}")
        print(f"F1-Score:              {f1:.4f}")

        labels_present = sorted(set(y_true + y_pred))
        target_names = [class_names[l] if l < len(class_names)
                        else f"Invalida_{l}" for l in labels_present]

        print("\nReporte de Clasificación Detallado:")
        print(classification_report(y_true, y_pred, labels=labels_present,
                                    target_names=target_names, zero_division=0))

        cm = confusion_matrix(y_true, y_pred, labels=labels_present)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_names, yticklabels=target_names)
        plt.title('HIL (Camera-in-the-Loop) Confusion Matrix')
        plt.ylabel('True Label (stimulus shown)')
        plt.xlabel('Predicted Label (HM01B0 -> Portenta H7)')
        plt.tight_layout()
        cm_path = os.path.join(output_dir, f"HIL_Confusion_Matrix_{model_tag}.png")
        plt.savefig(cm_path)
        plt.close()
        print(f"[OK]   Matriz de confusión HIL guardada en: {cm_path}")

        # --- metrics_<tag>.json: mismo esquema que MIL / SIL / PIL ---
        p_m, r_m, f_m, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0)
        report_dict = classification_report(
            y_true, y_pred, labels=labels_present, target_names=target_names,
            zero_division=0, output_dict=True)
        n_total = len(rows)
        n_resp = len(y_true)
        exp = conditions.get('expected_model') or {}
        resultados = {
            "nivel": "HIL",
            "generado_utc": datetime.now(timezone.utc).isoformat(),
            "modelo": {
                "path": exp.get('path'),
                "nombre": model_tag,
                "sha256": exp.get('sha256'),
                "bytes": exp.get('bytes'),
                "fnv1a_tflite": exp.get('fnv1a'),
                "fnv1a_chip": (conditions.get('firmware_model') or {}).get('model_fnv1a'),
                "identidad_verificada": conditions.get('model_identity_verified'),
                "nota": ("identidad_verificada=True significa que la huella FNV-1a del "
                         "g_model[] en la placa coincide con la del .tflite indicado: "
                         "el chip ejecuto exactamente ese archivo."),
            },
            "corrida": {
                "puerto": conditions.get('port'),
                "baud": conditions.get('baud'),
                "settle_s": conditions.get('settle_s'),
                "gap_s": conditions.get('gap_s'),
                "seed": conditions.get('seed'),
                "frame_dump": conditions.get('frame_dump'),
                "condiciones_completas": conditions.get('conditions_complete'),
                "illuminance_lux": conditions.get('illuminance_lux'),
                "camera_to_screen_distance_cm": conditions.get('camera_to_screen_distance_cm'),
                "ambient_temp_c": conditions.get('ambient_temp_c'),
            },
            "dataset": {
                "stimulus_source": conditions.get('stimulus_source'),
                "manifest_subset": conditions.get('manifest_subset'),
                "class_names": list(class_names),
                "n_estimulos": n_total,
                "n_respondidas": n_resp,
                "n_sin_respuesta": n_total - n_resp,
            },
            "metricas_globales": {
                "accuracy": float(accuracy),
                "precision_weighted": float(precision),
                "recall_weighted": float(recall),
                "f1_weighted": float(f1),
                "precision_macro": float(p_m),
                "recall_macro": float(r_m),
                "f1_macro": float(f_m),
            },
            "metricas_por_clase": report_dict,
            "matriz_confusion": {
                "labels": target_names,
                "matriz": cm.tolist(),
                "orden": "filas = etiqueta real, columnas = etiqueta predicha",
            },
            "latencia": {
                "fuente": ("contador de ciclos DWT->CYCCNT del Cortex-M7, reportado por "
                           "el firmware en las lineas CYC_*/US_*; incluye la fase CAPTURE"),
                "n_imagenes_con_medida": sum(1 for r in rows if r.get('us_total') is not None),
                "metodo_p95": "rango mas cercano sobre la muestra ordenada, sin interpolar",
                "por_campo": _latency_summary(rows),
            },
            "validacion_cruzada": {
                "sil_en_pc": conditions.get('sil_model'),
                "mil_en_pc": conditions.get('mil_model'),
                "n_con_sil": sum(1 for r in rows if r.get('sil_label') is not None),
                "n_con_mil": sum(1 for r in rows if r.get('mil_label') is not None),
            },
            "artefactos": {
                "matriz_png": cm_path,
                "latencias_csv": csv_path,
                "condiciones_json": cond_path,
            },
        }
        metrics_path = os.path.join(output_dir, f"metrics_{model_tag}.json")
        with open(metrics_path, "w", encoding="utf-8") as fh:
            json.dump(resultados, fh, indent=2, ensure_ascii=False, default=_json_safe)
        print(f"[OK]   Métricas (JSON, mismo esquema que MIL/SIL/PIL) en: {metrics_path}")
        print(f"       F1-macro = {f_m:.4f}   errores = {int(cm.sum() - np.trace(cm))}")
    except ImportError:
        print("\n[WARN] Faltan librerías para la matriz de confusión.")
        print("Instálalas con: pip install scikit-learn seaborn matplotlib")

    # --- Resumen de latencias por fase ---
    def stats(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        if not vals:
            return None
        arr = np.array(vals, dtype=float) / 1000.0  # us -> ms
        return arr.mean(), arr.std(), arr.min(), arr.max()

    print("\n" + "=" * 60)
    print("  LATENCIA POR FASE (sensor→predicción)   [ms]")
    print(f"  {'Fase':<10} {'media':>9} {'std':>9} {'min':>9} {'max':>9}")
    print("  " + "-" * 50)
    for key, name in [('us_capture', 'CAPTURE'), ('us_pre', 'PRE'),
                      ('us_inf', 'INF'), ('us_post', 'POST'),
                      ('us_total', 'TOTAL')]:
        s = stats(key)
        if s:
            print(f"  {name:<10} {s[0]:>9.3f} {s[1]:>9.3f} "
                  f"{s[2]:>9.3f} {s[3]:>9.3f}")
    print("=" * 60)

    # --- Validacion cruzada: escalera MIL/SIL/HIL sobre el MISMO frame ---
    # Solo cuentan los frames donde estan disponibles TODOS los niveles
    # activados, para que las accuracies sean comparables entre si.
    levels = [('pred_label', 'HIL (TFLM INT8, Cortex-M7)')]
    if any(r.get('sil_label') is not None for r in rows):
        levels.insert(0, ('sil_label', 'SIL (TFLite INT8, PC)'))
    if any(r.get('mil_label') is not None for r in rows):
        levels.insert(0, ('mil_label', 'MIL (Keras float32, PC)'))

    if len(levels) > 1:
        keys = [k for k, _ in levels]
        paired = [r for r in rows
                  if all(r.get(k) is not None for k in keys)]
        if paired:
            n = len(paired)
            print("\n" + "=" * 64)
            print("  ESCALERA DE FIDELIDAD — mismo frame capturado por la HM01B0")
            print(f"  Frames emparejados: {n}")
            print(f"\n  {'Nivel':<30} {'Accuracy':>10}")
            print("  " + "-" * 42)
            for key, label in levels:
                acc = sum(1 for r in paired
                          if r[key] == r['true_label']) / n
                print(f"  {label:<30} {acc:>10.4f}")

            print(f"\n  {'Acuerdo por pares':<38} {'%':>7}   atribuible a")
            print("  " + "-" * 60)
            pair_meaning = {
                ('mil_label', 'sil_label'): 'cuantizacion INT8',
                ('sil_label', 'pred_label'): 'ejecucion chip vs PC',
                ('mil_label', 'pred_label'): 'cuantizacion + ejecucion',
            }
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    a, b = keys[i], keys[j]
                    agree = sum(1 for r in paired if r[a] == r[b])
                    la = levels[i][1].split(' ')[0]
                    lb = levels[j][1].split(' ')[0]
                    meaning = pair_meaning.get((a, b), '')
                    print(f"  {la} vs {lb:<32} {agree / n * 100:>6.1f}   "
                          f"{meaning}")
            print("\n  → La caída de TODOS los niveles vs sus accuracies PIL/")
            print("    dataset limpio = degradación del FRONTEND físico")
            print("    (pantalla/óptica/sensor/iluminación): la brecha PIL→HIL.")
            print("=" * 64)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def run_benchmark(args):
    port = normalize_port(args.port)

    # --- Actividad 50: condiciones fisicas obligatorias (falla ANTES del rig) ---
    check_required_conditions(args)

    # --- Actividad 43/55: etiqueta del modelo y .tflite esperado ---
    tflite_path = args.tflite or args.sil_model
    if args.model_tag:
        model_tag = args.model_tag
    elif tflite_path:
        model_tag = os.path.splitext(os.path.basename(tflite_path))[0]
    else:
        print("[ERROR] Necesito saber que modelo va en la placa para nombrar las "
              "salidas: pasa --tflite <ruta al .tflite flasheado> (recomendado, "
              "ademas verifica la huella) o al menos --model-tag <nombre>.")
        sys.exit(2)
    expected = expected_model_fingerprint(tflite_path)
    if tflite_path and expected is None:
        print(f"[ERROR] No existe el .tflite indicado: {tflite_path}")
        sys.exit(2)
    if args.folder:
        print(f"[WARN] --folder pasado explicitamente: se usa {args.folder} "
              f"y NO se aplica el filtro hil_subset del manifiesto.")
        fuente = f"folder:{os.path.abspath(args.folder)}"
        image_data, class_names = collect_dataset(args.folder, args.count,
                                                  args.seed)
    else:
        fuente = f"manifest:{os.path.abspath(args.manifest)}"
        image_data, class_names = collect_from_manifest(
            args.manifest, use_all_test=args.all_test, seed=args.seed)
        if args.count is not None and args.count < len(image_data):
            random.seed(args.seed)
            image_data = random.sample(image_data, args.count)
            print(f"[INFO] --count {args.count}: recortado a {args.count} "
                  f"estimulos.")

    print(f"[INFO] Conectando a {port} @ {args.baud} baud ...")
    try:
        ser = serial.Serial(port, args.baud, timeout=10)
    except serial.SerialException as e:
        print(f"ERROR al abrir el puerto: {e}")
        sys.exit(1)

    print(f"[INFO] Puerto abierto. Esperando {args.delay}s (setup del firmware)...")
    time.sleep(args.delay)

    # Handshake con el firmware HIL dedicado (hil_camera_firmware.ino)
    fw = wait_ready(ser)
    identity_ok = verify_model_identity(fw, expected, model_tag, args.output_dir,
                                        allow_mismatch=args.allow_model_mismatch)

    # Validacion cruzada: frame-dump + MIL/SIL en PC
    sil_model = None
    mil_model = None
    dump_frames = (args.dump_frames or args.sil_model is not None
                   or args.mil_model is not None)
    frames_dir = None
    if args.sil_model:
        sil_model = SilModel(args.sil_model)
    if args.mil_model:
        mil_model = MilModel(args.mil_model)
    if dump_frames:
        ser.write(CMD_DUMP_ON)
        ser.flush()
        read_lines_until(ser, lambda l: l == "FRAME_DUMP:ON", timeout_s=5,
                         echo=False)
        # Frames bajo data/ (git-ignored segun checklist; solo trazabilidad)
        frames_dir = args.frames_dir
        os.makedirs(frames_dir, exist_ok=True)
        print(f"[INFO] Frame-dump activado; frames en: {frames_dir}")

    # Rig de estimulo: ventana en el monitor controlada por el host
    open_stimulus_window(fullscreen=not args.windowed)
    print("[INFO] Ventana de estímulo abierta. Alinea la cámara con la "
          "pantalla (distancia y encuadre fijos) y verifica el enfoque.")
    if not args.no_confirm:
        input("       Pulsa ENTER para iniciar el barrido de estímulos... ")

    conditions = {
        'rig': 'monitor-controlled-stimulus (checklist Fase 1C, opcion a)',
        'utc_start': datetime.now(timezone.utc).isoformat(),
        'model_tag': model_tag,
        'firmware_model': fw,
        'expected_model': expected,
        'model_identity_verified': identity_ok,
        'conditions_complete': all(v is not None for v in
                                   (args.lux, args.distance_cm, args.ambient_temp)),
        'port': args.port,
        'baud': args.baud,
        'stimulus_source': fuente,
        'manifest': (os.path.abspath(args.manifest)
                     if not args.folder else None),
        'manifest_subset': ('all-test' if args.all_test else 'hil_subset==1')
                           if not args.folder else None,
        'dataset_folder': (os.path.abspath(args.folder)
                           if args.folder else None),
        'class_names': class_names,
        'n_stimuli': len(image_data),
        'settle_s': args.settle,
        'gap_s': args.gap,
        'seed': args.seed,
        'illuminance_lux': args.lux,
        'camera_to_screen_distance_cm': args.distance_cm,
        'ambient_temp_c': args.ambient_temp,
        'frame_dump': dump_frames,
        'sil_model': (os.path.abspath(args.sil_model)
                      if args.sil_model else None),
        'mil_model': (os.path.abspath(args.mil_model)
                      if args.mil_model else None),
        'frames_dir': (os.path.abspath(frames_dir) if frames_dir else None),
        'notes': args.notes,
    }

    rows = []
    counts = Counter()
    errors = 0
    total = len(image_data)

    try:
        for idx, (image_path, true_label) in enumerate(image_data, start=1):
            name = os.path.basename(image_path)
            print(f"\n[INFO] ({idx}/{total}) Estímulo: {name} "
                  f"(clase real: {class_names[true_label]})")

            if not show_stimulus(image_path, args.settle):
                print(f"  [WARN] No se pudo cargar {image_path}; se omite.")
                errors += 1
                continue

            predicted, telemetry = trigger_inference(ser)

            # Frame-dump + MIL/SIL en PC sobre el frame REAL capturado
            sil_label = None
            mil_label = None
            frame_file = None
            if dump_frames and predicted is not None:
                frame = read_frame(ser)
                if frame is not None:
                    frame_file = f"frame_{idx:04d}_{name}.png"
                    cv2.imwrite(os.path.join(frames_dir, frame_file), frame)
                    if sil_model is not None:
                        sil_label = sil_model.predict(frame)
                    if mil_model is not None:
                        mil_label = mil_model.predict(frame)

            row = {
                'idx': idx,
                'utc_iso': datetime.now(timezone.utc).isoformat(),
                'image': name,
                'true_label': true_label,
                'true_class': class_names[true_label],
                'pred_label': predicted,
                'pred_class': (class_names[predicted]
                               if predicted is not None
                               and predicted < len(class_names)
                               else None),
                'sil_label': sil_label,
                'sil_class': (class_names[sil_label]
                              if sil_label is not None
                              and sil_label < len(class_names)
                              else None),
                'mil_label': mil_label,
                'mil_class': (class_names[mil_label]
                              if mil_label is not None
                              and mil_label < len(class_names)
                              else None),
                'frame_file': frame_file,
            }
            for field in TELEMETRY_FIELDS:
                row[field.lower()] = telemetry.get(field)
            rows.append(row)

            if predicted is not None:
                counts[predicted] += 1
                marker = "✓" if predicted == true_label else "✗"
                us_total = telemetry.get('US_TOTAL')
                lat = (f" | sensor→pred: {us_total / 1000.0:.2f} ms"
                       if us_total else "")
                xval = ""
                if sil_label is not None:
                    agree = "=" if sil_label == predicted else "≠"
                    xval += f" | SIL(PC): {row['sil_class']} {agree} HIL"
                if mil_label is not None:
                    agree = "=" if mil_label == predicted else "≠"
                    xval += f" | MIL(PC): {row['mil_class']} {agree} HIL"
                print(f"  {marker} Predicho: {row['pred_class']}{lat}{xval}")
            else:
                errors += 1
                print("  [WARN] Sin respuesta del firmware para este estímulo.")

            if idx < total and args.gap > 0:
                time.sleep(args.gap)
    except KeyboardInterrupt:
        print("\n[WARN] Interrumpido por el usuario; se guardan los datos "
              "recolectados hasta ahora.")
    finally:
        conditions['utc_end'] = datetime.now(timezone.utc).isoformat()
        # Apagar frame-dump y cerrar limpio
        try:
            if dump_frames:
                ser.write(CMD_DUMP_OFF)
                ser.flush()
                read_lines_until(ser, lambda l: l == "FRAME_DUMP:OFF",
                                 timeout_s=3, echo=False)
        except serial.SerialException:
            pass
        ser.close()
        cv2.destroyAllWindows()

    respondidas = len(rows) - sum(1 for r in rows if r['pred_label'] is None)
    print("\n" + "=" * 45)
    print("  RESUMEN — BANCO HIL (cámara en el lazo)")
    print(f"  Estímulos: {total}  |  Respondidos: {respondidas}  "
          f"|  Fallos: {errors}")
    print("=" * 45)

    write_outputs(rows, class_names, conditions, args.output_dir, model_tag)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Banco HIL real: la HM01B0 captura estímulos presentados "
                    "en el monitor; empareja predicción↔ground truth y mide "
                    "latencia sensor→predicción por fases.")
    parser.add_argument('--port', default="COM9",
                        help="Puerto serial (ej. COM9 o /dev/ttyUSB0)")
    parser.add_argument('--baud', type=int, default=115200,
                        help="Baudrate (default: 115200)")
    parser.add_argument('--manifest',
                        default=os.path.join('data', 'splits',
                                             'test_manifest.csv'),
                        help="Manifiesto de la particion de test "
                             "(default: data/splits/test_manifest.csv). "
                             "Fuente por defecto de los estimulos.")
    parser.add_argument('--all-test', action='store_true',
                        help="Usar las 1128 imagenes de test en vez de las "
                             "marcadas con hil_subset == 1 (~200). Multiplica "
                             "por ~5.6 la duracion de la sesion.")
    parser.add_argument('--folder', default=None,
                        help="OVERRIDE explicito: dataset con subcarpetas por "
                             "clase. Si se pasa, se ignora el manifiesto y NO "
                             "se aplica el filtro hil_subset.")
    parser.add_argument('--count', type=int, default=None, metavar='N',
                        help="Nº de estímulos aleatorios (default: todos)")
    parser.add_argument('--seed', type=int, default=42,
                        help="Semilla para muestreo/orden reproducibles "
                             "(default: 42)")
    parser.add_argument('--settle', type=float, default=1.5,
                        help="Segundos de estabilización tras mostrar cada "
                             "estímulo, antes del trigger (default: 1.5)")
    parser.add_argument('--gap', type=float, default=0.5,
                        help="Segundos entre estímulos (default: 0.5)")
    parser.add_argument('--delay', type=float, default=2.0,
                        help="Espera inicial tras abrir el puerto (default: 2.0)")
    parser.add_argument('--output-dir', default=os.path.join('results', 'hil'),
                        help="Directorio de salida (default: results/hil)")
    parser.add_argument('--windowed', action='store_true',
                        help="Ventana normal en vez de pantalla completa")
    parser.add_argument('--no-confirm', action='store_true',
                        help="No esperar ENTER antes de iniciar el barrido")
    # Condiciones ambientales del rig (checklist Fase 1C)
    parser.add_argument('--lux', type=float, default=None,
                        help="Iluminancia medida en el plano de la pantalla [lux]")
    parser.add_argument('--distance-cm', type=float, default=None,
                        help="Distancia cámara↔pantalla [cm]")
    parser.add_argument('--ambient-temp', type=float, default=None,
                        help="Temperatura ambiente [°C]")
    # Identidad del modelo (actividad 43) y nombres de salida (actividad 55)
    parser.add_argument('--tflite', default=None, metavar='TFLITE',
                        help="Ruta al .tflite INT8 que generó el model.h "
                             "flasheado. Se compara su huella FNV-1a con la "
                             "que imprime el firmware al arrancar y el banco "
                             "ABORTA si no coinciden. Si no se pasa, se usa "
                             "--sil-model como referencia.")
    parser.add_argument('--model-tag', default=None, metavar='NOMBRE',
                        help="Etiqueta para los nombres de salida (default: "
                             "nombre del .tflite sin extensión).")
    parser.add_argument('--allow-model-mismatch', action='store_true',
                        help="NO abortar si la huella del chip no coincide "
                             "(solo depuración; la corrida queda marcada).")
    parser.add_argument('--allow-missing-conditions', action='store_true',
                        help="NO abortar si faltan --lux/--distance-cm/"
                             "--ambient-temp (solo pruebas de humo).")
    parser.add_argument('--notes', default="",
                        help="Notas libres del montaje (encuadre, soporte, "
                             "modelo de monitor, brillo...)")
    # Validacion cruzada HIL(chip) vs SIL(PC) sobre el mismo frame
    parser.add_argument('--dump-frames', action='store_true',
                        help="Recuperar y guardar cada frame capturado por "
                             "la HM01B0 (trazabilidad)")
    parser.add_argument('--sil-model', default=None, metavar='TFLITE',
                        help="Ruta al MISMO .tflite INT8 del firmware: se "
                             "ejecuta en el PC sobre cada frame capturado "
                             "(implica --dump-frames). Aisla la divergencia "
                             "de ejecucion chip vs PC.")
    parser.add_argument('--mil-model', default=None, metavar='KERAS',
                        help="Ruta al .keras float original: se ejecuta en "
                             "el PC sobre cada frame capturado (implica "
                             "--dump-frames). Junto a --sil-model aisla la "
                             "perdida por cuantizacion INT8.")
    parser.add_argument('--frames-dir', default=os.path.join('data',
                                                             'hil_frames'),
                        help="Directorio de frames capturados, git-ignored "
                             "(default: data/hil_frames)")
    return parser.parse_args()


if __name__ == '__main__':
    run_benchmark(parse_args())
