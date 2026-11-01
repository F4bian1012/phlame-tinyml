r"""
hil_camera_benchmark.py
=======================
Real Camera-in-the-Loop (HIL) benchmark for PHLAME — Phase 1C, Path 2.

Unlike `pil_benchmark.py` (PIL benchmark: preprocessed images sent over serial),
this script closes the loop through the physical optical sensor: the HM01B0
camera on the Portenta Vision Shield captures the real scene. The host PC
orchestrates the controlled-stimulus rig (monitor displaying dataset images in
a timed sequence), so the ground-truth label of every inference is known by
construction:

    host                                    Portenta H7 + Vision Shield
    ----                                    ---------------------------
    1. displays stimulus (known class)
       in fullscreen
    2. waits --settle s (display refresh
       + camera auto-exposure settling)
    3. sends trigger 'T'          ------>   4. captures HM01B0 frame (CAPTURE)
                                            5. resizes + quantizes   (PRE)
                                            6. Invoke()              (INF)
                                            7. argmax                (POST)
    8. reads class + telemetry    <------      TS_MS / class / CYC_* / US_*
    9. pairs prediction <-> label
       of the displayed stimulus

Outputs (in --output-dir, default results/hil/):
  - HIL_Confusion_Matrix_<tag>.png : sensor->prediction confusion matrix
  - hil_latencies_<tag>.csv        : one row per inference with per-phase
                                     latencies (CAPTURE/PRE/INF/POST/TOTAL),
                                     TEMP_C, and ground truth
  - metrics_<tag>.json             : metrics JSON (same schema as MIL/SIL/PIL)
  - hil_conditions_<tag>.json      : physical rig conditions + model fingerprint
                                     — the environmental protocol required for
                                     reproducible reporting

Companion firmware: deployment/hil_camera_firmware/hil_camera_firmware.ino
(dedicated HIL sketch; the PIL sketch pil_firmware.ino and its host benchmark
pil_benchmark.py remain separate and untouched).

Serial protocol:
  READY_HIL   <- boot handshake (camera initialized and ready)
  'T'         -> triggers one capture + inference cycle
  'F1' / 'F0' -> enables / disables raw frame-dump (SIL/MIL cross-validation)

Cross-validation (optional, --sil-model / --mil-model): the firmware sends back
each captured camera frame (command F1) and the host evaluates the lower levels
of the fidelity ladder on THAT EXACT SAME captured frame:
  MIL  (--mil-model, .keras float on PC)   <- test_model.py equivalent
  SIL  (--sil-model, .tflite INT8 on PC)   <- test_tflite_model.py equivalent
  HIL  (TFLM INT8 on the Cortex-M7)        <- this benchmark
Evaluating all three levels on identical sensor bytes decomposes the end-to-end
gap into attributable components:
  MIL vs SIL  = INT8 QUANTIZATION loss (same hardware, same captured frame)
  SIL vs HIL  = MCU vs PC EXECUTION divergence (TFLM vs TFLite kernels)
  PIL vs HIL  = physical FRONTEND degradation (display / optics / sensor)

Typical usage (full fidelity ladder on live camera frames, manifest hil_subset):
  python src/hil_camera_benchmark.py --port COM9 \
      --settle 1.0 --lux 10 --distance-cm 24 --ambient-temp 24 \
      --sil-model models/tflite/{model_name}_int8.tflite \
      --mil-model models/checkpoints/{model_name}.keras

Requirements: pyserial, opencv-python, numpy (+ scikit-learn, seaborn,
matplotlib for the confusion matrix and JSON metrics).
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
# Model identity verification
#
# At boot, BEFORE emitting READY_HIL, the firmware prints:
#     MODEL_BYTES:<g_model_len>      size of the g_model[] array in flash
#     MODEL_FNV1A:<hex32>            32-bit FNV-1a hash over those bytes
#     ARENA_USED:<bytes>             interpreter->arena_used_bytes()
#
# g_model[] contains the exact .tflite bytes (tflite_to_c.py), so the 32-bit
# FNV-1a computed on the host over the .tflite file must match the on-chip
# hash EXACTLY. If they differ, the board is NOT running the model the operator
# expects, and the benchmark aborts before wasting a physical rig session.
#
# FNV-1a is used instead of SHA-256 because the firmware has no mbedtls
# dependency and FNV-1a takes four lines of C; 32 bits easily distinguish all
# candidate models.
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
    """Extracts MODEL_BYTES / MODEL_FNV1A / ARENA_USED from firmware boot lines.
    Returns a dict with None for any field omitted by legacy firmware."""
    fw = {'model_bytes': None, 'model_fnv1a': None, 'arena_used_bytes': None}
    for l in lines:
        if l.startswith("MODEL_BYTES:"):
            fw['model_bytes'] = int(l.split(":", 1)[1].strip())
        elif l.startswith("MODEL_FNV1A:"):
            fw['model_fnv1a'] = l.split(":", 1)[1].strip().lower().replace("0x", "")
        elif l.startswith("ARENA_USED:"):
            fw['arena_used_bytes'] = int(l.split(":", 1)[1].strip())
    return fw


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _relpath_repo(path: str):
    """Path relative to the repository root using '/' separators, so HIL JSON
    artifacts never leak local machine paths (C:\\Users\\...) and stay portable
    across clones, matching MIL, SIL, and PIL."""
    if not path:
        return None
    try:
        rel = os.path.relpath(os.path.abspath(path), REPO_ROOT)
        if not rel.startswith(".."):
            return rel.replace("\\", "/")
    except ValueError:
        pass
    return os.path.normpath(path).replace("\\", "/")


def expected_model_fingerprint(tflite_path: str):
    """Fingerprint of the .tflite file the operator claims to have flashed."""
    if not tflite_path or not os.path.isfile(tflite_path):
        return None
    return {
        'path': _relpath_repo(tflite_path),
        'bytes': os.path.getsize(tflite_path),
        'fnv1a': f"{fnv1a32_file(tflite_path):08x}",
        'sha256': sha256_file(tflite_path),
    }


def previous_runs_fingerprints(output_dir: str):
    """Reads existing hil_conditions_*.json files: returns (model_tag, fnv1a, filename)."""
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
    Runs three checks in order:
      1. whether the firmware reported a model fingerprint (warns if legacy);
      2. whether the on-chip fingerprint matches the given .tflite (ABORTS on mismatch);
      3. whether any PREVIOUS run in output_dir recorded the exact same on-chip
         fingerprint under a DIFFERENT model tag (ABORTS on collision — prevents
         mislabeling an already-flashed binary under a new architecture name).
    Returns True if model identity was positively verified.
    """
    ok = False
    if fw['model_fnv1a'] is None:
        print("[WARN] Firmware did NOT report MODEL_FNV1A: legacy firmware without "
              "fingerprinting. This run's label is NOT verified against the "
              "board. Reflash the updated hil_camera_firmware.ino sketch.")
    elif expected is None:
        print("[WARN] No --tflite provided: cannot compare on-chip fingerprint "
              f"({fw['model_fnv1a']}, {fw['model_bytes']} B) against a reference file. "
              "Pass --tflite <path> to verify model identity.")
    else:
        same = (fw['model_fnv1a'] == expected['fnv1a']
                and fw['model_bytes'] == expected['bytes'])
        print(f"[INFO] Chip fingerprint   : fnv1a={fw['model_fnv1a']} bytes={fw['model_bytes']}")
        print(f"[INFO] .tflite fingerprint: fnv1a={expected['fnv1a']} bytes={expected['bytes']}  "
              f"({os.path.basename(expected['path'])})")
        if same:
            print("[OK]   IDENTITY VERIFIED: the board is running the exact "
                  "specified .tflite binary.")
            ok = True
        else:
            print("[ERROR] MODEL MISMATCH ON BOARD. The on-chip fingerprint does "
                  "not match the specified .tflite file. Regenerate model.h with "
                  "tflite_to_c.py, recompile, reflash, and try again.")
            if not allow_mismatch:
                sys.exit(2)
            print("[WARN] --allow-model-mismatch set: proceeding at your own risk.")

    # 3) Check for fingerprint collision with previous runs under a different tag
    if fw['model_fnv1a'] is not None:
        for prev_tag, prev_fnv, fname in previous_runs_fingerprints(output_dir):
            if prev_fnv == fw['model_fnv1a'] and prev_tag and prev_tag != model_tag:
                print(f"[ERROR] Fingerprint {fw['model_fnv1a']} already exists in {fname} "
                      f"under tag '{prev_tag}'. You are about to benchmark THE "
                      f"EXACT SAME binary under tag '{model_tag}'. Reflash the board.")
                if not allow_mismatch:
                    sys.exit(2)
    return ok


def check_required_conditions(args):
    """Physical rig conditions are mandatory for a reportable HIL run."""
    missing = [n for n, v in (("--lux", args.lux),
                              ("--distance-cm", args.distance_cm),
                              ("--ambient-temp", args.ambient_temp)) if v is None]
    if missing and not args.allow_missing_conditions:
        print("[ERROR] Missing physical rig conditions: " + ", ".join(missing) + ".")
        print("        Two runs of the same model differed by 0.074 F1-macro "
              "when environmental variables went unrecorded: without them, an "
              "HIL run is not reportable. Provide them, or pass "
              "--allow-missing-conditions for a quick smoke test.")
        sys.exit(2)
    if missing:
        print("[WARN] Incomplete conditions (" + ", ".join(missing) +
              "): run will be flagged as non-reportable in hil_conditions.")
    if not args.notes.strip():
        print("[WARN] --notes is empty: record at least monitor model, brightness, and mount.")


def warn_condition_drift(args, output_dir: str):
    """Warns before opening the stimulus rig if the CLI physical conditions
    differ from previous runs in output_dir (or in results/hil/ when output_dir
    is a replicate subfolder), catching typos such as 23 vs 24 cm early."""
    search_dirs = []
    for d in (output_dir, os.path.join(REPO_ROOT, "results", "hil")):
        if d and os.path.isdir(d) and os.path.abspath(d) not in [os.path.abspath(x) for x in search_dirs]:
            search_dirs.append(d)

    prev_records = []
    for d in search_dirs:
        for name in sorted(os.listdir(d)):
            if name.startswith("hil_conditions_") and name.endswith(".json"):
                try:
                    with open(os.path.join(d, name), encoding="utf-8") as fh:
                        c = json.load(fh)
                    prev_records.append((name, c))
                except Exception:
                    continue

    if not prev_records:
        return

    checks = [
        ("--lux", "illuminance_lux", args.lux, "lux"),
        ("--distance-cm", "camera_to_screen_distance_cm", args.distance_cm, "cm"),
        ("--ambient-temp", "ambient_temp_c", args.ambient_temp, "°C"),
    ]
    for flag, key, cur_val, unit in checks:
        if cur_val is None:
            continue
        prev_vals = sorted({float(c[key]) for _, c in prev_records if c.get(key) is not None})
        if prev_vals and float(cur_val) not in prev_vals:
            prev_str = ", ".join(f"{v:g} {unit}" for v in prev_vals)
            print(f"[WARN] Condition drift on {flag}: you passed {cur_val:g} {unit}, "
                  f"whereas previous runs recorded [{prev_str}]. Double-check "
                  f"that this is not a typo before starting the sweep.")


try:
    import serial
except ImportError:
    print("ERROR: pyserial is not installed. Run: pip install pyserial")
    sys.exit(1)

try:
    import numpy as np
    import cv2
except ImportError:
    print("ERROR: OpenCV and/or numpy are not installed.")
    print("Run: pip install opencv-python numpy")
    sys.exit(1)


# ---- Serial protocol commands (see hil_camera_firmware.ino) ----
CMD_TRIGGER   = b'T'
CMD_DUMP_ON   = b'F1'
CMD_DUMP_OFF  = b'F0'

# #...@ packet framing and byte escaping (identical to pil_benchmark.py / firmware)
MARKER_START = 0x23  # '#'
MARKER_END   = 0x40  # '@'
ESCAPE_BYTE  = 0x1B  # ESC

# Telemetry prefixes emitted by the firmware
TELEMETRY_RE = re.compile(r'^([A-Z][A-Z0-9_]*):(-?[\d.]+)$')
# Telemetry fields exported to the CSV (in this exact column order)
TELEMETRY_FIELDS = [
    'TS_MS', 'TEMP_C',
    'CYC_CAPTURE', 'CYC_PRE', 'CYC_INF', 'CYC_POST', 'CYC_TOTAL',
    'US_CAPTURE', 'US_PRE', 'US_INF', 'US_POST', 'US_TOTAL',
]
# The firmware prints its end-of-inference banner in Spanish
# ("--- Listo. Esperando siguiente trigger... ---"), so END_MARKERS must keep
# the exact Spanish substrings emitted by hil_camera_firmware.ino.
END_MARKERS = ("siguiente trigger", "siguiente imagen", "Listo")

WINDOW_NAME = "PHLAME HIL Stimulus"


def normalize_port(port: str) -> str:
    """
    Normalizes the serial port name according to the host platform (identical
    to pil_benchmark.py):
      - Native Windows : COM9+ -> \\\\.\\COMx
      - WSL (Linux)    : COMx  -> /dev/ttySx
      - Native Linux   : used as is (/dev/ttyUSB0, etc.)
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
            print(f"[INFO] Port normalized for WSL: {port}")
        elif sys.platform == 'win32' and n >= 9:
            port = f'\\\\.\\COM{n}'
            print(f"[INFO] Port normalized for Windows: {port}")
    return port


def collect_from_manifest(manifest_path: str, use_all_test=False, seed=42):
    """
    Enumerates (image_path, int_label) pairs from the test split manifest
    generated by src/split_dataset.py.

    By default, ONLY rows with hil_subset == 1 (the stratified subset flagged
    for the physical rig) are selected. With use_all_test=True, every row in
    the test manifest is used.

    Why read from the manifest instead of scanning a folder:
      - The manifest is the ONLY place where the hil_subset flag lives. Without
        that filter, a lab session grows from ~200 stimuli to the entire test set.
      - Class ordering comes directly from the manifest (integer label ->
        class_name sorted by label), guaranteeing alignment with the firmware's
        argmax index regardless of filesystem traversal order.
      - Provides an auditable, git-versioned record of which exact stimuli were
        evaluated.

    Returns (image_data, class_names).
    """
    if not os.path.isfile(manifest_path):
        print(f"[ERROR] Manifest not found: {manifest_path}")
        print("        Generate the dataset split first:")
        print("          python src/split_dataset.py")
        print("        Or point to an existing manifest with --manifest PATH.")
        sys.exit(1)

    with open(manifest_path, newline='', encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print(f"[ERROR] Manifest is empty: {manifest_path}")
        sys.exit(1)

    requeridas = {'class_name', 'label', 'src_path', 'dst_path'}
    faltan = requeridas - set(rows[0].keys())
    if faltan:
        print(f"[ERROR] Manifest is missing required columns {sorted(faltan)}.")
        print(f"        Columns found: {sorted(rows[0].keys())}")
        sys.exit(1)

    # Canonical class order from manifest: integer label -> class_name
    label_a_nombre = {}
    for r in rows:
        label_a_nombre[int(r['label'])] = r['class_name']
    class_names = [label_a_nombre[k] for k in sorted(label_a_nombre)]
    print(f"[INFO] Classes loaded from manifest (order = firmware argmax index): "
          f"{class_names}")

    # Filter rows flagged for the physical HIL rig
    if use_all_test:
        sel = rows
        print(f"[INFO] --all-test: using all {len(sel)} test images.")
    else:
        if 'hil_subset' not in rows[0]:
            print("[ERROR] Manifest has no 'hil_subset' column.")
            print("        Regenerate the split or pass --all-test.")
            sys.exit(1)
        sel = [r for r in rows
               if str(r['hil_subset']).strip() in ('1', 'True', 'true')]
        if not sel:
            print("[ERROR] No manifest rows have hil_subset == 1.")
            print("        Regenerate the split or pass --all-test.")
            sys.exit(1)
        print(f"[INFO] HIL subset: {len(sel)} of {len(rows)} test images "
              f"(hil_subset == 1).")

    # Resolve file paths: prefer dst_path (materialized split), fall back to src_path
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
        print(f"[ERROR] {len(ausentes)} manifest images do not exist on disk. "
              f"First 5 missing paths:")
        for p in ausentes[:5]:
            print(f"          {p}")
        print("        Materialize the dataset split with:")
        print("          python src/split_dataset.py")
        sys.exit(1)

    reparto = Counter(class_names[lbl] for _, lbl in image_data)
    print(f"[INFO] Per-class stimulus breakdown: {dict(reparto)}")

    # ALWAYS shuffle with a fixed seed. Manifest rows are grouped by class:
    # presenting them in block order would evaluate one class while the MCU is
    # cold and the other after thermal warm-up, confounding thermal/lighting
    # drift with class effects.
    random.seed(seed)
    random.shuffle(image_data)
    print(f"[INFO] Stimulus presentation order shuffled (seed {seed}).")
    return image_data, class_names


def collect_dataset(folder: str, n_samples=None, seed=None):
    """
    Enumerates (image_path, int_label) pairs from per-class subfolders,
    matching pil_benchmark.py. Returns (image_data, class_names).
    """
    extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    subdirs = sorted(d for d in os.listdir(folder)
                     if os.path.isdir(os.path.join(folder, d)))
    if not subdirs:
        print(f"[ERROR] Per-class subfolders are required in {folder} "
              f"(cannot build a valid HIL confusion matrix without ground truth).")
        sys.exit(1)

    class_names = subdirs
    print(f"[INFO] Ground-truth classes detected: {class_names}")

    image_data = []
    for label_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(folder, class_name)
        for f in os.listdir(class_dir):
            if f.lower().endswith(extensions):
                image_data.append((os.path.join(class_dir, f), label_idx))

    image_data = sorted(set(image_data))
    if not image_data:
        print(f"[ERROR] No valid images found in: {folder}")
        sys.exit(1)

    available = len(image_data)
    if seed is not None:
        random.seed(seed)
    if n_samples is not None and n_samples < available:
        image_data = random.sample(image_data, n_samples)
        print(f"[INFO] Selected {n_samples} images at random "
              f"out of {available} available")
    else:
        print(f"[INFO] {available} images found (using all)")

    # Shuffle presentation order to avoid class blocks that would confound
    # temporal drift (temperature, ambient light) with class effects.
    random.shuffle(image_data)
    return image_data, class_names


# --------------------------------------------------------------------------
# Serial communication
# --------------------------------------------------------------------------

def read_lines_until(ser, predicate, timeout_s=30.0, echo=True):
    """
    Reads lines from the serial port until predicate(line) returns True or
    timeout_s expires. Returns (all_lines_read, matching_line | None).
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
    Waits for the boot handshake from the dedicated HIL firmware
    (hil_camera_firmware.ino): CAM_INIT:OK ... READY_HIL.
    Aborts if camera initialization fails or times out (e.g. PIL firmware flashed).
    """
    lines, hit = read_lines_until(
        ser, lambda l: l == "READY_HIL" or l == "CAM_INIT:FAIL",
        timeout_s=timeout_s)
    if hit == "READY_HIL":
        print("[OK]   HIL firmware ready (HM01B0 camera initialized).")
        return parse_firmware_fingerprint(lines)
    if hit == "CAM_INIT:FAIL":
        print("[ERROR] HM01B0 camera failed to initialize (CAM_INIT:FAIL). "
              "Is the Portenta Vision Shield properly attached?")
    else:
        print("[ERROR] Timed out waiting for READY_HIL. Did you flash "
              "deployment/hil_camera_firmware/hil_camera_firmware.ino? "
              "(the PIL firmware does not respond to this benchmark). If the "
              "HIL firmware was already running, press RESET on the board and retry.")
    ser.close()
    sys.exit(1)


def parse_inference(lines):
    """
    Extracts (predicted_class, telemetry_dict) from one inference response.
    The predicted class is the standalone integer on its own line (compatible
    with pil_benchmark.py); telemetry lines follow the 'PREFIX:value' format.
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
    Sends an HIL capture trigger ('T') and parses the complete response.
    Returns (predicted_class | None, telemetry_dict).
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
    Reads the raw captured frame dump sent by the firmware after an HIL
    inference (FRAME_BEGIN:<n> + escaped #...@ packet + FRAME_END).
    Returns np.ndarray of shape (120, 160) uint8, or None on error/timeout.
    Called AFTER the end-of-inference marker of trigger_inference, outside the
    timed region on the MCU, so frame dumping never perturbs latency metrics.
    """
    # 1) Wait for FRAME_BEGIN:<n>
    lines, hit = read_lines_until(
        ser, lambda l: l.startswith("FRAME_BEGIN:"),
        timeout_s=timeout_s, echo=False)
    if hit is None:
        print("  [WARN] FRAME_BEGIN not received (is frame-dump enabled?).")
        return None
    n_expected = int(hit.split(':', 1)[1])

    # 2) Read raw bytes: '#' ... escaped payload ... '@'
    #    Byte-by-byte state machine; escape_pending survives across read()
    #    chunk boundaries when ESC lands at the end of a USB packet.
    payload = bytearray()
    in_packet = False
    escape_pending = False
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        chunk = ser.read(max(1, ser.in_waiting or 1))
        if not chunk:
            continue
        for idx_b, b in enumerate(chunk):
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
                # 3) Consume the trailing FRAME_END line unless it was already
                #    coalesced into the remainder of this USB read() chunk.
                if b"FRAME_END" not in chunk[idx_b + 1:]:
                    read_lines_until(ser, lambda l: l == "FRAME_END",
                                     timeout_s=3, echo=False)
                if len(payload) != n_expected:
                    print(f"  [WARN] Incomplete frame: {len(payload)}/"
                          f"{n_expected} bytes.")
                    return None
                arr = np.frombuffer(bytes(payload), dtype=np.uint8)
                return arr.reshape(120, 160)  # HM01B0 grayscale frame: 160x120
            else:
                payload.append(b)
    print("  [WARN] Timed out while reading captured frame.")
    return None


# --------------------------------------------------------------------------
# Cross-validation: PC-side SIL & MIL on the REAL HM01B0 captured frame
# --------------------------------------------------------------------------

class SilModel:
    """
    Runs the EXACT SAME INT8 .tflite model on the host PC over the captured
    frame returned by the firmware (frame-dump). Replicates the firmware's
    preprocessing pipeline (cameraFrameToImageBuffer + loadImageToInputTensor):
      - NEAREST resize from 160x120 to the model's input geometry
      - single-channel grayscale replicated to 3 channels if the model is RGB
      - raw pixel [0, 255] quantized using the input tensor's scale/zero_point
    Consequently, any disagreement between HIL (MCU) and SIL (PC) on the same
    frame is strictly attributable to EXECUTION (TFLM vs TFLite kernels and
    rounding), whereas any drop from clean-image SIL/PIL to captured-frame SIL
    is attributable to the physical FRONTEND (display, optics, sensor, lighting).
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
                print("ERROR: tensorflow or tflite-runtime is required for "
                      "--sil-model. Run: pip install tensorflow")
                sys.exit(1)
        self._interp.allocate_tensors()
        self._in = self._interp.get_input_details()[0]
        self._out = self._interp.get_output_details()[0]
        shape = self._in['shape']  # [1, H, W, C]
        self.h, self.w, self.c = int(shape[1]), int(shape[2]), int(shape[3])
        print(f"[INFO] PC-side SIL ready: {model_path} "
              f"(input {self.h}x{self.w}x{self.c})")

    def predict(self, frame_gray):
        """frame_gray: np.ndarray (120, 160) uint8 -> predicted class (int)."""
        # NEAREST resize (identical to the firmware's nearest-neighbor resize)
        img = cv2.resize(frame_gray, (self.w, self.h),
                         interpolation=cv2.INTER_NEAREST)
        if self.c == 3:
            img = np.stack([img] * 3, axis=-1)
        else:
            img = img[..., np.newaxis]
        x = img[np.newaxis].astype(np.float32)  # raw pixel [0, 255]

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
    MIL level of the ladder: runs the floating-point .keras model on the host
    PC over the real frame captured by the HM01B0 sensor (matching test_model.py
    in numerical representation, but fed with live sensor bytes instead of clean
    dataset files). Shares the exact preprocessing contract of the firmware and
    SilModel: NEAREST resize, grayscale channel replication if RGB, raw [0, 255]
    pixels (rescaling layers live inside the Keras graph where applicable).
    """

    def __init__(self, model_path):
        try:
            import tensorflow as tf
        except ImportError:
            print("ERROR: tensorflow is required for --mil-model. "
                  "Run: pip install tensorflow")
            sys.exit(1)
        self._model = tf.keras.models.load_model(model_path, compile=False)
        shape = self._model.input_shape  # (None, H, W, C)
        self.h, self.w, self.c = int(shape[1]), int(shape[2]), int(shape[3])
        print(f"[INFO] PC-side MIL ready: {model_path} "
              f"(input {self.h}x{self.w}x{self.c}, float32)")

    def predict(self, frame_gray):
        """frame_gray: np.ndarray (120, 160) uint8 -> predicted class (int)."""
        img = cv2.resize(frame_gray, (self.w, self.h),
                         interpolation=cv2.INTER_NEAREST)
        if self.c == 3:
            img = np.stack([img] * 3, axis=-1)
        else:
            img = img[..., np.newaxis]
        x = img[np.newaxis].astype(np.float32)  # raw pixel [0, 255]
        y = self._model.predict(x, verbose=0)
        return int(np.argmax(y))


# --------------------------------------------------------------------------
# Stimulus rig: host-controlled monitor display
# --------------------------------------------------------------------------

def open_stimulus_window(fullscreen=True):
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    if fullscreen:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)


def show_stimulus(image_path, settle_s):
    """
    Displays the stimulus centered on a black canvas and waits settle_s
    (monitor refresh + HM01B0 auto-exposure settling time).
    Uses NEAREST interpolation so the display never synthesizes smoothed pixels:
    optical filtering comes solely from the physical camera path.
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
    # waitKey pumps the GUI event loop, which is required for the frame to
    # actually render on screen before sending the capture trigger.
    cv2.waitKey(max(1, int(settle_s * 1000)))
    return True


# --------------------------------------------------------------------------
# Reporting and output artifacts
# --------------------------------------------------------------------------

def _json_safe(o):
    """Converts numpy scalars/arrays to native Python types for json.dump."""
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
    raise TypeError(f"Cannot serialize to JSON: {type(o).__name__}")


def _latency_summary(rows):
    """Per-field telemetry summary matching the pil_benchmark.py JSON schema."""
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
    Writes all HIL benchmark artifacts tagged with model_tag so consecutive
    runs across different architectures never overwrite each other:
      - hil_latencies_<tag>.csv
      - hil_conditions_<tag>.json
      - HIL_Confusion_Matrix_<tag>.png
      - metrics_<tag>.json (unified MIL/SIL/PIL/HIL schema)
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Per-inference sensor->prediction latency CSV ---
    csv_path = os.path.join(output_dir, f"hil_latencies_{model_tag}.csv")
    fieldnames = (['idx', 'utc_iso', 'image', 'true_label', 'true_class',
                   'pred_label', 'pred_class', 'sil_label', 'sil_class',
                   'mil_label', 'mil_class', 'frame_file']
                  + [f.lower() for f in TELEMETRY_FIELDS])
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK]   Per-phase latencies saved to: {csv_path}")

    # --- Physical rig conditions (environmental protocol) ---
    cond_path = os.path.join(output_dir, f"hil_conditions_{model_tag}.json")
    with open(cond_path, 'w', encoding='utf-8') as f:
        # Output schema documented in docs/results_schema.md - update it if you add, rename or remove keys.
        json.dump(conditions, f, indent=2, ensure_ascii=False)
    print(f"[OK]   Rig conditions saved to: {cond_path}")

    # --- Classification metrics + confusion matrix ---
    y_true = [r['true_label'] for r in rows if r['pred_label'] is not None]
    y_pred = [r['pred_label'] for r in rows if r['pred_label'] is not None]
    if not y_true:
        print("[WARN] No valid predictions received: skipping confusion matrix.")
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
        print("      METRICS REPORT — HIL (Camera-in-the-Loop)")
        print("=" * 50)
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0)
        print(f"Accuracy:              {accuracy:.4f}")
        print(f"Precision:             {precision:.4f}")
        print(f"Recall:                {recall:.4f}")
        print(f"F1-Score (weighted):   {f1:.4f}")

        labels_present = sorted(set(y_true + y_pred))
        target_names = [class_names[l] if 0 <= l < len(class_names)
                        else f"Invalida_{l}" for l in labels_present]

        print("\nDetailed Classification Report:")
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
        print(f"[OK]   HIL confusion matrix saved to: {cm_path}")

        # --- metrics_<tag>.json: same schema as MIL / SIL / PIL ---
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
                "matriz_png": _relpath_repo(cm_path),
                "latencias_csv": _relpath_repo(csv_path),
                "condiciones_json": _relpath_repo(cond_path),
            },
        }
        metrics_path = os.path.join(output_dir, f"metrics_{model_tag}.json")
        with open(metrics_path, "w", encoding="utf-8") as fh:
            # Output schema documented in docs/results_schema.md - update it if you add, rename or remove keys.
            json.dump(resultados, fh, indent=2, ensure_ascii=False, default=_json_safe)
        print(f"[OK]   JSON metrics (same schema as MIL/SIL/PIL) saved to: {metrics_path}")
        print(f"       F1-macro = {f_m:.4f}   errors = {int(cm.sum() - np.trace(cm))}")
    except ImportError:
        print("\n[WARN] Missing libraries required for the confusion matrix and JSON metrics.")
        print("Install them with: pip install scikit-learn seaborn matplotlib")

    # --- Per-phase latency summary table ---
    def stats(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        if not vals:
            return None
        arr = np.array(vals, dtype=float) / 1000.0  # us -> ms
        return arr.mean(), arr.std(), arr.min(), arr.max()

    print("\n" + "=" * 60)
    print("  LATENCY BY PHASE (sensor -> prediction)   [ms]")
    print(f"  {'Phase':<10} {'mean':>9} {'std':>9} {'min':>9} {'max':>9}")
    print("  " + "-" * 50)
    for key, name in [('us_capture', 'CAPTURE'), ('us_pre', 'PRE'),
                      ('us_inf', 'INF'), ('us_post', 'POST'),
                      ('us_total', 'TOTAL')]:
        s = stats(key)
        if s:
            print(f"  {name:<10} {s[0]:>9.3f} {s[1]:>9.3f} "
                  f"{s[2]:>9.3f} {s[3]:>9.3f}")
    print("=" * 60)

    # --- Cross-validation: MIL/SIL/HIL fidelity ladder on the SAME frame ---
    # Only frames where ALL enabled levels produced a prediction are included,
    # keeping accuracies strictly paired across levels.
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
            print("  FIDELITY LADDER — same frame captured by the HM01B0 sensor")
            print(f"  Paired frames: {n}")
            print(f"\n  {'Level':<30} {'Accuracy':>10}")
            print("  " + "-" * 42)
            for key, label in levels:
                acc = sum(1 for r in paired
                          if r[key] == r['true_label']) / n
                print(f"  {label:<30} {acc:>10.4f}")

            print(f"\n  {'Pairwise agreement':<38} {'%':>7}   attributable to")
            print("  " + "-" * 60)
            pair_meaning = {
                ('mil_label', 'sil_label'): 'INT8 quantization',
                ('sil_label', 'pred_label'): 'MCU vs PC execution',
                ('mil_label', 'pred_label'): 'quantization + execution',
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
            print("\n  -> Any drop across ALL levels vs their clean-dataset / PIL")
            print("     accuracies reflects physical FRONTEND degradation")
            print("     (display / optics / sensor / lighting): the PIL -> HIL gap.")
            print("=" * 64)


# --------------------------------------------------------------------------
# Main benchmark loop
# --------------------------------------------------------------------------

def run_benchmark(args):
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(errors="replace")
        except Exception:
            pass
    port = normalize_port(args.port)

    # --- Require physical rig conditions before opening serial or GUI ---
    check_required_conditions(args)
    # --- Warn early if CLI conditions drift from previous runs ---
    warn_condition_drift(args, args.output_dir)

    # --- Resolve model tag and expected .tflite reference ---
    tflite_path = args.tflite or args.sil_model
    if args.model_tag:
        model_tag = args.model_tag
    elif tflite_path:
        model_tag = os.path.splitext(os.path.basename(tflite_path))[0]
    else:
        print("[ERROR] Cannot determine which model is on the board to name output "
              "artifacts: pass --tflite <path to flashed .tflite> (recommended; "
              "also verifies the on-chip FNV-1a hash) or at least --model-tag <name>.")
        sys.exit(2)
    expected = expected_model_fingerprint(tflite_path)
    if tflite_path and expected is None:
        print(f"[ERROR] Specified .tflite file does not exist: {tflite_path}")
        sys.exit(2)
    if args.folder:
        print(f"[WARN] Explicit --folder provided: reading from {args.folder} "
              f"and bypassing the manifest hil_subset filter.")
        fuente = f"folder:{_relpath_repo(args.folder)}"
        image_data, class_names = collect_dataset(args.folder, args.count,
                                                  args.seed)
    else:
        fuente = f"manifest:{_relpath_repo(args.manifest)}"
        image_data, class_names = collect_from_manifest(
            args.manifest, use_all_test=args.all_test, seed=args.seed)
        if args.count is not None and args.count < len(image_data):
            random.seed(args.seed)
            image_data = random.sample(image_data, args.count)
            print(f"[INFO] --count {args.count}: subsampled to {args.count} "
                  f"stimuli.")

    print(f"[INFO] Connecting to {port} @ {args.baud} baud ...")
    try:
        ser = serial.Serial(port, args.baud, timeout=10)
    except serial.SerialException as e:
        print(f"ERROR opening serial port: {e}")
        sys.exit(1)

    print(f"[INFO] Port open. Waiting {args.delay}s (firmware setup)...")
    time.sleep(args.delay)

    # Handshake with the dedicated HIL firmware (hil_camera_firmware.ino)
    fw = wait_ready(ser)
    identity_ok = verify_model_identity(fw, expected, model_tag, args.output_dir,
                                        allow_mismatch=args.allow_model_mismatch)

    # Cross-validation: raw frame-dump + PC-side MIL/SIL
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
        # Store captured frames under data/ (git-ignored; for local traceability)
        frames_dir = args.frames_dir
        os.makedirs(frames_dir, exist_ok=True)
        print(f"[INFO] Frame-dump enabled; saving captured frames to: {frames_dir}")

    # Stimulus rig: host-controlled monitor window
    open_stimulus_window(fullscreen=not args.windowed)
    print("[INFO] Stimulus window open. Align the camera with the screen "
          "(fixed distance and framing) and verify optical focus.")
    if not args.no_confirm:
        input("       Press ENTER to start the stimulus sweep... ")

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
        'manifest': (_relpath_repo(args.manifest)
                     if not args.folder else None),
        'manifest_subset': ('all-test' if args.all_test else 'hil_subset==1')
                           if not args.folder else None,
        'dataset_folder': (_relpath_repo(args.folder)
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
        'sil_model': (_relpath_repo(args.sil_model)
                      if args.sil_model else None),
        'mil_model': (_relpath_repo(args.mil_model)
                      if args.mil_model else None),
        'frames_dir': (_relpath_repo(frames_dir) if frames_dir else None),
        'notes': args.notes,
    }

    rows = []
    counts = Counter()
    errors = 0
    total = len(image_data)

    try:
        for idx, (image_path, true_label) in enumerate(image_data, start=1):
            name = os.path.basename(image_path)
            print(f"\n[INFO] ({idx}/{total}) Stimulus: {name} "
                  f"(ground-truth class: {class_names[true_label]})")

            if not show_stimulus(image_path, args.settle):
                print(f"  [WARN] Could not load {image_path}; skipping.")
                errors += 1
                continue

            predicted, telemetry = trigger_inference(ser)

            # Frame-dump + PC-side MIL/SIL evaluation on the REAL captured frame
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
                               and 0 <= predicted < len(class_names)
                               else None),
                'sil_label': sil_label,
                'sil_class': (class_names[sil_label]
                              if sil_label is not None
                              and 0 <= sil_label < len(class_names)
                              else None),
                'mil_label': mil_label,
                'mil_class': (class_names[mil_label]
                              if mil_label is not None
                              and 0 <= mil_label < len(class_names)
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
                lat = (f" | sensor->pred: {us_total / 1000.0:.2f} ms"
                       if us_total else "")
                xval = ""
                if sil_label is not None:
                    agree = "=" if sil_label == predicted else "≠"
                    xval += f" | SIL(PC): {row['sil_class']} {agree} HIL"
                if mil_label is not None:
                    agree = "=" if mil_label == predicted else "≠"
                    xval += f" | MIL(PC): {row['mil_class']} {agree} HIL"
                print(f"  {marker} Predicted: {row['pred_class']}{lat}{xval}")
            else:
                errors += 1
                print("  [WARN] No response from firmware for this stimulus.")

            if idx < total and args.gap > 0:
                time.sleep(args.gap)
    except KeyboardInterrupt:
        print("\n[WARN] Interrupted by user; saving all data collected so far.")
    finally:
        conditions['utc_end'] = datetime.now(timezone.utc).isoformat()
        # Disable frame-dump and close cleanly
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
    print("  SUMMARY — HIL BENCHMARK (Camera-in-the-Loop)")
    print(f"  Stimuli: {total}  |  Answered: {respondidas}  "
          f"|  Failures: {errors}")
    print("=" * 45)

    write_outputs(rows, class_names, conditions, args.output_dir, model_tag)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real Camera-in-the-Loop HIL benchmark: the HM01B0 captures "
                    "stimuli displayed on a monitor, pairs predictions against "
                    "ground truth, and measures per-phase sensor->prediction latency.")
    parser.add_argument('--port', default="COM9",
                        help="Serial port (e.g. COM9 or /dev/ttyUSB0)")
    parser.add_argument('--baud', type=int, default=115200,
                        help="Baud rate (default: 115200)")
    parser.add_argument('--manifest',
                        default=os.path.join('data', 'splits',
                                             'test_manifest.csv'),
                        help="Test split manifest CSV "
                             "(default: data/splits/test_manifest.csv). "
                             "Default stimulus source.")
    parser.add_argument('--all-test', action='store_true',
                        help="Use every image in the test manifest instead of "
                             "only rows flagged with hil_subset == 1 (~200). "
                             "Increases lab session duration proportionally.")
    parser.add_argument('--folder', default=None,
                        help="Explicit OVERRIDE: dataset folder with per-class "
                             "subfolders. Ignores the manifest and bypasses "
                             "the hil_subset filter.")
    parser.add_argument('--count', type=int, default=None, metavar='N',
                        help="Number of random stimuli to evaluate (default: all)")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for reproducible sampling and ordering "
                             "(default: 42)")
    parser.add_argument('--settle', type=float, default=1.5,
                        help="Settling time in seconds after displaying each "
                             "stimulus before triggering capture (default: 1.5)")
    parser.add_argument('--gap', type=float, default=0.5,
                        help="Seconds to wait between consecutive stimuli (default: 0.5)")
    parser.add_argument('--delay', type=float, default=2.0,
                        help="Initial wait in seconds after opening the serial port (default: 2.0)")
    parser.add_argument('--output-dir', default=os.path.join('results', 'hil'),
                        help="Output directory (default: results/hil)")
    parser.add_argument('--windowed', action='store_true',
                        help="Open a normal resizable window instead of fullscreen")
    parser.add_argument('--no-confirm', action='store_true',
                        help="Start the stimulus sweep immediately without waiting for ENTER")
    # Environmental rig conditions
    parser.add_argument('--lux', type=float, default=None,
                        help="Illuminance measured at the screen plane [lux]")
    parser.add_argument('--distance-cm', type=float, default=None,
                        help="Camera-to-screen distance [cm]")
    parser.add_argument('--ambient-temp', type=float, default=None,
                        help="Ambient room temperature [°C]")
    # Model identity verification and output tagging
    parser.add_argument('--tflite', default=None, metavar='TFLITE',
                        help="Path to the INT8 .tflite file used to generate "
                             "the flashed model.h. Its FNV-1a hash is verified "
                             "against the hash printed by the firmware at boot, "
                             "aborting on mismatch. Falls back to --sil-model "
                             "when omitted.")
    parser.add_argument('--model-tag', default=None, metavar='NAME',
                        help="Tag used to name output files (default: .tflite "
                             "basename without extension).")
    parser.add_argument('--allow-model-mismatch', action='store_true',
                        help="Do NOT abort on on-chip FNV-1a mismatch or "
                             "previous-run tag collision (debugging only; "
                             "recorded in output JSON).")
    parser.add_argument('--allow-missing-conditions', action='store_true',
                        help="Do NOT abort when --lux/--distance-cm/"
                             "--ambient-temp are omitted (smoke tests only).")
    parser.add_argument('--notes', default="",
                        help="Free-form setup notes (framing, mount, monitor "
                             "model, brightness setting, etc.)")
    # Cross-validation: HIL (MCU) vs SIL/MIL (PC) on the exact same frame
    parser.add_argument('--dump-frames', action='store_true',
                        help="Retrieve and save every raw frame captured by "
                             "the HM01B0 camera (traceability)")
    parser.add_argument('--sil-model', default=None, metavar='TFLITE',
                        help="Path to the SAME INT8 .tflite flashed on the MCU: "
                             "runs on the PC over each captured frame (implies "
                             "--dump-frames). Isolates MCU vs PC execution divergence.")
    parser.add_argument('--mil-model', default=None, metavar='KERAS',
                        help="Path to the original float .keras model: runs on "
                             "the PC over each captured frame (implies "
                             "--dump-frames). Combined with --sil-model, "
                             "isolates INT8 quantization loss.")
    parser.add_argument('--frames-dir', default=os.path.join('data',
                                                             'hil_frames'),
                        help="Directory for saved captured frames, git-ignored "
                             "(default: data/hil_frames)")
    return parser.parse_args()


if __name__ == '__main__':
    run_benchmark(parse_args())
