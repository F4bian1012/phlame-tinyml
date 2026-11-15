r"""
pil_benchmark.py
================
Sends preprocessed images to the board (Arduino Portenta H7 or Nicla Vision)
over the serial port for Processor-in-the-Loop (PIL) evaluation.

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

Before the first image the bench asks the board what it is running (boot lines,
or an empty packet #@ when the firmware was already running) and stops unless:
  - the board is the one given with --board;
  - the FNV-1a fingerprint and the size of the model in flash are those of the
    .tflite given with --model_path (--allow_model_mismatch measures anyway and
    records model.identity_verified = false; a firmware built before November
    2026 prints no fingerprint and needs that option);
  - the images it would send have the input shape the firmware announces
    (--width and --height may be left out: they are taken from the firmware).

A run that does not reach its last image (the serial link is lost for good,
the board stops answering, Ctrl+C, an error) keeps what it measured: the two
CSV files are written row by row and metrics_*.json is written with
run.completed = false, the reason in run.stopped_because and the number of
images planned in dataset.n_images_planned.

Exit codes:
  0  every planned image was sent and recorded
  1  the run could not start (port, images, the board does not answer, output
     files that would be replaced)
  2  refused before measuring: wrong board, model not verified, input shape
  3  the run was cut short; its files are marked completed = false

Options:
  --port        Serial port (default: COM9; e.g. COM3, /dev/ttyUSB0)
  --board       Board the firmware runs on: portenta (default) or nicla. Sets the
                default output folder (results/pil or results/nicla/pil) and is
                checked against the board the firmware announces
  --model_path  Path to the flashed .tflite model (required; hashed & used to tag outputs)
  --image       Path to a specific image to send
  --splits_dir  Directory with the splits produced by split_dataset.py
                (default: data/splits; the benchmark uses ONLY <splits_dir>/test)
  --folder      Explicit folder to send, instead of <splits_dir>/test
  --baud        Baud rate (default: 115200)
  --width       Width of the images sent (default: the one the firmware announces)
  --height      Height of the images sent (default: the one the firmware announces)
  --color       Send in RGB color (default: grayscale)
  --delay       Seconds to wait after opening the port before talking to the board (default: 1.0)
  --gap         Seconds to wait between consecutive images (default: 0.5)
  --count       Number of random images to send from the folder (default: all)
  --max_unanswered        Images in a row without an answer after which the run
                          is given up (default: 5; 0 never gives up)
  --allow_model_mismatch  Measure although the model on the board could not be
                          verified against --model_path
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
import artifacts
import boards
import cli

try:
    import serial
except ImportError:
    print("ERROR: pyserial is not installed. Run: pip install pyserial")
    sys.exit(1)

try:
    # images.py is the loader shared with MIL and SIL; it needs Pillow and numpy.
    import images
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


def _sha256_file(path, chunk=1024 * 1024):
    """SHA-256 of the file, read in chunks.

    Returns None if the path is not a file: a JSON without a checksum is
    preferable to losing the whole benchmark on the last line, after having
    sent every image to the board.
    """
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _latency_summary(rows, fields):
    """Per-field statistics from the CYC_*/US_*/TEMP_C lines.

    Values arrive as text over the serial port and may be missing (an image
    with no response leaves the field empty), so non-numeric ones are
    discarded instead of aborting the whole summary. A field with no valid
    measurement at all is left as None, which is information: it says the
    firmware did not report it.

    p95 by nearest rank over the sorted sample (no interpolation): with few
    images this is more honest than inventing an intermediate value.
    """
    summary = {}
    for field in fields:
        vals = []
        for row in rows:
            try:
                vals.append(float(row.get(field, "")))
            except (TypeError, ValueError):
                continue
        if not vals:
            summary[field] = None
            continue
        ordered = sorted(vals)
        idx95 = int(round(0.95 * (len(ordered) - 1)))
        summary[field] = {
            "n": len(vals),
            "mean": statistics.fmean(vals),
            "median": statistics.median(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "min": ordered[0],
            "max": ordered[-1],
            "p95": ordered[idx95],
        }
    return summary


def _class_name(idx, class_names):
    """Class name from the index returned by the board.

    The firmware returns a raw integer. If it does not correspond to any known
    class (firmware flashed with another model, or a different dataset) it is
    marked as Invalid_<n> instead of blowing up the row: the CSV must record
    what happened, including an incoherent response.
    """
    if idx is None or idx < 0:
        return ""
    if idx < len(class_names):
        return class_names[idx]
    return "Invalid_%d" % idx


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

    The array comes from images.load_image(), the loader MIL and SIL also use:
    the bytes sent to the board are the ones the PC levels evaluate.
    """
    arr = images.load_image(image_path, width, height, grayscale)  # (H, W) or (H, W, 3)

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


# How long to wait for a serial port that disappeared to come back, and how many
# times in a row the same image is sent again before the run is given up.
RECONNECT_WAIT_S = 30
MAX_ATTEMPTS_PER_IMAGE = 5

# How long the board has to answer one image (or the request for its identity).
RESPONSE_TIMEOUT_S = 30
# A board that stops answering makes every image wait for that whole time. After
# this many images in a row without an answer the run ends instead of waiting
# through the rest of the folder (default of --max_unanswered; 0 never gives up).
MAX_CONSECUTIVE_UNANSWERED = 5
# The firmware prints its boot lines about 2 s after the port is opened for the
# first time after a reset. If nothing has arrived this long after opening the
# port, the firmware was already running and has to be asked.
BOOT_LINES_S = 4.0
# The boot lines are over when the port has been quiet for this long.
QUIET_S = 1.0

# Exit codes (see the module docstring).
EXIT_REFUSED = 2
EXIT_INCOMPLETE = 3


def reopen_port(port: str, baud: int, wait_s: float = RECONNECT_WAIT_S):
    """Opens the port again after the link dropped, retrying until it is back.

    A USB serial link can disappear for a moment (cable, hub, the host's power
    management) while the board keeps running. Returns the open port, or None
    when it did not come back within wait_s seconds."""
    deadline = time.time() + wait_s
    while time.time() < deadline:
        try:
            ser = serial.Serial(port, baud, timeout=10)
            time.sleep(0.5)   # let the link settle before anything is written
            return ser
        except serial.SerialException:
            time.sleep(0.5)
    return None


class LinkLost(Exception):
    """The serial link dropped in the middle of an image and did not come back."""

    def __init__(self, message, reconnects):
        super().__init__(message)
        self.reconnects = reconnects


def read_line(ser):
    """One line of the firmware, echoed; '' when it was empty."""
    line = ser.readline().decode('utf-8', errors='replace').rstrip()
    if line:
        print(f"  Arduino >>> {line}")
    return line


def listen_for_boot_lines(ser, opened_at):
    """Reads what the firmware prints by itself after the port is opened.

    The PIL sketch prints its boot lines only once, when the port is opened for
    the first time after a reset, and they end with READY_PIL. Returns the lines
    read: all of them up to READY_PIL, or the ones that arrived before the port
    went quiet, or none when the firmware was already running.
    """
    lines = []
    last = opened_at
    give_up = opened_at + BOOT_LINES_S + RESPONSE_TIMEOUT_S   # a board that never stops printing
    while time.time() < give_up:
        if ser.in_waiting:
            line = read_line(ser)
            if line:
                lines.append(line)
                last = time.time()
                if line.strip() == boards.READY_PIL:
                    return lines
            continue
        now = time.time()
        if lines and now - last >= QUIET_S:
            return lines
        if not lines and now - opened_at >= BOOT_LINES_S:
            return lines
        time.sleep(0.02)
    return lines


def ask_identity(ser):
    """Sends an empty packet (#@) and reads the answer.

    Returns (lines, kind). kind is 'identity' when the firmware answered with
    its identity lines (they end with READY_PIL), 'inference' when it ran an
    inference instead (a build without identity lines takes the empty packet
    for an image), and 'silent' when nothing closed the answer in time.
    """
    ser.reset_input_buffer()
    ser.write(MARKER_START + MARKER_END)
    ser.flush()
    lines = []
    deadline = time.time() + RESPONSE_TIMEOUT_S
    while time.time() < deadline:
        if ser.in_waiting:
            line = read_line(ser)
            if not line:
                continue
            lines.append(line)
            if line.strip() == boards.READY_PIL:
                return lines, "identity"
            if boards.is_end_of_inference(line):
                return lines, "inference"
        else:
            time.sleep(0.02)
    return lines, "silent"


def handshake(ser, opened_at):
    """Finds out what the board is running, before anything is measured.

    Returns (identity, has_identity): identity is boards.identity() of the
    lines the firmware printed, and has_identity says whether they were its
    identity lines (False for a build that has none: its fields are then None,
    except the board when the build names it with every answer). Ends the run
    when the board does not answer at all or runs the HIL sketch.
    """
    lines = listen_for_boot_lines(ser, opened_at)
    if any(l.strip() == boards.READY_HIL for l in lines):
        print("[ERROR] The board is running the HIL firmware (READY_HIL), which does not take "
              "images over serial. Flash the PIL one: python src/compile_upload_arduino.py --target pil")
        ser.close()
        sys.exit(1)
    if any(l.strip() == boards.READY_PIL for l in lines):
        return boards.identity(lines), True

    # No identity lines yet: the firmware was already running (its boot lines
    # are gone), or it is a build that has none. Asked twice, because a board
    # left in the middle of a packet by an interrupted run can take the first
    # request for the end of that packet.
    seen = list(lines)
    for _ in range(2):
        answer, kind = ask_identity(ser)
        if kind == "identity":
            return boards.identity(answer), True
        seen += answer
        if kind == "silent":
            print(f"[ERROR] The board did not answer in {RESPONSE_TIMEOUT_S} s. Check that the PIL "
                  "firmware is flashed (python src/compile_upload_arduino.py --target pil) and that "
                  "it finished its set-up; press RESET on the board and run again."
                  + (f" Last line received: {seen[-1]!r}" if seen else " Nothing was received."))
            ser.close()
            sys.exit(1)
    return boards.identity(seen), False


def expected_model(model_path):
    """Size and FNV-1a fingerprint of the .tflite the operator says is on the board, or None."""
    if not model_path or not os.path.isfile(model_path):
        return None
    return {"bytes": os.path.getsize(model_path), "fnv1a": boards.fnv1a32_file(model_path)}


def verify_model_identity(found, has_identity, expected, model_path, allow_mismatch=False):
    """True when the board carries exactly the .tflite given with --model_path.

    Stops the run (exit code 2) when that cannot be established: the firmware
    printed no fingerprint, there is no file to compare with, or the fingerprint
    or the size differ. allow_mismatch (--allow_model_mismatch) turns the stop
    into a warning and the run is recorded with model.identity_verified = false.
    """
    problem = None
    if not has_identity or found["model_fnv1a"] is None:
        problem = ("The firmware did not report the fingerprint of its model (no MODEL_FNV1A line): it "
                   "is a build earlier than November 2026, so the model on the board cannot be verified. "
                   "Flash the current sketch (python src/compile_upload_arduino.py --target pil).")
    elif expected is None:
        problem = (f"There is no .tflite to compare the fingerprint of the board with "
                   f"({found['model_fnv1a']}, {found['model_bytes']} bytes): --model_path "
                   f"{model_path!r} is not a file.")
    else:
        print(f"[INFO] Chip fingerprint   : fnv1a={found['model_fnv1a']} bytes={found['model_bytes']}")
        print(f"[INFO] .tflite fingerprint: fnv1a={expected['fnv1a']} bytes={expected['bytes']}  "
              f"({os.path.basename(model_path)})")
        if found["model_fnv1a"] == expected["fnv1a"] and found["model_bytes"] == expected["bytes"]:
            print("[OK]   IDENTITY VERIFIED: the board is running exactly the given .tflite file.")
            return True
        problem = ("MODEL MISMATCH ON BOARD: the fingerprint of the model in flash is not that of the "
                   "given .tflite. Write the header again with tflite_to_c.py, compile, upload "
                   "(python src/compile_upload_arduino.py --target pil) and run again.")
    print(f"[ERROR] {problem}")
    if not allow_mismatch:
        print("        Nothing was measured. --allow_model_mismatch measures anyway and records "
              "model.identity_verified = false.")
        sys.exit(EXIT_REFUSED)
    print("[WARN] --allow_model_mismatch set: this run is NOT verified against the board.")
    return False


def resolve_input_shape(width, height, grayscale, announced_shape):
    """The size of the images to send, checked against what the firmware announces.

    Returns (width, height, verified). A width or height left out (None) is
    taken from the firmware. Stops the run (exit code 2) when the images would
    not have the input shape of the model on the board, or when neither the
    command line nor the firmware says which size to send: an image of another
    size is padded or cut by the firmware and gives a complete, wrong run.
    """
    channels = 1 if grayscale else 3
    if announced_shape:
        w = announced_shape["width"] if width is None else width
        h = announced_shape["height"] if height is None else height
        if (w, h, channels) != (announced_shape["width"], announced_shape["height"], announced_shape["channels"]):
            print(f"[ERROR] The model on the board takes images of {announced_shape['width']}x"
                  f"{announced_shape['height']} with {announced_shape['channels']} channel(s), and this run "
                  f"would send {w}x{h} with {channels}. Use --width {announced_shape['width']} --height "
                  f"{announced_shape['height']} (or leave them out) and "
                  + ("--color." if announced_shape["channels"] == 3 else "no --color."))
            sys.exit(EXIT_REFUSED)
        return w, h, True
    if width is None or height is None:
        print("[ERROR] This firmware does not announce the input shape of its model: pass --width and "
              "--height (the size the model takes; an image of another size gives a wrong run).")
        sys.exit(EXIT_REFUSED)
    return width, height, False


def open_session(port, baud, pre_delay, board_key, model_path, width, height, grayscale,
                 allow_model_mismatch=False):
    """Opens the port and checks the board before anything is measured or written.

    Returns (ser, session): the open port and a dict with what the firmware
    announced ('identity'), the fingerprint of the given file ('expected'),
    whether the model was verified ('identity_verified') and the size of the
    images to send ('width', 'height', 'shape_verified'). Stops the run, with
    the port closed, when the board is not the one named with --board, the
    model cannot be verified or the images would have another shape.
    """
    expected = expected_model(model_path)
    print(f"[INFO] Connecting to {port} @ {baud} baud ...")
    try:
        ser = serial.Serial(port, baud, timeout=10)
    except serial.SerialException as e:
        print(f"ERROR opening the port: {e}")
        sys.exit(1)
    opened_at = time.time()

    print(f"[INFO] Port open. Waiting {pre_delay}s before talking to the board...")
    time.sleep(pre_delay)
    try:
        found, has_identity = handshake(ser, opened_at)
        problem = boards.mismatch(board_key, found["board"])
        if problem:
            print(f"[ERROR] Wrong board: {problem}")
            sys.exit(EXIT_REFUSED)
        if found["board"] is None:
            print("[WARN] The firmware did not announce its board (no BOARD: line): it is "
                  "a build earlier than the two-board sketch. The board is recorded as null.")
        verified = verify_model_identity(found, has_identity, expected, model_path, allow_model_mismatch)
        width, height, shape_verified = resolve_input_shape(width, height, grayscale, found["input_shape"])
    except serial.SerialException as exc:
        print(f"[ERROR] The serial link to {port} was lost before the first image ({exc}). "
              "Nothing was measured or written.")
        try:
            ser.close()
        except serial.SerialException:
            pass
        sys.exit(1)
    except SystemExit:
        ser.close()
        raise
    return ser, {"identity": found, "expected": expected, "identity_verified": verified,
                 "width": width, "height": height, "shape_verified": shape_verified}


def exchange(ser, packet, port, baud, position=""):
    """One image: packet out, answer in.

    Returns (ser, predicted_class, metrics, reconnects). If the serial link
    drops in the middle (the port disappears and the driver raises), the board
    keeps running, so the port is reopened and the SAME image is sent again;
    the firmware restarts a packet when it sees a new start marker. The latency
    is measured on the chip, so a repeated exchange gives a valid measurement
    like any other. ser is the port in use at the end, which is a new one after
    a reconnection. Raises LinkLost when the port does not come back.
    """
    attempt = 0
    reconnects = 0
    while True:
        attempt += 1
        predicted_class = None
        current_metrics = {}
        try:
            ser.reset_input_buffer()
            ser.write(packet)
            ser.flush()
            print("[OK]   Packet sent. Waiting for the predicted class...")

            # Read the response: the Arduino sends only the class number, then the latencies
            timeout = time.time() + RESPONSE_TIMEOUT_S
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

                    if boards.is_end_of_inference(line):
                        break

                    try:
                        # If the line is just an integer, assume it is the predicted class
                        parsed_int = int(line.strip())
                        predicted_class = parsed_int
                    except ValueError:
                        pass
                else:
                    time.sleep(0.05)
            return ser, predicted_class, current_metrics, reconnects
        except serial.SerialException as exc:
            reconnects += 1
            print(f"  [WARN] Serial link lost ({exc}).")
            try:
                ser.close()
            except serial.SerialException:
                pass
            ser = reopen_port(port, baud) if attempt < MAX_ATTEMPTS_PER_IMAGE else None
            if ser is None:
                raise LinkLost(f"serial link lost{position} and did not recover "
                               f"(attempt {attempt} of {MAX_ATTEMPTS_PER_IMAGE}): {exc}", reconnects)
            print(f"  [INFO] Port reopened; sending the same image again "
                  f"(attempt {attempt + 1} of {MAX_ATTEMPTS_PER_IMAGE}).")


def send_image(port: str, baud: int, image_path: str,
               width: int, height: int, grayscale: bool,
               pre_delay: float, model_tag: str,
               output_dir: str = os.path.join("results", "pil"),
               model_path: str = None,
               board_key: str = boards.DEFAULT_BOARD,
               allow_model_mismatch: bool = False):
    """
    Opens the serial port, checks the board like send_folder does, and sends
    one image with the #...@ protocol. Its latencies are appended to
    latency_metrics_single_<model>.csv.
    """
    ser, session = open_session(port, baud, pre_delay, board_key, model_path, width, height,
                                grayscale, allow_model_mismatch)
    width, height = session["width"], session["height"]
    raw_payload = preprocess_image(image_path, width, height, grayscale)
    packet = build_packet(raw_payload)

    print(f"[INFO] Sending packet: {len(packet)} bytes total "
          f"(raw payload={len(raw_payload)}, escaped={len(packet)-2})")
    try:
        ser, predicted_class, current_metrics, _ = exchange(ser, packet, port, baud)
    except LinkLost as lost:
        print(f"[ERROR] {lost}. Nothing was written.")
        sys.exit(EXIT_INCOMPLETE)
    finally:
        try:
            ser.close()
        except serial.SerialException:
            pass
    print("\n[INFO] Port closed. End of transmission.")
    if predicted_class is None:
        print(f"[ERROR] The board did not answer in {RESPONSE_TIMEOUT_S} s. Nothing was written.")
        sys.exit(EXIT_INCOMPLETE)
    print(f"  -> Predicted class: {predicted_class}")

    csv_path = os.path.join(output_dir,
                            f"latency_metrics_single_{model_tag}.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    # Check if file exists to write header or not
    file_exists = os.path.isfile(csv_path)
    metrics_headers = ["TEMP_C", "CYC_PRE", "CYC_INF", "CYC_POST", "CYC_TOTAL", "US_PRE", "US_INF", "US_POST", "US_TOTAL"]
    with open(csv_path, "a", newline="") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=";")
        if not file_exists:
            csv_writer.writerow(["Image"] + metrics_headers)
        csv_writer.writerow([os.path.basename(image_path)] + [current_metrics.get(h, "") for h in metrics_headers])
    print(f"[INFO] Metrics saved to {csv_path}")


def send_folder(port: str, baud: int, folder: str,
                width: int, height: int, grayscale: bool,
                pre_delay: float, gap: float,
                model_tag: str,
                model_path: str = None,
                n_samples: int = None,
                output_dir: str = os.path.join("results", "pil"),
                board_key: str = boards.DEFAULT_BOARD,
                allow_model_mismatch: bool = False,
                max_unanswered: int = MAX_CONSECUTIVE_UNANSWERED):
    """
    Sends the images in `folder` to the Arduino in sequence.

    Before the first image the board is asked what it is running (see
    open_session): the run stops, with nothing written, when the board, the
    model or the input shape is not the expected one. A run that does not reach
    its last image keeps what it measured and is marked completed = false.

    If `n_samples` is None, every image found is sent.
    If `n_samples` is an integer, that number of images is selected
    at random (without replacement) before sending them.

    The Arduino returns the predicted class index along with on-device telemetry
    lines (TEMP_C, CYC_*, US_*). On completion, a prediction summary is printed
    and artifacts are saved under `output_dir` (latency_metrics_*.csv,
    predictions_*.csv, confusion_matrix_*.png, and metrics_*.json).

    Parameters
    ----------
    port       : Serial port (e.g. 'COM9' or '/dev/ttyUSB0')
    baud       : Baud rate
    folder     : Folder containing the images
    width      : Width of the images sent; None takes the one the firmware announces
    height     : Height of the images sent; None takes the one the firmware announces
    grayscale  : True -> grayscale, False -> RGB
    pre_delay  : Seconds to wait before sending the first image
    gap        : Seconds to wait between consecutive images
    model_tag  : Model stem used to name the output files
    output_dir : Folder for the output files (default: results/pil)
    model_path : Optional path to the flashed .tflite file (hashed in metrics JSON)
    n_samples  : Number of images to select at random.
                 If None (or >= total available), all of them are sent.
    board_key  : Board named with --board; the run stops if the firmware
                 announces a different one
    allow_model_mismatch : measure although the model on the board could not be
                 verified against model_path (recorded as identity_verified = false)
    max_unanswered : images in a row without an answer after which the run is
                 given up; 0 never gives up
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

    # The board is checked before any output file is opened: a run refused
    # here leaves the results of an earlier run as they were.
    ser, session = open_session(port, baud, pre_delay, board_key, model_path, width, height,
                                grayscale, allow_model_mismatch)
    width, height = session["width"], session["height"]
    announced = session["identity"]["board"]   # board named by the firmware in its BOARD: line

    total = len(image_data)

    csv_path = os.path.join(output_dir,
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
    pred_csv_path = os.path.join(output_dir,
                                 f"predictions_{model_tag}.csv")
    pred_file = open(pred_csv_path, "w", newline="", encoding="utf-8")
    pred_writer = csv.writer(pred_file, delimiter=";")
    pred_writer.writerow(["Image", "RelPath", "TrueLabel", "TrueClass",
                          "PredLabel", "PredClass", "Correct"])

    class_counts: Counter = Counter()
    errors = 0
    reconnects = 0     # times the serial link dropped and the port was reopened
    recorded = 0       # images with a row in the two CSV files
    unanswered_in_a_row = 0
    stopped_because = None   # why the run ended before its last image
    y_true = []
    y_pred = []
    # Accumulated in addition to the CSV: the JSON summarizes per phase, and
    # re-reading the just-written CSV only for that would be a detour.
    latencies = []
    idx = 0
    rel_path = ""      # image being processed, relative to the folder

    try:
        for idx, (image_path, true_label) in enumerate(image_data, start=1):
            rel_path = os.path.relpath(image_path, folder).replace(os.sep, "/")
            print(f"\n[INFO] ({idx}/{total}) Processing: {os.path.basename(image_path)}")

            raw_payload = preprocess_image(image_path, width, height, grayscale)
            packet = build_packet(raw_payload)

            print(f"[INFO] Sending packet: {len(packet)} bytes total "
                  f"(raw payload={len(raw_payload)}, escaped={len(packet)-2})")

            try:
                ser, predicted_class, current_metrics, dropped = exchange(
                    ser, packet, port, baud, position=f" at image {idx} of {total}")
            except LinkLost as lost:
                reconnects += lost.reconnects
                stopped_because = str(lost)
                break
            reconnects += dropped

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
            label_known = (true_label != -1)
            got_response = (predicted_class is not None)
            if label_known and got_response:
                correct = 1 if true_label == predicted_class else 0
            else:
                correct = ""
            # The two rows of an image are written one after the other and flushed:
            # a serial benchmark can be interrupted, what has already been measured
            # should not be lost, and the two files should hold the same images.
            csv_writer.writerow([os.path.basename(image_path)]
                                + [current_metrics.get(h, "") for h in metrics_headers])
            pred_writer.writerow([
                os.path.basename(image_path),
                rel_path,
                true_label if label_known else "",
                _class_name(true_label, class_names) if label_known else "",
                predicted_class if got_response else "",
                _class_name(predicted_class, class_names) if got_response else "",
                correct,
            ])
            csv_file.flush()
            pred_file.flush()
            latencies.append(dict(current_metrics))
            recorded += 1

            # A board that stopped answering would make every remaining image
            # wait for its whole timeout: give the run up instead.
            unanswered_in_a_row = 0 if got_response else unanswered_in_a_row + 1
            if max_unanswered and unanswered_in_a_row >= max_unanswered:
                stopped_because = (f"the board stopped answering: {unanswered_in_a_row} images in a row "
                                   f"without an answer (last one: image {idx} of {total})")
                break

            if idx < total:
                print(f"[INFO] Waiting {gap}s before the next image...")
                time.sleep(gap)
    except KeyboardInterrupt:
        stopped_because = f"interrupted by the operator (Ctrl+C) at image {idx} of {total}"
    except Exception as exc:   # noqa: BLE001 - whatever ends the run, what was measured is kept
        print(f"[ERROR] {type(exc).__name__}: {exc}")
        # The reason goes to the metrics file: the image is named by its path inside
        # the folder, and the text of the error, which can carry a local path, is not.
        stopped_because = f"{type(exc).__name__} at image {idx} of {total} ({rel_path})"
    finally:
        csv_file.close()
        pred_file.close()
        try:
            ser.close()
        except serial.SerialException:
            pass

    completed = stopped_because is None and recorded == total
    if not completed and stopped_because is None:
        stopped_because = f"the run ended unexpectedly after image {recorded} of {total}"
    print(f"\n[INFO] Latency metrics saved to {csv_path}")
    print(f"[INFO] Per-image predictions saved to {pred_csv_path}")
    print("\n[INFO] Port closed. " + ("All images sent." if completed
                                      else f"{recorded} of {total} images recorded."))
    if reconnects:
        print(f"[WARN] The serial link dropped {reconnects} time(s) during the run; each time the port "
              f"was reopened and the image sent again.")

    total_planned = total
    total = recorded   # from here on the figures cover the images that were recorded
    answered = total - errors
    print("\n" + "=" * 45)
    print("  PREDICTION SUMMARY")
    print(f"  Folder  : {folder}")
    print(f"  Total   : {total} images  |  Answered: {answered}  |  No response: {errors}"
          + ("" if completed else f"  |  Planned: {total_planned}"))
    print("=" * 45)
    if class_counts:
        print(f"  {'Class':>6}  {'Count':>9}  {'Percentage':>11}")
        print("  " + "-" * 32)
        for cls in sorted(class_counts):
            cnt = class_counts[cls]
            pct = cnt / answered * 100 if answered else 0
            print(f"  {cls:>6}  {cnt:>9}  {pct:>10.1f}%")
    print("=" * 45)

    # Defaults: if there are no ground-truth labels or sklearn is missing, the
    # JSON is written anyway with these fields as null and the latency intact.
    global_metrics = None
    report_dict = None
    confusion = None
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
                    target_names.append(f"Invalid_{lab}")

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

            # Same criterion as mil_benchmark.py (MIL): the name is derived from the
            # evaluated model, so that two runs with different models do not
            # overwrite each other's matrix.
            os.makedirs(output_dir, exist_ok=True)
            cm_plot_name = f"confusion_matrix_{model_tag}.png"
            cm_plot_path = os.path.join(output_dir, cm_plot_name)
            plt.savefig(cm_plot_path)
            print(f"Confusion matrix plot saved to: {cm_plot_path}")

            # Collected here, inside the try, so that the JSON further below
            # finds them already computed; if sklearn is missing they stay None.
            global_metrics = {
                "accuracy": float(accuracy),
                "precision_weighted": float(precision),
                "recall_weighted": float(recall),
                "f1_weighted": float(f1),
                "f1_macro": float(f1_macro),
            }
            report_dict = classification_report(y_true, y_pred, labels=labels_present,
                                                target_names=target_names,
                                                zero_division=0, output_dict=True)
            confusion = {
                "labels": target_names,
                "matrix": cm.tolist(),
                "order": "rows = true label, columns = predicted label",
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
    # also information. It is also written when the run was cut short: its
    # figures then cover the images recorded, and run.completed says so.
    # ------------------------------------------------------------------
    json_dir = output_dir
    os.makedirs(json_dir, exist_ok=True)
    metrics_path = os.path.join(json_dir, f"metrics_{model_tag}.json")

    results = {
        "level": "PIL",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "path": model_path,
            "name": model_tag,
            "sha256": _sha256_file(model_path),
            "bytes": os.path.getsize(model_path) if model_path and os.path.isfile(model_path) else None,
            "fnv1a_tflite": (session["expected"] or {}).get("fnv1a"),
            "fnv1a_chip": session["identity"]["model_fnv1a"],
            "identity_verified": session["identity_verified"],
            "arena_used_bytes": session["identity"]["arena_used_bytes"],
            "note": ("sha256 and fnv1a_tflite are those of the .tflite given with --model_path. "
                     "identity_verified=true means that, before the first image, the FNV-1a "
                     "fingerprint and the size of g_model[] on the board matched that file: the "
                     "chip ran exactly it. false means the run was allowed with "
                     "--allow_model_mismatch and nothing ties the file to the board."),
        },
        "run": {
            "port": port,
            "board": announced,
            "baud": baud,
            "completed": completed,
            "stopped_because": stopped_because,
            "input_shape": {"width": width, "height": height,
                            "channels": 1 if grayscale else 3},
            "input_shape_verified": session["shape_verified"],
            "pre_delay_s": pre_delay,
            "gap_s": gap,
            "n_samples_requested": n_samples,
            "n_reconnects": reconnects,
            "max_unanswered": max_unanswered,
        },
        "dataset": {
            "folder": folder,
            "class_names": list(class_names),
            "n_images_planned": total_planned,
            "n_images": total,
            "n_answered": answered,
            "n_unanswered": errors,
        },
        "global_metrics": global_metrics,
        "per_class_metrics": report_dict,
        "confusion_matrix": confusion,
        "latency": {
            "source": ("DWT->CYCCNT cycle counter of the Cortex-M7, reported by "
                       "the firmware in the CYC_*/US_* lines"),
            "n_images_measured": len(latencies),
            "p95_method": "nearest rank on the sorted sample, no interpolation",
            "per_field": _latency_summary(latencies, metrics_headers),
        },
        "prediction_counts": {str(k): v for k, v in sorted(class_counts.items())},
        # Host side only: what decoded the images, sent them and computed the figures
        # (see artifacts.software_versions). The model runs on the board.
        "software": artifacts.software_versions(["numpy", "pillow", "pyserial", "scikit_learn"]),
        "artifacts": {
            "confusion_matrix_png": cm_plot_path,
            "latency_csv": csv_path,
            "predictions_csv": pred_csv_path,
        },
    }

    with open(metrics_path, "w", encoding="utf-8") as fh:
        # Output schema documented in docs/results_schema.md - update it if you add, rename or remove keys.
        json.dump(results, fh, indent=2, ensure_ascii=False, default=_json_safe)
    print(f"[INFO] JSON metrics saved to {metrics_path}")

    if not completed:
        print(f"\n[ERROR] The run did NOT finish: {total} of {total_planned} images recorded "
              f"({stopped_because}).")
        print("        The CSV files hold what was measured and the metrics file is marked "
              "completed = false. Check the board and the cable, then run again with --overwrite.")
        sys.exit(EXIT_INCOMPLETE)



def parse_args():
    parser = cli.ArgumentParser(
        description="Sends image(s) to the Arduino over serial (#...@ protocol)"
    )
    parser.add_argument('--port',   default="COM9", required=False,
                        help="Serial port (e.g. COM9 or /dev/ttyUSB0)")
    parser.add_argument('--board', choices=sorted(boards.BOARDS), default=boards.DEFAULT_BOARD,
                        help="Board the firmware runs on (default: portenta). It sets "
                             "the default output folder and is checked against the "
                             "board the firmware announces")

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
                             "executed on the PC: its fingerprint is compared with the "
                             "one the firmware reports before anything is measured, its "
                             "SHA-256/size are recorded in metrics_*.json and its stem "
                             "names the results/pil/ artifacts (required)")

    parser.add_argument('--baud',   type=int, default=115200,
                        help="Baud rate (default: 115200)")
    parser.add_argument('--width',  type=int, default=None,
                        help="Width of the images sent. Default: the input width the "
                             "firmware announces; a different value stops the run")
    parser.add_argument('--height', type=int, default=None,
                        help="Height of the images sent. Default: the input height the "
                             "firmware announces; a different value stops the run")
    parser.add_argument('--color',  action='store_true',
                        help="Send in RGB color (default: grayscale). It must match the "
                             "channels of the model input the firmware announces")
    parser.add_argument('--delay',  type=float, default=1.0,
                        help="Seconds to wait after opening the port before talking to "
                             "the board (default: 1.0)")
    parser.add_argument('--gap',    type=float, default=0.5,
                        help="Seconds between consecutive images (default: 0.5)")
    parser.add_argument('--count',  type=int, default=None, metavar='N',
                        help="Number of random images to send from the folder. "
                             "If not specified, all of them are sent.")
    parser.add_argument('--output_dir', default=None,
                        help="Output directory for the matrix, JSON and CSV files "
                             "(default: results/pil; results/nicla/pil with --board nicla)")
    parser.add_argument('--overwrite', action='store_true',
                        help="Replace the matrix, JSON and CSV files if they already exist")
    parser.add_argument('--max_unanswered', type=int, default=MAX_CONSECUTIVE_UNANSWERED, metavar='N',
                        help="Give the run up after N images in a row without an answer "
                             f"(default: {MAX_CONSECUTIVE_UNANSWERED}; each one waits "
                             f"{RESPONSE_TIMEOUT_S} s). 0 never gives up. A run given up keeps "
                             "what it measured, marked completed = false")
    parser.add_argument('--allow_model_mismatch', action='store_true',
                        help="DEBUG ONLY: measure although the model on the board could not "
                             "be verified against --model_path (another fingerprint, a firmware "
                             "that reports none, or a file that does not exist). The run is "
                             "recorded with model.identity_verified = false")
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
        size = f"{args.width}x{args.height}" if args.width and args.height else "<width>x<height>"
        print("Run the dataset split first:")
        print(f"  python src/split_dataset.py"
              f" --input_dir data/processed/{size}"
              f" --output_dir {args.splits_dir}")
        sys.exit(1)
    return folder


if __name__ == '__main__':
    args = parse_args()
    port = normalize_port(args.port)
    if args.output_dir is None:
        args.output_dir = boards.results_dir(args.board, "pil")

    # Run tag: the model name without extension, the same criterion that
    # mil_benchmark.py uses to name its confusion matrix.
    model_tag = os.path.splitext(os.path.basename(args.model_path))[0]

    # Found out before the port is opened: without the file there is nothing to
    # compare the fingerprint of the board with.
    if not os.path.isfile(args.model_path) and not args.allow_model_mismatch:
        print(f"[ERROR] --model_path is not a file: {args.model_path}")
        print("        Its fingerprint is compared with the one the firmware reports. Pass the "
              ".tflite that was flashed (or --allow_model_mismatch to use only its name).")
        sys.exit(EXIT_REFUSED)
    if args.max_unanswered < 0:
        print("[ERROR] --max_unanswered must be 0 (never give up) or a positive number.")
        sys.exit(EXIT_REFUSED)

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
            output_dir=args.output_dir,
            model_path=args.model_path,
            board_key=args.board,
            allow_model_mismatch=args.allow_model_mismatch,
        )
    else:
        folder = resolve_folder(args)
        # Checked before opening the port: a run must not replace the results of
        # another one (the published ones are tracked in results/).
        artifacts.refuse_to_overwrite(
            [os.path.join(args.output_dir, name) for name in
             (f"confusion_matrix_{model_tag}.png", f"predictions_{model_tag}.csv",
              f"metrics_{model_tag}.json", f"latency_metrics_{model_tag}.csv")],
            args.overwrite,
            hint="write somewhere else with --output_dir")
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
            output_dir=args.output_dir,
            board_key=args.board,
            allow_model_mismatch=args.allow_model_mismatch,
            max_unanswered=args.max_unanswered,
        )
