#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calibrate_hil_rig.py - Checks the HIL rig (the camera of the board against the
stimulus monitor) before a hil_benchmark.py session.

It shows its patterns through the stimulus window of the bench and reads the
captured frames with the serial protocol of the bench (show_stimulus, the 'T'
trigger, the 'F1' frame dump), so the geometry measured here is the geometry
the bench will see. The board must run the HIL firmware
(deployment/hil_firmware); the model inside it does not matter, because only
the captured frames are used.

Patterns (160x120, displayed exactly like a dataset image):
  geometry   black field with 4 corner squares + 1 centre square. The captured
             centroids are fitted to the nominal ones (similarity transform):
             shift, scale, rotation and residual (keystone / lens distortion).
  gray       8 vertical bands 0..255: tone response, clipping, dynamic range.
  flat       uniform mid gray: vignetting (corner/centre ratio) and exposure
             stability across --repeats captures.
  focus      coarse (16 px) and fine (4 px) checkerboards: fine/coarse
             contrast ratio (a one-number focus check).
  images     --images test images: normalised cross-correlation between the
             captured frame and the image shown (the end-to-end fidelity the
             ladder measures).

Every figure is compared with an acceptance criterion and reported as PASS or
WARN. Nothing is enforced: the report is evidence of the state of the rig, to
keep next to the results of the session.

Usage (repository root):
  # live loop while you move the camera or the monitor (geometry only, Ctrl+C to stop)
  python src/calibrate_hil_rig.py --port COM9 --live
  # full report, written to results/hil/calibration/
  python src/calibrate_hil_rig.py --port COM9 --report --images A.png B.png C.png

Use the same --settle, --windowed and --monitor as in the session that follows.

Requirements: those of hil_benchmark.py (pyserial, opencv-python, numpy) and Pillow.
"""
import csv
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone

import boards
import cli
import hil_benchmark as hb    # ends with a message when pyserial, OpenCV or numpy are missing
import images

cv2, np, serial = hb.cv2, hb.np, hb.serial

W, H = 160, 120                           # frame the firmware sends back, and size of the patterns
SQ = 12                                   # corner square side (px at 160x120)
DRAW = [(14, 14), (146, 14), (14, 106), (146, 106), (80, 60)]   # square centres as drawn
NOMINAL = np.array(DRAW, np.float32) - 0.5                     # their pixel-centroid positions
SQUARE_NAMES = ["top-left", "top-right", "bottom-left", "bottom-right", "centre"]
GRAY_LEVELS = [0, 36, 73, 109, 146, 182, 219, 255]

# Acceptance criteria (pixels at 160x120). Reported as PASS/WARN, never enforced.
CRITERIA = {"shift_px": 2.0, "scale_err": 0.03, "rot_deg": 1.0, "resid_px": 1.5,
            "range_min": 150, "vignette_min": 0.60, "exposure_sd_max": 2.0,
            "focus_ratio_min": 0.50, "clip_pct_max": 2.0}

# What the live mode says (the preview window and its lines in the terminal), per --lang. The
# report, its JSON and every other message are in English whatever the language.
TEXT = {
    "en": {
        "too_few": "only %d bright squares visible: the camera does not see the stimulus (aim/distance)",
        "no_layout": "%d bright spots but not the square layout: reflections of a lamp/window (darken the room) "
                     "or the stimulus is far off the frame",
        "small": "image too small in the frame: move the camera CLOSER",
        "large": "image too large (cropped): move the camera AWAY",
        "shift_right": "image shifted right by %.1f px: aim the camera to the RIGHT",
        "shift_left": "image shifted left by %.1f px: aim the camera to the LEFT",
        "shift_down": "image shifted down by %.1f px: aim the camera DOWN",
        "shift_up": "image shifted up by %.1f px: aim the camera UP",
        "rot_ccw": "rotated %.1f deg: roll the camera counter-clockwise (seen from behind it)",
        "rot_cw": "rotated %.1f deg: roll the camera clockwise (seen from behind it)",
        "resid": "residual %.1f px: camera not parallel to the screen (tilt) or lens distortion",
        "missing": "square(s) not seen: %s (cut off by the frame edge?)",
        "extra": "%d extra bright spot(s): reflection of a lamp/window on the screen",
        "ok": "geometry within tolerance",
        "squares": {},
        "pass": "PASS", "adjust": "ADJUST", "warn": "WARN",
        "bar": "[%03d] %s  shift (%+.1f, %+.1f) px  scale %.3f  rot %+.2f deg  resid %.2f px",
        "bar_none": "[%03d] %s  %d of 5 squares found",
        "window": "PHLAME calibration preview (camera frame)",
        "live_start": "[LIVE] Geometry loop. Adjust the camera; Ctrl+C to stop.",
        "capture_failed": "  capture failed",
        "live_line": "  [%03d] %s shift (%+.1f, %+.1f) px | scale %.3f | rot %+.2f deg | resid %.2f px -> %s",
        "live_none": "  [%03d] %s %s",
        "live_stop": "\n[LIVE] stopped.",
    },
    "es": {
        "too_few": "solo se ven %d cuadros brillantes: la cámara no ve el estímulo (apuntado o distancia)",
        "no_layout": "%d puntos brillantes, pero no en la disposición de los cuadros: reflejos de una lámpara o "
                     "ventana (oscurece la sala) o el estímulo está muy fuera del fotograma",
        "small": "imagen muy pequeña: ACERCA la cámara",
        "large": "imagen muy grande (recortada): ALEJA la cámara",
        "shift_right": "desplazada %.1f px a la derecha: apunta la cámara a la DERECHA",
        "shift_left": "desplazada %.1f px a la izquierda: apunta la cámara a la IZQUIERDA",
        "shift_down": "desplazada %.1f px hacia abajo: apunta la cámara hacia ABAJO",
        "shift_up": "desplazada %.1f px hacia arriba: apunta la cámara hacia ARRIBA",
        "rot_ccw": "girada %.1f grados: gírala en sentido antihorario (vista desde atrás)",
        "rot_cw": "girada %.1f grados: gírala en sentido horario (vista desde atrás)",
        "resid": "residuo %.1f px: cámara no paralela a la pantalla (inclinación) o distorsión de la lente",
        "missing": "cuadro(s) que no se ven: %s (¿cortados por el borde del fotograma?)",
        "extra": "%d punto(s) brillante(s) de más: reflejo de una lámpara o ventana en la pantalla",
        "ok": "geometría dentro de la tolerancia",
        "squares": {"top-left": "arriba-izquierda", "top-right": "arriba-derecha", "bottom-left": "abajo-izquierda",
                    "bottom-right": "abajo-derecha", "centre": "centro"},
        "pass": "CORRECTO", "adjust": "AJUSTAR", "warn": "AJUSTAR",
        "bar": "[%03d] %s  desplazamiento (%+.1f, %+.1f) px  escala %.3f  giro %+.2f grados  residuo %.2f px",
        "bar_none": "[%03d] %s  %d de 5 cuadros encontrados",
        "window": "PHLAME: vista previa del calibrador (fotograma)",     # ASCII: a window title, not drawn text
        "live_start": "[VIVO] Bucle de geometría. Ajusta la cámara; Ctrl+C para detener.",
        "capture_failed": "  falló la captura",
        "live_line": "  [%03d] %s desplazamiento (%+.1f, %+.1f) px | escala %.3f | giro %+.2f grados | "
                     "residuo %.2f px -> %s",
        "live_none": "  [%03d] %s %s",
        "live_stop": "\n[VIVO] detenido.",
    },
}


# ----------------------------------------------------------------------------- patterns
def pattern_geometry():
    a = np.zeros((H, W), np.uint8)
    for cx, cy in DRAW[:4]:
        a[cy - SQ // 2:cy + SQ // 2, cx - SQ // 2:cx + SQ // 2] = 255
    a[55:65, 75:85] = 255
    return a


def pattern_gray():
    a = np.zeros((H, W), np.uint8)
    for k, g in enumerate(GRAY_LEVELS):
        a[:, k * 20:(k + 1) * 20] = g
    return a


def pattern_flat():
    return np.full((H, W), 128, np.uint8)


def pattern_focus():
    y, x = np.mgrid[0:H, 0:W]
    coarse = (((x // 16) + (y // 16)) % 2) * 255
    fine = (((x // 4) + (y // 4)) % 2) * 255
    return np.where(x < W // 2, coarse, fine).astype(np.uint8)


PATTERNS = {"geometry": pattern_geometry, "gray": pattern_gray, "flat": pattern_flat, "focus": pattern_focus}


def write_patterns(folder):
    """Writes the four patterns as PNG (lossless) and returns {name: path}. The
    stimulus window shows files, so a pattern is shown like a dataset image."""
    paths = {}
    for name, make in PATTERNS.items():
        paths[name] = os.path.join(folder, "pattern_%s.png" % name)
        cv2.imwrite(paths[name], make())
    return paths


# ----------------------------------------------------------------------------- board I/O
def frame_dump_on(ser, timeout_s):
    """Asks the firmware to send back every captured frame. Returns (lines read, True if it agreed)."""
    ser.write(hb.CMD_DUMP_ON)
    ser.flush()
    lines, hit = hb.read_lines_until(ser, lambda l: l == "FRAME_DUMP:ON", timeout_s=timeout_s, echo=False)
    return lines, hit is not None


def reopen_port(ser, wait_s=20.0):
    """Opens the same port again after it disappeared (the board was reset and
    its USB port is enumerated again). Returns True when it is open again."""
    try:
        ser.close()
    except (serial.SerialException, OSError):
        pass
    deadline = time.time() + wait_s
    while time.time() < deadline:
        try:
            ser.open()
            return True
        except (serial.SerialException, OSError):
            time.sleep(0.2)
    return False


def wait_for_boot(ser, timeout_s=90.0):
    """Waits for the boot lines of the HIL firmware, up to READY_HIL.

    Meant to run while the operator presses RESET: the port disappears for a
    moment, and it is opened again at once, because the firmware waits only a
    few seconds for the host before it boots without it. Returns (lines read,
    "READY_HIL" | "CAM_INIT:FAIL" | None)."""
    lines = []
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            got, hit = hb.read_lines_until(ser, lambda l: l in ("READY_HIL", "CAM_INIT:FAIL"),
                                           timeout_s=max(0.5, deadline - time.time()), echo=True)
            lines += got
            if hit:
                return lines, hit
        except (serial.SerialException, OSError):
            print("[INFO] The port went away (the board is restarting); opening it again...", flush=True)
            if not reopen_port(ser):
                print("[ERROR] The port did not come back within 20 s.")
                break
            print("[INFO] Port open again; waiting for the firmware...", flush=True)
    return lines, None


def open_board(port, baud, board_key=boards.DEFAULT_BOARD):
    """Opens the port and turns the frame dump on.

    Returns (serial port, board announced by the firmware). The board is None
    when the firmware was already running and its boot lines were not seen, or
    when it is a build that does not announce it. Stops when the firmware runs
    on a board other than board_key or does not accept the frame dump.

    When the firmware does not answer (it is stuck, or it booted with nobody
    listening), the operator is asked to press RESET; this program stays on
    the port, opens it again when it comes back and waits for the boot lines.
    """
    port = hb.normalize_port(port)
    try:
        ser = serial.Serial(port, baud, timeout=10)
    except serial.SerialException as exc:
        print(f"ERROR opening serial port: {exc}")
        sys.exit(1)
    time.sleep(0.5)
    # A firmware that is still booting answers after its boot lines; they are
    # read on the way and say which board this is. (The input is not flushed
    # first, or those lines could be lost.)
    try:
        lines, accepted = frame_dump_on(ser, 6)
        if not accepted and "READY_HIL" in lines:
            # It finished booting while the command was on its way: ask once more.
            more, accepted = frame_dump_on(ser, 5)
            lines += more
    except (serial.SerialException, OSError):
        lines, accepted = [], False          # the board was reset while we asked
    if not accepted and "CAM_INIT:FAIL" not in lines:
        print("[ACTION] The firmware does not answer. Press RESET on the board now; this program "
              "waits for it (up to 90 s).", flush=True)
        boot, hit = wait_for_boot(ser, timeout_s=90)
        lines += boot
        if hit == "READY_HIL":
            _, accepted = frame_dump_on(ser, 5)
        elif hit is None:
            print("[ERROR] The firmware did not boot. Check that the HIL firmware "
                  "(deployment/hil_firmware) is flashed: python src/compile_upload_arduino.py --target hil")
            ser.close()
            sys.exit(1)
    announced = boards.announced_board(lines)
    if "CAM_INIT:FAIL" in lines:
        print("[ERROR] The camera failed to initialize (CAM_INIT:FAIL). On the Portenta H7, "
              "check that the Vision Shield is properly attached.")
        ser.close()
        sys.exit(1)
    if not accepted:
        print("[ERROR] The firmware does not accept the frame dump (F1). Is the HIL "
              "firmware (deployment/hil_firmware) flashed?")
        ser.close()
        sys.exit(1)
    problem = boards.mismatch(board_key, announced)
    if problem:
        print(f"[ERROR] Wrong board: {problem}")
        ser.close()
        sys.exit(2)
    return ser, announced


def capture(ser, stimulus_path, settle):
    """Shows an image file in the stimulus window, triggers one capture and
    returns the frame the camera took (120x160 uint8), or None."""
    if not hb.show_stimulus(stimulus_path, settle):
        print(f"  [WARN] Could not load {stimulus_path}.")
        return None
    ser.reset_input_buffer()
    ser.write(hb.CMD_TRIGGER)
    ser.flush()
    _, hit = hb.read_lines_until(ser, boards.is_end_of_inference, timeout_s=30, echo=False)
    if hit is None:
        return None
    return hb.read_frame(ser)


# ----------------------------------------------------------------------------- metrics
def geometry_metrics(frame):
    _, bw = cv2.threshold(cv2.GaussianBlur(frame, (3, 3), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, _, stats, cents = cv2.connectedComponentsWithStats(bw)
    blobs = []
    for i in range(1, n):                             # square-like blobs only (not glare or a bright edge)
        x, y, w, h, area = stats[i]
        if 15 <= area <= 1500 and 0.4 <= w / h <= 2.5 and area >= 0.5 * w * h:
            blobs.append((area, cents[i]))
    if len(blobs) < 3:
        return {"ok": False, "found": len(blobs), "hint": "only %d bright squares visible: the camera does not see "
                "the stimulus (aim/distance)" % len(blobs)}
    cand = np.array([c for _, c in sorted(blobs, key=lambda b: -b[0])[:9]], np.float32)
    # Hypothesise a similarity transform from every (2 blobs, 2 nominal squares) pairing and keep the one
    # that explains most squares (inliers within 10 px); a reflection or a cut-off square cannot break it.
    best = None
    for i in range(len(cand)):
        for j in range(len(cand)):
            if i == j:
                continue
            for p in range(5):
                for q in range(5):
                    if p == q:
                        continue
                    M, _ = cv2.estimateAffinePartial2D(NOMINAL[[p, q]], cand[[i, j]])
                    if (M is None or not 0.6 < np.hypot(M[0, 0], M[1, 0]) < 1.6
                            or abs(np.degrees(np.arctan2(M[1, 0], M[0, 0]))) > 30):
                        continue
                    proj = NOMINAL @ M[:, :2].T + M[:, 2]
                    d = np.linalg.norm(proj[:, None, :] - cand[None, :, :], axis=2)
                    near = d.argmin(axis=1)
                    inl = [k for k in range(5) if d[k, near[k]] < 10.0]
                    if len({int(near[k]) for k in inl}) < len(inl):
                        continue
                    err = float(np.mean([d[k, near[k]] for k in inl])) if inl else 99.0
                    if best is None or (len(inl), -err) > (len(best[0]), -best[1]):
                        best = (inl, err, near)
    if best is None or len(best[0]) < 3:
        return {"ok": False, "found": len(blobs), "hint": "%d bright spots but not the square layout: reflections "
                "of a lamp/window (darken the room) or the stimulus is far off the frame" % len(blobs)}
    inl, _, near = best
    got = np.full_like(NOMINAL, np.nan)
    for k in inl:
        got[k] = cand[near[k]]
    M, _ = cv2.estimateAffinePartial2D(NOMINAL[inl], got[inl])
    missing = [SQUARE_NAMES[k] for k in range(5) if k not in inl]
    extra = len(blobs) - len(inl)
    pred = NOMINAL @ M[:, :2].T + M[:, 2]
    got = np.where(np.isnan(got), pred, got)
    scale = float(np.hypot(M[0, 0], M[1, 0]))
    rot = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    resid = float(np.sqrt(np.mean(np.sum((pred[inl] - got[inl]) ** 2, axis=1))))
    shift = pred[4] - NOMINAL[4]                      # centre displacement (from the fit)
    out = {"ok": True, "shift_x_px": round(float(shift[0]), 2), "shift_y_px": round(float(shift[1]), 2),
           "scale": round(scale, 4), "rotation_deg": round(rot, 2), "residual_px": round(resid, 2),
           "corners_px": np.round(got, 1).tolist(), "extra_bright_blobs": int(extra), "missing": missing}
    out["pass"] = bool(not missing and abs(shift[0]) <= CRITERIA["shift_px"] and abs(shift[1]) <= CRITERIA["shift_px"]
                       and abs(scale - 1) <= CRITERIA["scale_err"] and abs(rot) <= CRITERIA["rot_deg"]
                       and resid <= CRITERIA["resid_px"])
    return out


def advice(g, lang="en"):
    """What to do about the geometry g, in one line. The report keeps it in English (lang "en")."""
    t = TEXT[lang]
    if not g.get("ok"):
        if "found" not in g:
            return g.get("hint", "")
        # the two ways geometry_metrics gives up, in the order it tests them
        return t["too_few" if g["found"] < 3 else "no_layout"] % g["found"]
    tips = []
    if g["scale"] < 1 - CRITERIA["scale_err"]:
        tips.append(t["small"])
    elif g["scale"] > 1 + CRITERIA["scale_err"]:
        tips.append(t["large"])
    if abs(g["shift_x_px"]) > CRITERIA["shift_px"]:
        tips.append(t["shift_right" if g["shift_x_px"] > 0 else "shift_left"] % abs(g["shift_x_px"]))
    if abs(g["shift_y_px"]) > CRITERIA["shift_px"]:
        tips.append(t["shift_down" if g["shift_y_px"] > 0 else "shift_up"] % abs(g["shift_y_px"]))
    if abs(g["rotation_deg"]) > CRITERIA["rot_deg"]:
        tips.append(t["rot_ccw" if g["rotation_deg"] < 0 else "rot_cw"] % g["rotation_deg"])
    if g["residual_px"] > CRITERIA["resid_px"]:
        tips.append(t["resid"] % g["residual_px"])
    if g.get("missing"):
        tips.insert(0, t["missing"] % ", ".join(t["squares"].get(name, name) for name in g["missing"]))
    if g.get("extra_bright_blobs"):
        tips.append(t["extra"] % g["extra_bright_blobs"])
    return "; ".join(tips) if tips else t["ok"]


def warp_to_nominal(frame, g):
    """Maps the frame back onto the 160x120 stimulus grid (for band/region sampling)."""
    if not g.get("ok"):
        return frame
    M, _ = cv2.estimateAffinePartial2D(np.array(g["corners_px"], np.float32), NOMINAL)
    return cv2.warpAffine(frame, M, (W, H), flags=cv2.INTER_LINEAR) if M is not None else frame


def gray_metrics(frame):
    lv = [float(np.median(frame[30:90, k * 20 + 5:(k + 1) * 20 - 5])) for k in range(len(GRAY_LEVELS))]
    mono = all(b > a for a, b in zip(lv, lv[1:]))
    rng = lv[-1] - lv[0]
    clip = 100.0 * float(np.mean((frame <= 1) | (frame >= 254)))
    return {"levels_in": GRAY_LEVELS, "levels_captured": [round(x, 1) for x in lv], "monotonic": mono,
            "dynamic_range": round(rng, 1), "clipped_pct": round(clip, 2),
            "pass": bool(mono and rng >= CRITERIA["range_min"] and lv[1] > lv[0] + 3 and lv[-2] < lv[-1] - 3)}


def flat_metrics(frames):
    f = np.mean(np.stack(frames).astype(np.float32), axis=0)
    c = float(f[45:75, 60:100].mean())
    corners = [f[5:25, 5:25].mean(), f[5:25, -25:-5].mean(), f[-25:-5, 5:25].mean(), f[-25:-5, -25:-5].mean()]
    vig = float(min(corners) / c) if c > 0 else 0.0
    means = [float(x.mean()) for x in frames]
    sd = float(np.std(means, ddof=1)) if len(means) > 1 else 0.0
    clip = 100.0 * float(np.mean([(x >= 254).mean() for x in frames]))
    return {"centre_mean": round(c, 1), "corner_means": [round(float(x), 1) for x in corners],
            "vignetting_ratio": round(vig, 3), "exposure_means": [round(x, 1) for x in means],
            "exposure_sd": round(sd, 2), "saturated_pct": round(clip, 2), "n_captures": len(frames),
            "pass": bool(vig >= CRITERIA["vignette_min"] and sd <= CRITERIA["exposure_sd_max"]
                         and clip <= CRITERIA["clip_pct_max"])}


def focus_metrics(frame):
    def contrast(r):
        lo, hi = np.percentile(r, 5), np.percentile(r, 95)
        return float((hi - lo) / (hi + lo)) if hi + lo > 0 else 0.0
    cc = contrast(frame[30:90, 10:70])
    cf = contrast(frame[30:90, 90:150])
    ratio = cf / cc if cc > 0 else 0.0
    return {"contrast_coarse": round(cc, 3), "contrast_fine": round(cf, 3), "fine_to_coarse": round(ratio, 3),
            "pass": bool(ratio >= CRITERIA["focus_ratio_min"])}


def image_ncc(stimulus, frame):
    """Normalised cross-correlation of two 160x120 grayscale images (1 = same up to gain and offset)."""
    a = stimulus.astype(np.float64)
    b = frame.astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


# ----------------------------------------------------------------------------- preview
PREVIEW = TEXT["en"]["window"]
BAR_H = 108                               # status bar under the frame: one line of figures, three of advice
ADVICE_COLUMNS = 104                      # characters of advice that fit in one line of the bar


def open_preview(stimulus_monitor, lang="en"):
    """Preview window, on a monitor other than the stimulus one when there is one."""
    name = TEXT[lang]["window"]
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    monitors = hb.list_monitors()
    others = [m for i, m in enumerate(monitors) if i != (stimulus_monitor if stimulus_monitor >= 0 else 0)]
    if others:
        x, y, _, _ = others[0]
        cv2.moveWindow(name, x + 40, y + 40)
    cv2.resizeWindow(name, W * 6, H * 6 + BAR_H)


def wrap_advice(text, columns=ADVICE_COLUMNS, lines=3):
    """The advice in at most `lines` lines of `columns` characters, cut between two pieces of advice
    where it can be (they are joined by '; ')."""
    out, current = [], ""
    for piece in text.split("; "):
        candidate = piece if not current else current + "; " + piece
        if len(candidate) <= columns or not current:
            current = candidate
        else:
            out.append(current + ";")
            current = piece
    out.append(current)
    out = [line[:columns] for line in out[:lines]]
    return out


def _font(pixels):
    """A TrueType font for text the built-in font of OpenCV cannot draw (it is ASCII only), or None."""
    try:
        from PIL import ImageFont
    except ImportError:
        return None
    for name in ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, pixels)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=pixels)     # Pillow 10.1 or later
    except (TypeError, OSError):
        return None


def draw_text(image, text, origin, scale, colour, truetype=False):
    """Draws one line of text on a BGR image, in place. origin is the left end of the baseline, as in
    cv2.putText, which draws it when the text is ASCII and truetype is not asked for; text with
    accents goes through Pillow, and loses them when no TrueType font is at hand."""
    if text.isascii() and not truetype:
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)
        return
    pixels = int(round(30 * scale))
    font = _font(pixels)
    if font is None:
        import unicodedata
        plain = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
        cv2.putText(image, plain, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)
        return
    from PIL import Image, ImageDraw
    picture = Image.fromarray(image[:, :, ::-1])       # BGR -> RGB
    ImageDraw.Draw(picture).text((origin[0], origin[1]), text, font=font, fill=tuple(colour[::-1]), anchor="ls")
    image[:, :, :] = np.asarray(picture)[:, :, ::-1]


def preview_picture(frame, g, k, lang="en"):
    """The picture of the preview window: the frame enlarged, the target squares (green), the squares
    found (red) and a bar with the figures and what to do about them."""
    t = TEXT[lang]
    s = 6
    big = cv2.cvtColor(cv2.resize(frame, (W * s, H * s), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    for cx, cy in NOMINAL:                            # target positions (green)
        cv2.rectangle(big, (int((cx - SQ / 2) * s), int((cy - SQ / 2) * s)),
                      (int((cx + SQ / 2) * s), int((cy + SQ / 2) * s)), (0, 200, 0), 2)
    if g.get("ok"):
        for cx, cy in g["corners_px"]:                # detected squares (red)
            cv2.drawMarker(big, (int(cx * s), int(cy * s)), (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
    bar = np.zeros((BAR_H, W * s, 3), np.uint8)
    if g.get("ok"):
        txt1 = t["bar"] % (k, t["pass"] if g["pass"] else t["adjust"], g["shift_x_px"], g["shift_y_px"], g["scale"],
                           g["rotation_deg"], g["residual_px"])
    else:
        txt1 = t["bar_none"] % (k, t["adjust"], g.get("found", 0))
    col = (0, 200, 0) if g.get("pass") else (0, 165, 255)
    truetype = lang != "en"                           # one font for the whole bar when a line may have accents
    draw_text(bar, txt1, (8, 24), 0.6, col, truetype)
    for row, line in enumerate(wrap_advice(advice(g, lang))):
        draw_text(bar, line, (8, 50 + 24 * row), 0.5, (220, 220, 220), truetype)
    return np.vstack([big, bar])


def show_preview(frame, g, k, lang="en"):
    cv2.imshow(TEXT[lang]["window"], preview_picture(frame, g, k, lang))
    cv2.waitKey(1)


# ----------------------------------------------------------------------------- report
def side_by_side(pairs, path):
    """Picture of every (image shown, frame captured) pair, one pair per row."""
    rows = [np.hstack([cv2.resize(p, (W * 3, H * 3), interpolation=cv2.INTER_NEAREST),
                       np.full((H * 3, 6), 255, np.uint8),
                       cv2.resize(f, (W * 3, H * 3), interpolation=cv2.INTER_NEAREST)]) for p, f in pairs]
    cv2.imwrite(path, np.vstack([np.vstack([r, np.full((6, r.shape[1]), 255, np.uint8)]) for r in rows]))


def resolve_image(name, manifest):
    """A test image given as a path, or as a file name of the test manifest."""
    if os.path.isfile(name):
        return name
    if os.path.isfile(manifest):
        with open(manifest, encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("filename") == name and os.path.isfile(row.get("dst_path", "")):
                    return row["dst_path"]
    print(f"[ERROR] Image not found: {name} (not a file, and not a file name of {manifest} "
          f"whose image is on disk).")
    sys.exit(1)


def live_loop(ser, pattern_path, args, output_dir):
    """Repeats the geometry pattern and says how to move the camera, until Ctrl+C."""
    lang = getattr(args, "lang", "en")
    t = TEXT[lang]
    print(t["live_start"])
    open_preview(args.monitor, lang)
    reference = pattern_geometry()
    k = 0
    try:
        while True:
            frame = capture(ser, pattern_path, args.settle)
            if frame is None:
                print(t["capture_failed"], flush=True)
                continue
            g = geometry_metrics(frame)
            k += 1
            if g["ok"]:
                print(t["live_line"] % (k, t["pass"] if g["pass"] else t["warn"], g["shift_x_px"], g["shift_y_px"],
                                        g["scale"], g["rotation_deg"], g["residual_px"], advice(g, lang)), flush=True)
            else:
                print(t["live_none"] % (k, t["warn"], advice(g, lang)), flush=True)
            show_preview(frame, g, k, lang)
            side_by_side([(reference, frame)], os.path.join(output_dir, "live_latest.png"))
    except KeyboardInterrupt:
        print(t["live_stop"])


def build_report(ser, patterns, args, announced, stimulus_monitor):
    """Runs every pattern once (the flat field --repeats times) and the test images.
    Returns (report, pairs of image shown and frame captured)."""
    report = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "port": args.port,
        "board": announced,
        "settle_s": args.settle,
        "windowed": bool(args.windowed),
        "stimulus_monitor": stimulus_monitor,
        "frame_size": {"width": W, "height": H},
        "criteria": CRITERIA,
        "notes": args.notes,
    }
    frame_geometry = capture(ser, patterns["geometry"], args.settle)
    report["geometry"] = geometry_metrics(frame_geometry) if frame_geometry is not None else {"ok": False}
    report["geometry"]["advice"] = advice(report["geometry"])
    frame_gray = capture(ser, patterns["gray"], args.settle)
    report["gray"] = gray_metrics(warp_to_nominal(frame_gray, report["geometry"])) if frame_gray is not None else {}
    flats = [f for f in (capture(ser, patterns["flat"], args.settle) for _ in range(args.repeats)) if f is not None]
    report["flat"] = flat_metrics(flats) if flats else {}
    frame_focus = capture(ser, patterns["focus"], args.settle)
    report["focus"] = (focus_metrics(warp_to_nominal(frame_focus, report["geometry"]))
                       if frame_focus is not None else {})
    pairs = [(pattern_geometry(), frame_geometry), (pattern_gray(), frame_gray),
             (pattern_flat(), flats[-1] if flats else None), (pattern_focus(), frame_focus)]
    report["images"] = []
    for name in args.images:
        path = resolve_image(name, args.manifest)
        # What the model receives for this file at the PC levels (images.py); the
        # window shows the file itself, as the bench does.
        reference = images.load_image(path, W, H)
        frame = capture(ser, path, args.settle)
        if frame is not None:
            report["images"].append({"image": os.path.basename(path), "ncc": round(image_ncc(reference, frame), 4)})
            pairs.append((reference, frame))
    if report["images"]:
        report["images_ncc_mean"] = round(float(np.mean([x["ncc"] for x in report["images"]])), 4)
    report["all_pass"] = all(report[k].get("pass", False) for k in ("geometry", "gray", "flat", "focus"))
    return report, [(p, f) for p, f in pairs if f is not None]


def print_report(report):
    g, gr, fl, fo = report["geometry"], report["gray"], report["flat"], report["focus"]
    print("\n=== HIL rig calibration ===")
    print("geometry : %s  %s" % ("PASS" if g.get("pass") else "WARN", g.get("advice")))
    if g.get("ok"):
        print("           shift (%+.1f, %+.1f) px, scale %.3f, rotation %+.2f deg, residual %.2f px"
              % (g["shift_x_px"], g["shift_y_px"], g["scale"], g["rotation_deg"], g["residual_px"]))
    if gr:
        print("gray     : %s  captured %s, range %.0f, clipped %.1f %%"
              % ("PASS" if gr["pass"] else "WARN", gr["levels_captured"], gr["dynamic_range"], gr["clipped_pct"]))
    if fl:
        print("flat     : %s  vignetting %.2f, exposure sd %.2f over %d captures, saturated %.1f %%"
              % ("PASS" if fl["pass"] else "WARN", fl["vignetting_ratio"], fl["exposure_sd"], fl["n_captures"],
                 fl["saturated_pct"]))
    if fo:
        print("focus    : %s  fine/coarse contrast %.2f" % ("PASS" if fo["pass"] else "WARN", fo["fine_to_coarse"]))
    for x in report["images"]:
        print("image    : %-28s NCC %.3f" % (x["image"], x["ncc"]))
    print("overall  : %s" % ("PASS" if report["all_pass"] else "WARN (see the lines above)"))


def parse_args(argv=None):
    parser = cli.ArgumentParser(
        description="Check the HIL rig (camera of the board against the stimulus monitor) before a "
                    "hil_benchmark.py session: geometry, tone response, vignetting, exposure and focus.")
    parser.add_argument("--port", default="COM9", help="Serial port (e.g. COM9 or /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    parser.add_argument("--board", choices=sorted(boards.BOARDS), default=boards.DEFAULT_BOARD,
                        help="Board the firmware runs on (default: portenta). It sets the default output "
                             "folder and is checked against the board the firmware announces")
    parser.add_argument("--settle", type=float, default=1.5,
                        help="Seconds each pattern stays on screen before the capture; same meaning and "
                             "default as in hil_benchmark.py (default: 1.5)")
    parser.add_argument("--live", action="store_true",
                        help="Repeat the geometry pattern and say how to move the camera, until Ctrl+C")
    parser.add_argument("--report", action="store_true",
                        help="Run every pattern once and write the report (JSON and picture)")
    parser.add_argument("--repeats", type=int, default=5,
                        help="Captures of the flat field, for the exposure stability (default: 5)")
    parser.add_argument("--images", nargs="*", default=[],
                        help="Test images (paths, or file names of the test manifest) to compare with "
                             "their captured frame")
    parser.add_argument("--splits_dir", default=os.path.join("data", "splits"),
                        help="Directory with the splits of split_dataset.py (default: data/splits); its "
                             "test_manifest.csv resolves the file names given with --images")
    parser.add_argument("--manifest", default=None,
                        help="Test manifest (default: <splits_dir>/test_manifest.csv)")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: results/hil/calibration; "
                             "results/nicla/hil/calibration with --board nicla)")
    parser.add_argument("--windowed", action="store_true",
                        help="Open a normal resizable window instead of fullscreen, as in hil_benchmark.py")
    parser.add_argument("--monitor", type=int, default=-1, metavar="N",
                        help="Monitor that shows the patterns; same as hil_benchmark.py --monitor")
    parser.add_argument("--notes", default="", help="Free-form notes kept in the report")
    parser.add_argument("--lang", choices=sorted(TEXT), default="en",
                        help="Language of the live mode, that is, of the preview window and of its lines in the "
                             "terminal (default: en). The report is in English whatever the language")
    args = parser.parse_args(argv)
    if not (args.live or args.report):
        parser.error("choose --live and/or --report")
    if args.manifest is None:
        args.manifest = os.path.join(args.splits_dir, "test_manifest.csv")
    if args.output_dir is None:
        args.output_dir = os.path.join(boards.results_dir(args.board, "hil"), "calibration")
    return args


def main(argv=None):
    args = parse_args(argv)
    hb.resolve_monitor(args.monitor)          # a monitor that does not exist: found out before the port is opened
    ser, announced = open_board(args.port, args.baud, args.board)
    os.makedirs(args.output_dir, exist_ok=True)
    stimulus_monitor = hb.open_stimulus_window(fullscreen=not args.windowed, monitor=args.monitor)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            patterns = write_patterns(tmp)
            if args.live:
                live_loop(ser, patterns["geometry"], args, args.output_dir)
            if args.report:
                report, pairs = build_report(ser, patterns, args, announced, stimulus_monitor)
                json_path = os.path.join(args.output_dir, f"calibration_{stamp}.json")
                with open(json_path, "w", encoding="utf-8") as fh:
                    json.dump(report, fh, indent=2, ensure_ascii=False)
                if pairs:
                    side_by_side(pairs, os.path.join(args.output_dir, f"calibration_{stamp}.png"))
                print_report(report)
                print("[INFO] Wrote %s%s" % (json_path, " (and the .png with every pattern next to its frame)"
                                                 if pairs else ""))
    finally:
        try:
            ser.write(hb.CMD_DUMP_OFF)
            ser.flush()
        except (serial.SerialException, OSError):
            pass
        try:
            ser.close()
        except (serial.SerialException, OSError):
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
