#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
What the host scripts need to know about the boards and their firmware.

There is one sketch per ladder level (deployment/pil_firmware and
deployment/hil_firmware) and each serves every board listed here: the sketch
picks its board block at compile time and says which board it runs on with a
BOARD:<name> line over serial. This module holds the other half: how to
compile for each board, where its results go, the text lines the two
benches wait for, and the lines in which the firmware says what it is running.

Standard library only, so the tests can import it without the heavy packages.
"""
import os

# key used on the command line (--board) -> what changes with the board
BOARDS = {
    "portenta": {
        "label": "Portenta H7",
        "announces": "PORTENTA_H7",
        "camera": "HM01B0",
        "core": "arduino:mbed_portenta",
        "fqbn": "arduino:mbed_portenta:envie_m7:split=100_0",
        "port_match": "portenta",
        "results_dir": "results",
    },
    "nicla": {
        "label": "Nicla Vision",
        "announces": "NICLA_VISION",
        "camera": "GC2145",
        "core": "arduino:mbed_nicla",
        "fqbn": "arduino:mbed_nicla:nicla_vision",
        "port_match": "nicla",
        # The board has no external SDRAM: the tensor arena is a static array
        # whose size is fixed when the sketch is compiled (see --arena_kb).
        "default_arena_kb": 368,
        "results_dir": os.path.join("results", "nicla"),
    },
}
DEFAULT_BOARD = "portenta"

BOARD_PREFIX = "BOARD:"

# The firmware closes every inference with a banner and the benches read until
# they see it. END_MARKERS is what the sketches print; LEGACY_END_MARKERS is
# what they printed until November 2026, still accepted so that a board
# flashed with an earlier build keeps working without being reflashed.
END_MARKERS = ("next image", "next trigger")
LEGACY_END_MARKERS = ("siguiente imagen", "siguiente trigger", "listo")


# The PIL sketch closes its identity lines with this one, at boot and every
# time the host asks for them with an empty packet (#@). The HIL sketch closes
# its boot lines with READY_HIL.
READY_PIL = "READY_PIL"
READY_HIL = "READY_HIL"

# Model fingerprint: 32-bit FNV-1a over the bytes of the .tflite. The firmware
# computes it over g_model[] in flash (MODEL_FNV1A:<hex>) and the host over the
# file; g_model[] is the file byte for byte (tflite_to_c.py), so the two are
# equal exactly when the board carries that file.
FNV1A32_OFFSET = 0x811C9DC5
FNV1A32_PRIME = 0x01000193


def fnv1a32_file(path, chunk=1 << 20):
    """FNV-1a fingerprint of a file as the firmware prints it: 8 hex digits, lower case."""
    h = FNV1A32_OFFSET
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            for b in block:
                h ^= b
                h = (h * FNV1A32_PRIME) & 0xFFFFFFFF
    return "%08x" % h


def _integer(text):
    try:
        return int(text.strip())
    except ValueError:
        return None


def identity(lines):
    """What the firmware says about itself in its parseable lines.

    Returns a dict with the board name (BOARD:), the input shape of the model
    (INPUT_SHAPE:<H>x<W>x<C>), the size and fingerprint of the model in flash
    (MODEL_BYTES:, MODEL_FNV1A:) and the part of the tensor arena it uses
    (ARENA_USED:). A field the firmware did not print, or printed in a form
    that cannot be read, is None: builds earlier than November 2026 print only
    some of them.
    """
    found = {"board": announced_board(lines), "input_shape": None, "model_bytes": None,
             "model_fnv1a": None, "arena_used_bytes": None}
    for line in lines:
        line = line.strip()
        if line.startswith("INPUT_SHAPE:"):
            parts = [_integer(p) for p in line.split(":", 1)[1].lower().split("x")]
            if len(parts) == 3 and None not in parts:
                found["input_shape"] = {"height": parts[0], "width": parts[1], "channels": parts[2]}
        elif line.startswith("MODEL_BYTES:"):
            found["model_bytes"] = _integer(line.split(":", 1)[1])
        elif line.startswith("MODEL_FNV1A:"):
            found["model_fnv1a"] = line.split(":", 1)[1].strip().lower().replace("0x", "") or None
        elif line.startswith("ARENA_USED:"):
            found["arena_used_bytes"] = _integer(line.split(":", 1)[1])
    return found


def is_end_of_inference(line):
    """True for the line that closes one inference (current or earlier firmware)."""
    low = line.lower()
    return any(marker in low for marker in END_MARKERS + LEGACY_END_MARKERS)


def results_dir(board, level):
    """Default output folder of a board-level bench: results/<level> for the
    Portenta H7 (where the published results are), results/nicla/<level> for
    the Nicla Vision."""
    return os.path.join(BOARDS[board]["results_dir"], level)


def announced_board(lines):
    """Board name from the BOARD:<name> lines of the firmware, or None when the
    firmware did not print one (builds earlier than the two-board sketches)."""
    name = None
    for line in lines:
        line = line.strip()
        if line.startswith(BOARD_PREFIX):
            name = line[len(BOARD_PREFIX):].strip() or None
    return name


def key_of(announced):
    """--board key of an announced name ('NICLA_VISION' -> 'nicla'), or None."""
    for key, board in BOARDS.items():
        if board["announces"] == announced:
            return key
    return None


def mismatch(expected_key, announced):
    """None when the firmware runs on the board the operator named with
    --board (or did not say); otherwise the message to show before stopping.
    Results are filed by board, so a silent mix-up would put them in the
    wrong folder."""
    if announced is None or announced == BOARDS[expected_key]["announces"]:
        return None
    other = key_of(announced)
    hint = ("run again with --board %s" % other if other
            else "this board is not one of: %s" % ", ".join(sorted(BOARDS)))
    return ("the firmware says it runs on %s, but --board is %s (%s); %s."
            % (announced, expected_key, BOARDS[expected_key]["announces"], hint))
