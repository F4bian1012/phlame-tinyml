#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
What the host scripts need to know about the boards and their firmware.

There is one sketch per ladder level (deployment/pil_firmware and
deployment/hil_firmware) and each serves every board listed here: the sketch
picks its board block at compile time and says which board it runs on with a
BOARD:<name> line over serial. This module holds the other half: how to
compile for each board, where its results go, and the text lines the two
benches wait for.

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
