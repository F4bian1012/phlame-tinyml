#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Checks the board table and the serial text shared by the PIL and HIL benches
(src/boards.py), that the two sketches print what the host expects, and the
two benches against a simulated board: a stand-in for the serial port that
answers like the firmware does. No hardware needed.

The first three test classes use the standard library only. The
simulated-board ones import the benches, so they are skipped when pyserial,
Pillow, numpy or OpenCV are not installed.

    python -m unittest discover -s tests -v
"""
import contextlib
import csv
import io
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("MPLBACKEND", "Agg")   # the benches draw the matrix; no window

import boards  # noqa: E402

BANNER = {
    "pil": "--- Done. Waiting for the next image... ---",
    "hil": "--- Done. Waiting for the next trigger... ---",
}
# what the sketches printed until November 2026
LEGACY_BANNER = {
    "pil": "--- Listo. Esperando siguiente imagen... ---",
    "hil": "--- Listo. Esperando siguiente trigger... ---",
}


def load_bench(name):
    """Imports a bench, or returns None when one of its packages is missing
    (the benches end with SystemExit in that case)."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return __import__(name)
    except (ImportError, SystemExit):
        return None


class EndOfInferenceTest(unittest.TestCase):
    def test_current_and_earlier_banners_close_an_inference(self):
        for banner in list(BANNER.values()) + list(LEGACY_BANNER.values()):
            self.assertTrue(boards.is_end_of_inference(banner), banner)

    def test_other_firmware_lines_do_not(self):
        for line in ("1", "TEMP_C:41", "CYC_TOTAL:96684532", "US_INF:241093.375", "TS_MS:120034",
                     "BOARD:NICLA_VISION", "READY_HIL", "FRAME_BEGIN:19200", "ERROR: Invoke() failed.",
                     "--- Image received. Starting inference... ---", "Setup completed successfully.", ""):
            self.assertFalse(boards.is_end_of_inference(line), line)


class BoardTableTest(unittest.TestCase):
    def test_announced_board_is_read_from_the_firmware_lines(self):
        self.assertEqual(boards.announced_board(["CAM_INIT:OK", "BOARD:NICLA_VISION", "READY_HIL"]),
                         "NICLA_VISION")
        self.assertEqual(boards.announced_board(["  BOARD: PORTENTA_H7  "]), "PORTENTA_H7")
        self.assertIsNone(boards.announced_board(["CAM_INIT:OK", "READY_HIL"]))

    def test_every_board_has_what_the_scripts_use(self):
        announced = set()
        for key, board in boards.BOARDS.items():
            for field in ("label", "announces", "camera", "core", "fqbn", "port_match", "results_dir"):
                self.assertTrue(board.get(field), "%s lacks %s" % (key, field))
            self.assertTrue(board["fqbn"].startswith(board["core"] + ":"), key)
            self.assertEqual(boards.key_of(board["announces"]), key)
            announced.add(board["announces"])
        self.assertEqual(len(announced), len(boards.BOARDS), "two boards announce the same name")
        self.assertIn(boards.DEFAULT_BOARD, boards.BOARDS)

    def test_published_results_stay_where_they_are(self):
        self.assertEqual(boards.results_dir("portenta", "pil"), os.path.join("results", "pil"))
        self.assertEqual(boards.results_dir("portenta", "hil"), os.path.join("results", "hil"))
        self.assertEqual(boards.results_dir("nicla", "pil"), os.path.join("results", "nicla", "pil"))

    def test_mismatch_only_when_the_firmware_names_another_board(self):
        self.assertIsNone(boards.mismatch("portenta", "PORTENTA_H7"))
        self.assertIsNone(boards.mismatch("nicla", None))            # earlier firmware: unknown, not wrong
        message = boards.mismatch("portenta", "NICLA_VISION")
        self.assertIn("--board nicla", message)
        self.assertIn("NICLA_VISION", boards.mismatch("nicla", "SOMETHING_ELSE") or "")
        self.assertIsNotNone(boards.mismatch("nicla", "SOMETHING_ELSE"))


SKETCHES = {
    "pil": REPO / "deployment" / "pil_firmware" / "pil_firmware.ino",
    "hil": REPO / "deployment" / "hil_firmware" / "hil_firmware.ino",
}
PRINTED = re.compile(r'Serial\.print(?:ln)?\(\s*"((?:[^"\\]|\\.)*)"')


class SketchTextTest(unittest.TestCase):
    """The sketches and the host scripts are compiled apart, so nothing but
    these checks keeps the text they exchange in step."""

    def source(self, level):
        return SKETCHES[level].read_text(encoding="utf-8")

    def test_each_sketch_closes_an_inference_with_the_banner_the_benches_wait_for(self):
        for level in SKETCHES:
            closing = [text for text in PRINTED.findall(self.source(level))
                       if boards.is_end_of_inference(text)]
            self.assertTrue(closing, "%s firmware prints no end-of-inference banner" % level)
            for text in closing:
                # only the banner may match: any other line with these words would
                # make the bench stop reading before the answer is complete
                self.assertIn(BANNER[level], text)

    def test_each_sketch_announces_the_boards_of_the_table(self):
        for level in SKETCHES:
            source = self.source(level)
            self.assertIn('Serial.print("%s");' % boards.BOARD_PREFIX, source, level)
            for board in boards.BOARDS.values():
                self.assertIn('#define BOARD_NAME "%s"' % board["announces"], source, level)

    def test_default_arena_of_the_nicla_is_the_documented_one(self):
        expected = "#define TENSOR_ARENA_KB %d" % boards.BOARDS["nicla"]["default_arena_kb"]
        for level in SKETCHES:
            nicla_block = self.source(level).split("#elif defined(ARDUINO_NICLA_VISION)")[1].split("#else")[0]
            self.assertIn(expected, nicla_block, level)


class SimulatedBoard:
    """Stand-in for serial.Serial: every packet or trigger written to it is
    answered with the lines of one inference, as the firmware prints them."""

    def __init__(self, level, board=None, banner=None, boot=()):
        self.level = level
        self.board = board
        self.banner = banner or BANNER[level]
        self.queue = [line.encode() + b"\r\n" for line in boot]
        self.closed = False
        self.inferences = 0

    def answer(self):
        self.inferences += 1
        lines = ["", "--- Image received. Starting inference... ---"]
        if self.level == "pil" and self.board:
            lines.append("BOARD:" + self.board)
        lines += ["TEMP_C:40"]
        if self.level == "hil":
            lines += ["TS_MS:%d" % (1000 * self.inferences), "CYC_CAPTURE:400000", "US_CAPTURE:1000.000"]
        lines += ["1", "CYC_PRE:4000", "CYC_INF:400000", "CYC_POST:400", "CYC_TOTAL:404400",
                  "US_PRE:10.000", "US_INF:1000.000", "US_POST:1.000", "US_TOTAL:1011.000", self.banner, ""]
        self.queue += [line.encode() + b"\r\n" for line in lines]

    # --- the part of the pyserial interface the benches use
    @property
    def in_waiting(self):
        return sum(len(chunk) for chunk in self.queue)

    def readline(self):
        return self.queue.pop(0) if self.queue else b""

    def write(self, data):
        if data[:1] in (b"#", b"T"):
            self.answer()
        return len(data)

    def reset_input_buffer(self):
        self.queue = []

    def flush(self):
        pass

    def close(self):
        self.closed = True


pil_benchmark = load_bench("pil_benchmark")


@unittest.skipIf(pil_benchmark is None, "pil_benchmark needs pyserial, Pillow and numpy")
class PilBenchOnSimulatedBoardTest(unittest.TestCase):
    def setUp(self):
        from PIL import Image
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = os.path.join(self.tmp.name, "test")
        for cls, shade in (("non_person", 40), ("person", 200)):
            os.makedirs(os.path.join(self.folder, cls))
            for i in range(2):
                Image.new("L", (16, 12), shade + i).save(os.path.join(self.folder, cls, "img_%d.png" % i))
        self.out = os.path.join(self.tmp.name, "out")

    def run_bench(self, board, board_key="portenta"):
        with mock.patch.object(pil_benchmark.serial, "Serial", return_value=board), \
                mock.patch.object(pil_benchmark.time, "sleep"), \
                contextlib.redirect_stdout(io.StringIO()) as log:
            pil_benchmark.send_folder(port="SIM", baud=115200, folder=self.folder, width=16, height=12,
                                      grayscale=True, pre_delay=0, gap=0, model_tag="sim_int8",
                                      model_path=None, output_dir=self.out, board_key=board_key)
        with open(os.path.join(self.out, "metrics_sim_int8.json"), encoding="utf-8") as fh:
            return json.load(fh), log.getvalue()

    def test_run_is_complete_and_records_the_board(self):
        metrics, _ = self.run_bench(SimulatedBoard("pil", board="NICLA_VISION"), board_key="nicla")
        self.assertEqual(metrics["level"], "PIL")
        self.assertEqual(metrics["run"]["board"], "NICLA_VISION")
        self.assertEqual((metrics["dataset"]["n_images"], metrics["dataset"]["n_answered"],
                          metrics["dataset"]["n_unanswered"]), (4, 4, 0))
        self.assertEqual(metrics["prediction_counts"], {"1": 4})
        self.assertEqual(metrics["latency"]["per_field"]["US_TOTAL"]["mean"], 1011.0)
        with open(os.path.join(self.out, "predictions_sim_int8.csv"), encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh, delimiter=";"))
        self.assertEqual(len(rows), 4)
        self.assertEqual({r["PredClass"] for r in rows}, {"person"})

    def test_firmware_built_before_the_change_still_works(self):
        board = SimulatedBoard("pil", board=None, banner=LEGACY_BANNER["pil"])
        metrics, log = self.run_bench(board)
        self.assertEqual(metrics["dataset"]["n_answered"], 4)
        self.assertIsNone(metrics["run"]["board"])
        self.assertIn("did not announce its board", log)

    def test_wrong_board_stops_before_measuring_and_leaves_nothing(self):
        board = SimulatedBoard("pil", board="NICLA_VISION")
        with mock.patch.object(pil_benchmark.serial, "Serial", return_value=board), \
                mock.patch.object(pil_benchmark.time, "sleep"), \
                contextlib.redirect_stdout(io.StringIO()) as log, \
                self.assertRaises(SystemExit) as ctx:
            pil_benchmark.send_folder(port="SIM", baud=115200, folder=self.folder, width=16, height=12,
                                      grayscale=True, pre_delay=0, gap=0, model_tag="sim_int8",
                                      model_path=None, output_dir=self.out, board_key="portenta")
        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("--board nicla", log.getvalue())
        self.assertEqual(board.inferences, 1)
        self.assertTrue(board.closed)
        self.assertEqual(os.listdir(self.out), [])


hil_benchmark = load_bench("hil_benchmark")
BOOT = ["BOARD:NICLA_VISION", "CAM_INIT:OK", "INPUT_SHAPE:120x160x1", "MODEL_BYTES:645640",
        "MODEL_FNV1A:0a1b2c3d", "ARENA_USED:356404", "READY_HIL"]


@unittest.skipIf(hil_benchmark is None, "hil_benchmark needs pyserial, OpenCV and numpy")
class HilBenchOnSimulatedBoardTest(unittest.TestCase):
    def quiet(self):
        return contextlib.redirect_stdout(io.StringIO())

    def test_handshake_returns_the_fingerprint_and_the_board(self):
        with mock.patch.object(hil_benchmark.time, "sleep"), self.quiet():
            fw, announced = hil_benchmark.wait_ready(SimulatedBoard("hil", boot=BOOT), "nicla", timeout_s=2)
        self.assertEqual(announced, "NICLA_VISION")
        self.assertEqual(fw, {"model_bytes": 645640, "model_fnv1a": "0a1b2c3d", "arena_used_bytes": 356404})

    def test_handshake_of_an_earlier_firmware_has_no_board(self):
        boot = [line for line in BOOT if not line.startswith("BOARD:")]
        with mock.patch.object(hil_benchmark.time, "sleep"), self.quiet():
            _, announced = hil_benchmark.wait_ready(SimulatedBoard("hil", boot=boot), "portenta", timeout_s=2)
        self.assertIsNone(announced)

    def test_wrong_board_stops_at_the_handshake(self):
        board = SimulatedBoard("hil", boot=BOOT)
        with mock.patch.object(hil_benchmark.time, "sleep"), self.quiet(), self.assertRaises(SystemExit) as ctx:
            hil_benchmark.wait_ready(board, "portenta", timeout_s=2)
        self.assertEqual(ctx.exception.code, 2)
        self.assertTrue(board.closed)

    def test_trigger_reads_one_inference_with_either_banner(self):
        for banner in (BANNER["hil"], LEGACY_BANNER["hil"]):
            with mock.patch.object(hil_benchmark.time, "sleep"), self.quiet():
                predicted, telemetry = hil_benchmark.trigger_inference(
                    SimulatedBoard("hil", banner=banner), timeout_s=2)
            self.assertEqual(predicted, 1, banner)
            self.assertEqual(telemetry["US_TOTAL"], 1011.0)
            self.assertEqual(telemetry["CYC_CAPTURE"], 400000)
            self.assertEqual(telemetry["TS_MS"], 1000)


if __name__ == "__main__":
    unittest.main()
