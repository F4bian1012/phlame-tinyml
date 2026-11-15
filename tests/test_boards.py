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


class IdentityLinesTest(unittest.TestCase):
    def test_fingerprint_is_the_fnv1a_of_the_file(self):
        # published test vectors of 32-bit FNV-1a
        with tempfile.TemporaryDirectory() as tmp:
            for data, expected in ((b"", "811c9dc5"), (b"a", "e40c292c"), (b"foobar", "bf9cf968")):
                path = os.path.join(tmp, "model.tflite")
                with open(path, "wb") as fh:
                    fh.write(data)
                self.assertEqual(boards.fnv1a32_file(path), expected)
                self.assertEqual(boards.fnv1a32_file(path, chunk=1), expected)

    def test_identity_is_read_from_the_firmware_lines(self):
        found = boards.identity(["--- STARTING DIAGNOSTICS ---", "BOARD:NICLA_VISION", "INPUT_SHAPE:120x160x1",
                                 "MODEL_BYTES:644648", "MODEL_FNV1A:0A1B2C3D", "ARENA_USED:356404", "READY_PIL"])
        self.assertEqual(found, {"board": "NICLA_VISION",
                                 "input_shape": {"height": 120, "width": 160, "channels": 1},
                                 "model_bytes": 644648, "model_fnv1a": "0a1b2c3d", "arena_used_bytes": 356404})

    def test_lines_a_build_does_not_print_are_none(self):
        found = boards.identity(["BOARD:PORTENTA_H7", "ARENA_USED:181492", "TEMP_C:40", "1"])
        self.assertEqual((found["board"], found["arena_used_bytes"]), ("PORTENTA_H7", 181492))
        self.assertIsNone(found["input_shape"])
        self.assertIsNone(found["model_bytes"])
        self.assertIsNone(found["model_fnv1a"])

    def test_lines_that_cannot_be_read_are_none_not_an_error(self):
        found = boards.identity(["INPUT_SHAPE:120x160", "MODEL_BYTES:many", "MODEL_FNV1A:", "ARENA_USED:"])
        self.assertEqual(found, {"board": None, "input_shape": None, "model_bytes": None,
                                 "model_fnv1a": None, "arena_used_bytes": None})


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

    def test_pil_sketch_prints_its_identity_lines_and_answers_an_empty_packet(self):
        source = self.source("pil")
        for text in ("INPUT_SHAPE:", "MODEL_BYTES:", "MODEL_FNV1A:", "ARENA_USED:"):
            self.assertIn('Serial.print("%s");' % text, source)
        self.assertIn('Serial.println("%s");' % boards.READY_PIL, source)
        self.assertIn("if (image_bytes_received == 0) {", source)
        self.assertIn('Serial.println("%s");' % boards.READY_HIL, self.source("hil"))

    def test_both_sketches_compute_the_fingerprint_the_host_computes(self):
        for level in SKETCHES:
            source = self.source(level)
            self.assertIn("uint32_t h = %du;" % boards.FNV1A32_OFFSET, source, level)
            self.assertIn("h *= %du;" % boards.FNV1A32_PRIME, source, level)
            self.assertIn('snprintf(hex, sizeof(hex), "%08lx"', source, level)

    def test_default_arena_of_the_nicla_is_the_documented_one(self):
        expected = "#define TENSOR_ARENA_KB %d" % boards.BOARDS["nicla"]["default_arena_kb"]
        for level in SKETCHES:
            nicla_block = self.source(level).split("#elif defined(ARDUINO_NICLA_VISION)")[1].split("#else")[0]
            self.assertIn(expected, nicla_block, level)


class FakeClock:
    """Stands in for the time module inside a bench: sleeping costs nothing and
    moves the clock, so timeouts expire at once."""

    def __init__(self):
        self.now = 1000.0

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += max(0.0, seconds)


def pil_identity(model_path, board="PORTENTA_H7", shape="12x16x1", fnv1a=None, model_bytes=None,
                 arena=181492):
    """The identity lines of the PIL sketch for a board that carries model_path."""
    return ["BOARD:" + board, "INPUT_SHAPE:" + shape,
            "MODEL_BYTES:%d" % (os.path.getsize(model_path) if model_bytes is None else model_bytes),
            "MODEL_FNV1A:%s" % (fnv1a or boards.fnv1a32_file(model_path)),
            "ARENA_USED:%d" % arena, boards.READY_PIL]


class SimulatedBoard:
    """Stand-in for serial.Serial: every packet or trigger written to it is
    answered with the lines of one inference, as the firmware prints them.

    For the PIL level:
      identity      the lines printed for an empty packet (#@). None is a build
                    without identity lines: it takes the empty packet for an
                    image and answers with an inference
      mute          the board answers nothing at all
      silent_on     numbers of the images (1 = first) that get no answer
      silent_from   from this image on, no image gets an answer
      raise_at, failure   the n-th image raises failure (Ctrl+C, an error)
      drop_link_at  the n-th image finds the port gone
    inferences counts the images answered, packets the images received and
    empty_packets the identity requests.
    """

    def __init__(self, level, board=None, banner=None, boot=(), drop_link_at=None, identity=None,
                 mute=False, silent_on=(), silent_from=None, raise_at=None, failure=None):
        self.level = level
        self.board = board
        self.banner = banner or BANNER[level]
        self.queue = [line.encode() + b"\r\n" for line in boot]
        self.closed = False
        self.inferences = 0
        self.packets = 0
        self.empty_packets = 0
        self.drop_link_at = drop_link_at   # the n-th packet finds the port gone
        self.identity = identity
        self.mute = mute
        self.silent_on = set(silent_on)
        self.silent_from = silent_from
        self.raise_at, self.failure = raise_at, failure

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
        if self.mute:
            return len(data)
        if self.level == "pil" and data == b"#@":
            # the host asks what is running
            self.empty_packets += 1
            if self.identity is not None:
                self.queue += [line.encode() + b"\r\n" for line in self.identity]
            else:
                answered = self.inferences
                self.answer()                 # an earlier build runs an inference on nothing
                self.inferences = answered
            return len(data)
        if data[:1] in (b"#", b"T"):
            self.packets += 1
            if self.packets == self.drop_link_at:
                import serial
                raise serial.SerialException("WriteFile failed (the device does not recognise the command)")
            if self.packets == self.raise_at:
                raise self.failure
            if self.packets in self.silent_on or (self.silent_from and self.packets >= self.silent_from):
                return len(data)
            self.answer()
        return len(data)

    def reset_input_buffer(self):
        self.queue = []

    def flush(self):
        pass

    def close(self):
        self.closed = True


pil_benchmark = load_bench("pil_benchmark")


class PilFolderCase(unittest.TestCase):
    """A small class-per-folder test partition, a model file and an output folder, for the PIL bench.

    The bench runs against a simulated board and a fake clock: run_folder() gives it
    the port (or the ports, one per time the bench opens it) and returns the
    exit code, None for a run that ended normally, and what was printed.
    """

    N_PER_CLASS = 2

    def setUp(self):
        from PIL import Image
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = os.path.join(self.tmp.name, "test")
        for cls, shade in (("non_person", 40), ("person", 200)):
            os.makedirs(os.path.join(self.folder, cls))
            for i in range(self.N_PER_CLASS):
                Image.new("L", (16, 12), shade + i).save(os.path.join(self.folder, cls, "img_%d.png" % i))
        self.total = 2 * self.N_PER_CLASS
        self.model = os.path.join(self.tmp.name, "sim_int8.tflite")
        with open(self.model, "wb") as fh:
            fh.write(bytes(range(256)) * 8)
        self.out = os.path.join(self.tmp.name, "out")
        self.clock = FakeClock()

    def board(self, name="PORTENTA_H7", **kwargs):
        """A board flashed with the current sketch and with self.model."""
        return SimulatedBoard("pil", board=name, identity=pil_identity(self.model, board=name), **kwargs)

    def run_folder(self, ports, reopen=None, **options):
        ports = ports if isinstance(ports, list) else [ports]
        arguments = dict(port="SIM", baud=115200, folder=self.folder, width=16, height=12, grayscale=True,
                         pre_delay=0, gap=0, model_tag="sim_int8", model_path=self.model,
                         output_dir=self.out, board_key="portenta")
        arguments.update(options)
        patches = [mock.patch.object(pil_benchmark.serial, "Serial", side_effect=ports),
                   mock.patch.object(pil_benchmark, "time", self.clock)]
        if reopen is not None:
            patches.append(mock.patch.object(pil_benchmark, "reopen_port", reopen))
        code = None
        with contextlib.ExitStack() as stack, contextlib.redirect_stdout(io.StringIO()) as log:
            for patch in patches:
                stack.enter_context(patch)
            try:
                pil_benchmark.send_folder(**arguments)
            except SystemExit as stop:
                code = stop.code
        return code, log.getvalue()

    def metrics(self):
        with open(os.path.join(self.out, "metrics_sim_int8.json"), encoding="utf-8") as fh:
            return json.load(fh)

    def predictions(self):
        with open(os.path.join(self.out, "predictions_sim_int8.csv"), encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh, delimiter=";"))

    def written(self):
        return sorted(os.listdir(self.out)) if os.path.isdir(self.out) else []


@unittest.skipIf(pil_benchmark is None, "pil_benchmark needs pyserial, Pillow and numpy")
class PilBenchOnSimulatedBoardTest(PilFolderCase):
    def test_run_is_complete_and_records_the_board(self):
        code, log = self.run_folder(self.board("NICLA_VISION"), board_key="nicla")
        self.assertIsNone(code, log)
        metrics = self.metrics()
        self.assertEqual(metrics["level"], "PIL")
        self.assertEqual(metrics["run"]["board"], "NICLA_VISION")
        self.assertEqual((metrics["dataset"]["n_images"], metrics["dataset"]["n_answered"],
                          metrics["dataset"]["n_unanswered"]), (4, 4, 0))
        self.assertEqual(metrics["prediction_counts"], {"1": 4})
        self.assertEqual(metrics["latency"]["per_field"]["US_TOTAL"]["mean"], 1011.0)
        rows = self.predictions()
        self.assertEqual(len(rows), 4)
        self.assertEqual({r["PredClass"] for r in rows}, {"person"})

    def test_firmware_built_before_november_2026_runs_only_when_allowed(self):
        # no identity lines, no BOARD line, the Spanish banner
        board = SimulatedBoard("pil", board=None, banner=LEGACY_BANNER["pil"])
        code, log = self.run_folder(board)
        self.assertEqual(code, 2)
        self.assertIn("MODEL_FNV1A", log)
        self.assertEqual((board.inferences, self.written()), (0, []))

        board = SimulatedBoard("pil", board=None, banner=LEGACY_BANNER["pil"])
        code, log = self.run_folder(board, allow_model_mismatch=True)
        self.assertIsNone(code, log)
        metrics = self.metrics()
        self.assertEqual(metrics["dataset"]["n_answered"], 4)
        self.assertIsNone(metrics["run"]["board"])
        self.assertFalse(metrics["model"]["identity_verified"])
        self.assertIn("did not announce its board", log)

    def test_wrong_board_stops_before_measuring_and_leaves_nothing(self):
        board = self.board("NICLA_VISION")
        code, log = self.run_folder(board, board_key="portenta")
        self.assertEqual(code, 2)
        self.assertIn("--board nicla", log)
        self.assertEqual(board.inferences, 0)
        self.assertTrue(board.closed)
        self.assertEqual(self.written(), [])

    def test_wrong_board_is_also_found_out_on_a_build_without_identity_lines(self):
        board = SimulatedBoard("pil", board="NICLA_VISION")     # names its board with every answer
        code, log = self.run_folder(board, board_key="portenta", allow_model_mismatch=True)
        self.assertEqual(code, 2)
        self.assertIn("--board nicla", log)
        self.assertEqual((board.inferences, self.written()), (0, []))


@unittest.skipIf(pil_benchmark is None, "pil_benchmark needs pyserial, Pillow and numpy")
class PilBenchOnDroppedLinkTest(PilFolderCase):
    """A USB serial link can vanish for a moment while the board keeps running."""

    def test_port_is_reopened_and_the_same_image_sent_again(self):
        first = self.board(drop_link_at=3)
        second = self.board()
        code, log = self.run_folder([first, second])
        self.assertIsNone(code, log)
        metrics = self.metrics()
        self.assertEqual(metrics["run"]["n_reconnects"], 1)
        self.assertTrue(metrics["run"]["completed"])
        self.assertEqual((metrics["dataset"]["n_images"], metrics["dataset"]["n_answered"],
                          metrics["dataset"]["n_unanswered"]), (4, 4, 0))
        self.assertEqual((first.inferences, second.inferences), (2, 2))   # no image lost, none counted twice
        self.assertEqual(second.empty_packets, 0)                         # the board is not asked again
        self.assertTrue(first.closed)
        self.assertIn("Serial link lost", log)
        self.assertEqual(len(self.predictions()), 4)

    def test_run_without_drops_records_zero_reconnects(self):
        code, log = self.run_folder(self.board())
        self.assertIsNone(code, log)
        self.assertEqual(self.metrics()["run"]["n_reconnects"], 0)

    def test_link_that_does_not_come_back_ends_the_run_and_keeps_what_was_measured(self):
        first = self.board(drop_link_at=2)
        code, log = self.run_folder(first, reopen=lambda port, baud: None)
        self.assertEqual(code, 3)
        self.assertIn("did NOT finish", log)
        self.assertEqual(len(self.predictions()), 1)   # the image measured before the drop
        metrics = self.metrics()
        self.assertFalse(metrics["run"]["completed"])
        self.assertIn("serial link lost at image 2 of 4 and did not recover", metrics["run"]["stopped_because"])
        self.assertEqual((metrics["dataset"]["n_images_planned"], metrics["dataset"]["n_images"],
                          metrics["dataset"]["n_answered"], metrics["run"]["n_reconnects"]), (4, 1, 1, 1))


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
