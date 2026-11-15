#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Checks a whole session of the HIL bench (src/hil_benchmark.py) against a
simulated board and a simulated stimulus window. No hardware and no window.

What is checked is what a physical session cannot afford to get wrong:
  - every row is on disk as soon as its stimulus is measured, and the other
    files are written however the sweep ends (lost serial link, a board that
    stops answering, Ctrl+C, an error), marked as not completed;
  - a key pressed in the stimulus window does not shorten the settle time;
  - no stimulus is shown before the model on the board has been verified,
    whatever names the output files;
  - --monitor places the window and is recorded;
  - the frames of a session go to a folder of its own, which the conditions
    file names, and are not replaced by another session unless --overwrite or
    an explicit --frames_dir says so;
  - the keys the bench writes are the ones docs/results_schema.md documents.

The board is the SimulatedBoard of test_boards.py with the boot lines, the
frame dump and the failures a session can meet. Time is a fake clock: waiting
costs nothing, and a board that never answers does not slow the tests down.

Skipped when pyserial, OpenCV, numpy, scikit-learn or seaborn are missing.

    python -m unittest discover -s tests -v
"""
import contextlib
import csv
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("MPLBACKEND", "Agg")   # the bench draws the matrix; no window

import test_results_schema as schema  # noqa: E402
from test_boards import SimulatedBoard, load_bench  # noqa: E402

hil_benchmark = load_bench("hil_benchmark")
READY = (hil_benchmark is not None
         and all(importlib.util.find_spec(name) for name in ("sklearn", "seaborn", "matplotlib")))
FRAME_W, FRAME_H = 160, 120


class FakeClock:
    """Stands in for the time module inside the bench: sleeping costs nothing
    and moves the clock, so timeouts expire at once."""

    def __init__(self):
        self.now = 1000.0

    def time(self):
        return self.now

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += max(0.0, seconds)


def escape(payload):
    """The #...@ packet of the firmware: '#', '@' and ESC inside it are sent as ESC, byte ^ 0x20."""
    out = bytearray(b"#")
    for b in payload:
        if b in (0x23, 0x40, 0x1B):
            out += bytes([0x1B, b ^ 0x20])
        else:
            out.append(b)
    return bytes(out + b"@")


class RigBoard(SimulatedBoard):
    """The simulated board with what a whole HIL session needs: the boot lines
    with the fingerprint of a model file, the frame dump, and the ways a
    session breaks.

      fail_at, failure   the n-th trigger raises failure (a lost link, Ctrl+C...)
      silent_from        from the n-th trigger on, the board does not answer
      on_trigger         called with n before the n-th trigger is answered
      camera             returns the frame (120x160 uint8) the camera "took";
                         used when the bench asks for the frame dump
    """

    def __init__(self, model_path, clock, board="PORTENTA_H7", fingerprint=True, fnv1a=None,
                 fail_at=None, failure=None, silent_from=None, on_trigger=None, camera=None):
        data = Path(model_path).read_bytes()
        boot = ["BOARD:" + board, "CAM_INIT:OK", "INPUT_SHAPE:120x160x1"]
        if fingerprint:
            boot += ["MODEL_BYTES:%d" % len(data),
                     "MODEL_FNV1A:%s" % (fnv1a or "%08x" % hil_benchmark.fnv1a32_bytes(data))]
        boot += ["ARENA_USED:356404", "READY_HIL"]
        super().__init__("hil", boot=boot)
        self.clock = clock
        self.timeout = 10
        self.dump = False
        self.triggers = 0
        self.fail_at, self.failure = fail_at, failure
        self.silent_from = silent_from
        self.on_trigger = on_trigger
        self.camera = camera

    def write(self, data):
        if data == hil_benchmark.CMD_DUMP_ON:
            self.dump = True
            self.queue.append(b"FRAME_DUMP:ON\r\n")
        elif data == hil_benchmark.CMD_DUMP_OFF:
            self.dump = False
            self.queue.append(b"FRAME_DUMP:OFF\r\n")
        elif data == hil_benchmark.CMD_TRIGGER:
            self.triggers += 1
            if self.on_trigger:
                self.on_trigger(self.triggers)
            if self.fail_at == self.triggers:
                raise self.failure
            if self.silent_from and self.triggers >= self.silent_from:
                return len(data)
            self.answer()
            if self.dump and self.camera is not None:
                frame = self.camera().astype("uint8").tobytes()
                self.queue += [b"FRAME_BEGIN:%d\r\n" % len(frame), escape(frame), b"FRAME_END\r\n"]
        return len(data)

    def read(self, _size=1):
        if not self.queue:
            self.clock.sleep(self.timeout)     # a real port blocks for its timeout
            return b""
        return self.queue.pop(0)


@contextlib.contextmanager
def simulated_rig(bench, board, clock, shown=None, wait_key=None):
    """Runs the bench without a port, a window or a wait: the serial port is
    board, time is clock, and the window calls of OpenCV do nothing. shown
    (a list) receives every picture the stimulus window was asked to show."""

    def default_wait_key(ms):
        clock.sleep(ms / 1000.0)
        return -1

    cv2 = bench.cv2
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(bench.serial, "Serial", return_value=board))
        stack.enter_context(mock.patch.object(bench, "time", clock))
        for name in ("namedWindow", "setWindowProperty", "resizeWindow", "destroyAllWindows"):
            stack.enter_context(mock.patch.object(cv2, name))
        move = stack.enter_context(mock.patch.object(cv2, "moveWindow"))
        stack.enter_context(mock.patch.object(cv2, "getWindowImageRect", return_value=(0, 0, 640, 480)))
        stack.enter_context(mock.patch.object(
            cv2, "imshow", side_effect=lambda _name, picture: shown.append(picture) if shown is not None else None))
        stack.enter_context(mock.patch.object(cv2, "waitKey", side_effect=wait_key or default_wait_key))
        yield move


class HilSessionCase(unittest.TestCase):
    """A small class-per-folder set of stimuli, a model file and an output folder."""

    N_PER_CLASS = 2

    def setUp(self):
        import numpy as np
        self.np = np
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.folder = str(root / "stimuli")
        rng = np.random.default_rng(0)
        for cls in ("non_person", "person"):
            os.makedirs(os.path.join(self.folder, cls))
            for i in range(self.N_PER_CLASS):
                picture = rng.integers(0, 256, size=(FRAME_H, FRAME_W), dtype=np.uint8)
                hil_benchmark.cv2.imwrite(os.path.join(self.folder, cls, "img_%d.png" % i), picture)
        self.total = 2 * self.N_PER_CLASS
        self.model = str(root / "Tiny_sim+1+1+0.1+160+120_int8.tflite")
        Path(self.model).write_bytes(bytes(range(256)) * 8)
        self.tag = "Tiny_sim+1+1+0.1+160+120_int8"
        self.out = str(root / "out")
        self.clock = FakeClock()
        # Where a session without --frames_dir puts its frames: here, not in data/hil_frames of the repository.
        self.frames_root = str(root / "hil_frames")
        frames_root = mock.patch.object(hil_benchmark, "FRAMES_ROOT", self.frames_root)
        frames_root.start()
        self.addCleanup(frames_root.stop)

    def board(self, **kwargs):
        return RigBoard(self.model, self.clock, **kwargs)

    def argv(self, *extra, model=True, confirm=False):
        args = ["hil_benchmark.py", "--port", "SIM", "--folder", self.folder,
                "--lux", "10", "--distance_cm", "24", "--ambient_temp", "24", "--notes", "simulated rig",
                "--output_dir", self.out, "--delay", "0", "--gap", "0", "--settle", "0.2"]
        if model:
            args += ["--model_path", self.model]
        if not confirm:
            args.append("--no_confirm")
        return args + list(extra)

    def run_bench(self, board, *extra, shown=None, wait_key=None, **argv_options):
        """Runs a session. Returns (exit code or None, what was printed, the moveWindow mock)."""
        code = None
        with mock.patch.object(sys, "argv", self.argv(*extra, **argv_options)), \
                contextlib.redirect_stdout(io.StringIO()) as log, \
                contextlib.redirect_stderr(io.StringIO()), \
                simulated_rig(hil_benchmark, board, self.clock, shown=shown, wait_key=wait_key) as move:
            try:
                hil_benchmark.run_benchmark(hil_benchmark.parse_args())
            except SystemExit as stop:
                code = stop.code
        return code, log.getvalue(), move

    def path(self, kind, ext, tag=None):
        return os.path.join(self.out, "%s_%s.%s" % (kind, tag or self.tag, ext))

    def load(self, kind, tag=None):
        with open(self.path(kind, "json", tag), encoding="utf-8") as fh:
            return json.load(fh)

    def rows(self, tag=None):
        with open(self.path("latency_metrics", "csv", tag), encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))


@unittest.skipUnless(READY, "the HIL bench needs pyserial, OpenCV, numpy, scikit-learn and seaborn")
class CompleteSessionTest(HilSessionCase):
    def test_complete_run_writes_every_file_and_says_it_is_complete(self):
        board = self.board()
        code, log, _ = self.run_bench(board)
        self.assertIsNone(code, log)
        self.assertEqual(sorted(os.listdir(self.out)), sorted(
            "%s_%s.%s" % (kind, self.tag, ext) for kind, ext in (
                ("confusion_matrix", "png"), ("conditions", "json"), ("latency_metrics", "csv"),
                ("metrics", "json"))))
        conditions, metrics = self.load("conditions"), self.load("metrics")
        self.assertEqual((conditions["n_stimuli_planned"], conditions["n_stimuli"], conditions["completed"],
                          conditions["stopped_because"]), (self.total, self.total, True, None))
        self.assertIsNone(conditions["stimulus_monitor"])
        self.assertTrue(conditions["model_identity_verified"])
        self.assertEqual((metrics["run"]["completed"], metrics["dataset"]["n_stimuli_planned"],
                          metrics["dataset"]["n_stimuli"], metrics["dataset"]["n_answered"]),
                         (True, self.total, self.total, self.total))
        self.assertEqual(len(self.rows()), self.total)
        self.assertEqual(board.triggers, self.total)
        self.assertTrue(board.closed)
        self.assertNotIn("RUN NOT COMPLETED", log)

    def test_every_row_is_on_disk_before_the_next_stimulus_is_triggered(self):
        on_disk = []

        def count_rows(_n):
            path = self.path("latency_metrics", "csv")
            on_disk.append(len(self.rows()) if os.path.exists(path) else None)

        self.run_bench(self.board(on_trigger=count_rows))
        self.assertEqual(on_disk, list(range(self.total)))

    def test_key_pressed_during_a_session_is_reported_and_changes_nothing(self):
        pressed = iter([ord("q")])

        def wait_key(ms):
            key = next(pressed, None)
            if key is not None:
                self.clock.sleep(0.05)
                return key
            self.clock.sleep(ms / 1000.0)
            return -1

        board = self.board()
        code, log, _ = self.run_bench(board, wait_key=wait_key)
        self.assertIsNone(code, log)
        self.assertEqual(board.triggers, self.total)
        self.assertTrue(self.load("conditions")["completed"])
        self.assertEqual(log.count("key press(es) in the stimulus window ignored"), 1)

    def test_frame_dump_saves_the_frames_the_camera_took(self):
        np = self.np
        frame = (np.arange(FRAME_W * FRAME_H, dtype=np.uint32) % 251).astype(np.uint8).reshape(FRAME_H, FRAME_W)
        frame[0, :3] = (0x23, 0x40, 0x1B)       # the three bytes the packet escapes
        frames_dir = os.path.join(self.tmp.name, "frames")
        code, log, _ = self.run_bench(self.board(camera=lambda: frame), "--dump_frames", "--frames_dir", frames_dir)
        self.assertIsNone(code, log)
        rows = self.rows()
        self.assertEqual(len(os.listdir(frames_dir)), self.total)
        for row in rows:
            saved = hil_benchmark.cv2.imread(os.path.join(frames_dir, row["frame_file"]),
                                             hil_benchmark.cv2.IMREAD_UNCHANGED)
            self.assertTrue(np.array_equal(saved, frame), row["frame_file"])
        self.assertTrue(self.load("conditions")["frame_dump"])

    def test_the_files_have_the_documented_keys_and_no_others(self):
        with mock.patch.object(hil_benchmark, "list_monitors", return_value=[[0, 0, 1920, 1080]]), \
                mock.patch.object(hil_benchmark, "_make_process_dpi_aware"):
            code, log, _ = self.run_bench(self.board(), "--monitor", "0")
        self.assertIsNone(code, log)
        documented_metrics, documented_conditions = schema.documented_paths()
        for kind, documented, not_yet in (("metrics", documented_metrics, schema.NOT_YET_PUBLISHED_METRICS),
                                          ("conditions", documented_conditions,
                                           schema.NOT_YET_PUBLISHED_CONDITIONS)):
            written = set()
            schema.collect_paths(self.load(kind), "", written)
            self.assertEqual(sorted(written - documented), [], "%s keys missing from the docs" % kind)
            # the keys no published file has yet are documented from what the bench writes
            self.assertEqual(sorted(not_yet - written), [], "%s keys the bench does not write" % kind)


@unittest.skipUnless(READY, "the HIL bench needs pyserial, OpenCV, numpy, scikit-learn and seaborn")
class FramesFolderTest(HilSessionCase):
    """Where the dumped frames of a session go. The frame files are named after the stimulus
    only, so in one shared folder every session replaced the frames of the one before."""

    def camera(self, level):
        """A camera whose every frame is one gray level, to tell the frames of two sessions apart."""
        return lambda: self.np.full((FRAME_H, FRAME_W), level, dtype=self.np.uint8)

    def gray_levels(self, folder):
        """{frame file: the gray levels it holds} for every file of a folder."""
        cv2 = hil_benchmark.cv2
        return {name: self.np.unique(cv2.imread(os.path.join(folder, name), cv2.IMREAD_UNCHANGED)).tolist()
                for name in sorted(os.listdir(folder))}

    def assert_recorded(self, folder, tag=None):
        self.assertEqual(os.path.normpath(self.load("conditions", tag)["frames_dir"]), os.path.normpath(folder))

    def two_sessions(self, *extra):
        """Runs a session of each of two models, one after the other, into the same output folder,
        as a campaign does. Returns ((tag, gray level of its camera), ...)."""
        other_tag = "Small_sim+1+1+0.1+160+120_int8"
        other = os.path.join(self.tmp.name, other_tag + ".tflite")
        Path(other).write_bytes(bytes(reversed(range(256))) * 8)
        sessions = ((self.tag, self.model, 10), (other_tag, other, 200))
        for tag, model, level in sessions:
            board = RigBoard(model, self.clock, camera=self.camera(level))
            code, log, _ = self.run_bench(board, "--dump_frames", "--model_path", model, *extra)
            self.assertIsNone(code, log)
        return tuple((tag, level) for tag, _model, level in sessions)

    def test_the_default_is_a_folder_per_session_under_data_hil_frames(self):
        frames_root = os.path.join("data", "hil_frames")
        with mock.patch.object(hil_benchmark, "FRAMES_ROOT", frames_root):     # the value the module defines
            self.assertEqual(hil_benchmark.session_frames_dir(None, self.tag), os.path.join(frames_root, self.tag))
            self.assertEqual(hil_benchmark.session_frames_dir("elsewhere", self.tag), "elsewhere")
        text = (REPO / "src" / "hil_benchmark.py").read_text(encoding="utf-8")
        self.assertIn("FRAMES_ROOT = os.path.join('data', 'hil_frames')", text)
        # the conditions file names that folder relative to the repository, with '/'
        inside = os.path.join(hil_benchmark.REPO_ROOT, frames_root, self.tag)
        self.assertEqual(hil_benchmark._relpath_repo(inside), "data/hil_frames/" + self.tag)

    def test_the_folder_of_a_session_is_named_like_its_output_files(self):
        code, log, _ = self.run_bench(self.board(camera=self.camera(10)), "--dump_frames")
        self.assertIsNone(code, log)
        folder = os.path.join(self.frames_root, self.tag)
        self.assertEqual(os.listdir(self.frames_root), [self.tag])
        self.assertEqual(sorted(os.listdir(folder)), sorted(row["frame_file"] for row in self.rows()))
        self.assertEqual(len(os.listdir(folder)), self.total)
        self.assert_recorded(folder)
        self.assertIn("saving captured frames to: %s" % folder, log)

    def test_two_sessions_keep_their_frames_apart(self):
        sessions = self.two_sessions()
        self.assertEqual(sorted(os.listdir(self.frames_root)), sorted(tag for tag, _ in sessions))
        for tag, level in sessions:
            folder = os.path.join(self.frames_root, tag)
            frames = self.gray_levels(folder)
            # every frame of the session is there, and it is the one its own camera took
            self.assertEqual(sorted(frames), sorted(row["frame_file"] for row in self.rows(tag)))
            self.assertEqual(len(frames), self.total)
            self.assertEqual(set(map(tuple, frames.values())), {(level,)})
            self.assert_recorded(folder, tag)

    def test_frames_of_an_earlier_session_with_the_same_name_are_not_replaced_silently(self):
        code, log, _ = self.run_bench(self.board(camera=self.camera(10)), "--dump_frames")
        self.assertIsNone(code, log)
        folder = os.path.join(self.frames_root, self.tag)
        first = self.gray_levels(folder)
        # The same model again with its results in another folder: only the frames would collide.
        again = os.path.join(self.tmp.name, "out_again")
        board = self.board(camera=self.camera(200))
        code, log, _ = self.run_bench(board, "--dump_frames", "--output_dir", again)
        self.assertEqual(code, 1)
        self.assertIn("already holds %d frame(s) of an earlier session named %s" % (self.total, self.tag), log)
        self.assertIn("--frames_dir", log)
        self.assertEqual(board.triggers, 0)
        self.assertEqual(board.queue[-1], b"READY_HIL\r\n")      # the boot lines were never read
        self.assertFalse(os.path.exists(again))
        self.assertEqual(self.gray_levels(folder), first)
        # --overwrite is the way to replace them, and the bench says that it does
        code, log, _ = self.run_bench(self.board(camera=self.camera(200)), "--dump_frames", "--output_dir", again,
                                      "--overwrite")
        self.assertIsNone(code, log)
        self.assertIn("[WARN] --overwrite: %s holds %d frame(s) of an earlier session" % (folder, self.total), log)
        self.assertEqual(sorted(self.gray_levels(folder)), sorted(first))
        self.assertEqual(set(map(tuple, self.gray_levels(folder).values())), {(200,)})

    def test_a_folder_given_with_the_option_is_used_as_it_is(self):
        shared = os.path.join(self.tmp.name, "my_frames")
        (first, _), (second, level) = self.two_sessions("--frames_dir", shared)
        self.assert_recorded(shared, first)
        self.assert_recorded(shared, second)
        # No folder per session inside it and no refusal: it works as it did when every session
        # wrote to one folder, and the second session has replaced the frames of the first.
        frames = self.gray_levels(shared)
        self.assertEqual(sorted(frames), sorted(row["frame_file"] for row in self.rows(second)))
        self.assertEqual(set(map(tuple, frames.values())), {(level,)})
        self.assertFalse(os.path.exists(self.frames_root))

    def test_a_session_without_frame_dump_names_no_folder_and_creates_none(self):
        code, log, _ = self.run_bench(self.board())
        self.assertIsNone(code, log)
        conditions = self.load("conditions")
        self.assertEqual((conditions["frame_dump"], conditions["frames_dir"]), (False, None))
        self.assertEqual({row["frame_file"] for row in self.rows()}, {""})
        self.assertFalse(os.path.exists(self.frames_root))


@unittest.skipUnless(READY, "the HIL bench needs pyserial, OpenCV, numpy, scikit-learn and seaborn")
class InterruptedSessionTest(HilSessionCase):
    def assert_incomplete(self, code, measured, reason):
        self.assertEqual(code, hil_benchmark.EXIT_INCOMPLETE)
        conditions = self.load("conditions")
        self.assertEqual((conditions["n_stimuli_planned"], conditions["n_stimuli"], conditions["completed"]),
                         (self.total, measured, False))
        self.assertIn(reason, conditions["stopped_because"])
        self.assertEqual(len(self.rows()), measured)
        return conditions

    def test_lost_serial_link_keeps_what_was_measured_and_marks_the_run(self):
        import serial
        board = self.board(fail_at=3, failure=serial.SerialException("WriteFile failed"))
        code, log, _ = self.run_bench(board)
        conditions = self.assert_incomplete(code, 2, "serial link lost at stimulus 3 of 4")
        self.assertIn("WriteFile failed", conditions["stopped_because"])
        metrics = self.load("metrics")
        self.assertEqual((metrics["run"]["completed"], metrics["dataset"]["n_stimuli_planned"],
                          metrics["dataset"]["n_stimuli"]), (False, self.total, 2))
        self.assertTrue(os.path.exists(self.path("confusion_matrix", "png")))
        self.assertTrue(board.closed)
        self.assertIn("RUN NOT COMPLETED: 2 of 4 stimuli measured", log)

    def test_link_lost_on_the_first_stimulus_leaves_the_conditions_and_an_empty_table(self):
        import serial
        code, _, _ = self.run_bench(self.board(fail_at=1, failure=serial.SerialException("gone")))
        self.assert_incomplete(code, 0, "serial link lost at stimulus 1 of 4")
        self.assertFalse(os.path.exists(self.path("metrics", "json")))      # nothing to compute metrics from

    def test_ctrl_c_ends_the_run_with_what_was_measured(self):
        code, _, _ = self.run_bench(self.board(fail_at=4, failure=KeyboardInterrupt()))
        self.assert_incomplete(code, 3, "interrupted by the operator")
        self.assertFalse(self.load("metrics")["run"]["completed"])

    def test_any_other_error_in_the_middle_still_writes_the_files(self):
        code, _, _ = self.run_bench(self.board(fail_at=2, failure=ValueError("unexpected")))
        self.assert_incomplete(code, 1, "ValueError at stimulus 2 of 4: unexpected")

    def test_stimulus_that_cannot_be_shown_makes_the_run_incomplete(self):
        Path(self.folder, "person", "broken.png").write_text("not an image", encoding="utf-8")
        self.total += 1
        board = self.board()
        code, log, _ = self.run_bench(board)
        self.assert_incomplete(code, self.total - 1, "1 of 5 stimulus images could not be displayed")
        self.assertEqual(board.triggers, self.total - 1)     # the others were all measured
        self.assertIn("Could not load", log)


@unittest.skipUnless(READY, "the HIL bench needs pyserial, OpenCV, numpy, scikit-learn and seaborn")
class SilentBoardTest(HilSessionCase):
    N_PER_CLASS = 5

    def test_board_that_stops_answering_ends_the_sweep_early(self):
        board = self.board(silent_from=3)
        code, _, _ = self.run_bench(board)
        limit = hil_benchmark.MAX_CONSECUTIVE_UNANSWERED
        self.assertEqual(code, hil_benchmark.EXIT_INCOMPLETE)
        self.assertEqual(board.triggers, 2 + limit)          # not the ten planned
        conditions = self.load("conditions")
        self.assertEqual((conditions["n_stimuli_planned"], conditions["n_stimuli"]), (10, 2 + limit))
        self.assertIn("stopped answering", conditions["stopped_because"])
        rows = self.rows()
        self.assertEqual([r["pred_label"] for r in rows], ["1", "1"] + [""] * limit)
        dataset = self.load("metrics")["dataset"]
        self.assertEqual((dataset["n_answered"], dataset["n_unanswered"]), (2, limit))

    def test_isolated_unanswered_stimuli_do_not_end_the_sweep(self):
        answers = iter([True, False, True, False, False, True, True, False, True, True])
        board = self.board()
        real_answer = board.answer
        board.answer = lambda: real_answer() if next(answers) else None
        code, log, _ = self.run_bench(board)
        self.assertIsNone(code, log)
        conditions, dataset = self.load("conditions"), self.load("metrics")["dataset"]
        self.assertTrue(conditions["completed"])
        self.assertEqual((dataset["n_stimuli"], dataset["n_answered"], dataset["n_unanswered"]), (10, 6, 4))


@unittest.skipUnless(READY, "the HIL bench needs pyserial, OpenCV, numpy, scikit-learn and seaborn")
class SettleTimeTest(unittest.TestCase):
    def wait(self, settle_s, wait_key, clock):
        with mock.patch.object(hil_benchmark, "time", clock), \
                mock.patch.object(hil_benchmark.cv2, "waitKey", side_effect=wait_key):
            start = clock.now
            keys = hil_benchmark.wait_settle(settle_s)
        return keys, clock.now - start

    def test_key_presses_do_not_shorten_the_settle_time(self):
        clock, asked = FakeClock(), []

        def wait_key(ms):
            asked.append(ms)
            if len(asked) <= 2:              # a key arrives 100 ms into each of the first two waits
                clock.sleep(0.1)
                return ord("x")
            clock.sleep(ms / 1000.0)
            return -1

        keys, waited = self.wait(1.5, wait_key, clock)
        # One cv2.waitKey(1500), which is what the bench did, would have come
        # back after the first 100 ms and the trigger would have gone out then.
        self.assertEqual(keys, 2)
        self.assertEqual(asked, [1500, 1400, 1300])
        self.assertAlmostEqual(waited, 1.5, places=6)

    def test_without_keys_the_window_is_drawn_once_for_the_whole_time(self):
        clock, asked = FakeClock(), []

        def wait_key(ms):
            asked.append(ms)
            clock.sleep(ms / 1000.0)
            return -1

        keys, waited = self.wait(1.5, wait_key, clock)
        self.assertEqual((keys, asked), (0, [1500]))
        self.assertAlmostEqual(waited, 1.5, places=6)

    def test_zero_settle_time_still_draws_the_window(self):
        clock, asked = FakeClock(), []

        def wait_key(ms):
            asked.append(ms)
            clock.sleep(ms / 1000.0)
            return -1

        self.wait(0, wait_key, clock)
        self.assertEqual(asked, [1])


@unittest.skipUnless(READY, "the HIL bench needs pyserial, OpenCV, numpy, scikit-learn and seaborn")
class ModelIdentityTest(HilSessionCase):
    def assert_nothing_measured(self, board):
        self.assertEqual(board.triggers, 0)
        self.assertFalse(os.path.exists(self.out) and os.listdir(self.out))

    def test_model_tag_alone_is_refused_before_the_port_is_opened(self):
        board = self.board()
        code, log, _ = self.run_bench(board, "--model_tag", "my_run", model=False)
        self.assertEqual(code, 2)
        self.assertIn("--model_path", log)
        self.assert_nothing_measured(board)
        self.assertEqual(board.queue[-1], b"READY_HIL\r\n")    # the boot lines were never read

    def test_model_tag_names_the_files_and_the_model_is_still_verified(self):
        code, log, _ = self.run_bench(self.board(), "--model_tag", "my_run")
        self.assertIsNone(code, log)
        conditions = self.load("conditions", "my_run")
        self.assertEqual(conditions["model_tag"], "my_run")
        self.assertTrue(conditions["model_identity_verified"])
        self.assertEqual(self.load("metrics", "my_run")["model"]["name"], "my_run")
        self.assertIn("IDENTITY VERIFIED", log)

    def test_another_model_on_the_board_stops_before_the_first_stimulus(self):
        board = self.board(fnv1a="deadbeef")
        code, log, _ = self.run_bench(board)
        self.assertEqual(code, 2)
        self.assertIn("MODEL MISMATCH", log)
        self.assert_nothing_measured(board)

    def test_firmware_without_a_fingerprint_is_refused(self):
        board = self.board(fingerprint=False)
        code, log, _ = self.run_bench(board)
        self.assertEqual(code, 2)
        self.assertIn("did NOT report MODEL_FNV1A", log)
        self.assert_nothing_measured(board)

    def test_the_debugging_switch_measures_and_records_that_nothing_was_verified(self):
        for kwargs, extra, model in (({"fingerprint": False}, (), True),
                                     ({"fnv1a": "deadbeef"}, (), True),
                                     ({}, ("--model_tag", "untested"), False)):
            with self.subTest(board=kwargs, extra=extra):
                code, log, _ = self.run_bench(self.board(**kwargs), "--allow_model_mismatch", "--overwrite",
                                              *extra, model=model)
                self.assertIsNone(code, log)
                tag = "untested" if extra else None
                self.assertFalse(self.load("conditions", tag)["model_identity_verified"])
                self.assertFalse(self.load("metrics", tag)["model"]["identity_verified"])


@unittest.skipUnless(READY, "the HIL bench needs pyserial, OpenCV, numpy, scikit-learn and seaborn")
class StimulusWindowTest(HilSessionCase):
    MONITORS = [[0, 0, 1920, 1080], [1920, 0, 1280, 1024]]

    def run_with_monitors(self, board, *extra):
        with mock.patch.object(hil_benchmark, "list_monitors", return_value=self.MONITORS), \
                mock.patch.object(hil_benchmark, "_make_process_dpi_aware") as dpi:
            code, log, move = self.run_bench(board, *extra)
        return code, log, move, dpi

    def test_monitor_option_places_the_window_and_is_recorded(self):
        code, log, move, dpi = self.run_with_monitors(self.board(), "--monitor", "1")
        self.assertIsNone(code, log)
        move.assert_called_once_with(hil_benchmark.WINDOW_NAME, 1930, 10)
        self.assertTrue(dpi.called)
        self.assertEqual(self.load("conditions")["stimulus_monitor"],
                         {"index": 1, "geometry": [1920, 0, 1280, 1024], "monitors": self.MONITORS})

    def test_without_the_option_the_window_is_left_where_it_opens(self):
        code, log, move, dpi = self.run_with_monitors(self.board())
        self.assertIsNone(code, log)
        self.assertFalse(move.called)
        self.assertFalse(dpi.called)         # nothing changes for a session without --monitor
        self.assertIsNone(self.load("conditions")["stimulus_monitor"])

    def test_monitor_that_does_not_exist_stops_before_the_port_is_opened(self):
        board = self.board()
        code, log, _, _ = self.run_with_monitors(board, "--monitor", "2")
        self.assertEqual(code, 2)
        self.assertIn("only 2 monitor(s) found", log)
        self.assertEqual(board.queue[-1], b"READY_HIL\r\n")      # the boot lines were never read
        self.assertFalse(os.path.exists(self.out))

    def test_stimulus_is_shown_enlarged_without_smoothing(self):
        shown = []
        self.run_bench(self.board(), shown=shown)
        self.assertEqual(len(shown), self.total)
        files = sorted(Path(self.folder).rglob("*.png"))
        originals = [hil_benchmark.cv2.imread(str(f), hil_benchmark.cv2.IMREAD_UNCHANGED) for f in files]
        for picture in shown:
            self.assertEqual(picture.shape, (480, 640, 3))
            # a 640x480 window holds the 160x120 image four times larger, pixel for pixel
            small = picture[::4, ::4, 0]
            self.assertTrue(any(self.np.array_equal(small, original) for original in originals))
            self.assertTrue(self.np.array_equal(picture, self.np.repeat(self.np.repeat(
                small, 4, axis=0), 4, axis=1)[..., None].repeat(3, axis=2)))


@unittest.skipUnless(READY, "the HIL bench needs pyserial, OpenCV, numpy, scikit-learn and seaborn")
class EnterPromptTest(HilSessionCase):
    def test_the_sweep_waits_for_enter(self):
        board = self.board()
        with mock.patch("builtins.input", return_value="") as prompt:
            code, log, _ = self.run_bench(board, confirm=True)
        self.assertIsNone(code, log)
        prompt.assert_called_once()
        self.assertEqual(board.triggers, self.total)

    def test_no_confirm_does_not_ask(self):
        with mock.patch("builtins.input", side_effect=AssertionError("asked for ENTER")):
            code, log, _ = self.run_bench(self.board())
        self.assertIsNone(code, log)

    def test_without_a_terminal_the_bench_says_what_to_do(self):
        board = self.board()
        with mock.patch("builtins.input", side_effect=EOFError):
            code, log, _ = self.run_bench(board, confirm=True)
        self.assertEqual(code, 1)
        self.assertIn("--no_confirm", log)
        self.assertEqual(board.triggers, 0)
        self.assertTrue(board.closed)
        self.assertFalse(os.path.exists(self.out) and os.listdir(self.out))

    def test_ctrl_c_at_the_prompt_writes_nothing(self):
        board = self.board()
        with mock.patch("builtins.input", side_effect=KeyboardInterrupt):
            code, log, _ = self.run_bench(board, confirm=True)
        self.assertEqual(code, 1)
        self.assertIn("nothing was written", log)
        self.assertTrue(board.closed)


if __name__ == "__main__":
    unittest.main()
