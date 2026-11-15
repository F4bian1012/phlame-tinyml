#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Checks a whole run of the PIL bench (src/pil_benchmark.py) against a simulated
board. No hardware.

What is checked is what makes a PIL result worth trusting:
  - nothing is measured, and nothing is written, until the board has said that
    it carries the .tflite given with --model_path, that it is the board given
    with --board and that it takes images of the size about to be sent;
  - a firmware built before it printed those lines is refused unless the
    operator allows an unverified run, which is then marked as such;
  - a board that stops answering ends the run after a few images instead of
    making every remaining image wait;
  - a run that does not reach its last image (silent board, lost link, Ctrl+C,
    an error) keeps what it measured and says that it is not complete;
  - the run records where the firmware says its tensor arena is, and a run
    asked for the HIL subset of the test split measures exactly those images;
  - the keys the bench writes are the ones docs/results_schema.md documents.

The board is the SimulatedBoard of test_boards.py. Time is a fake clock:
waiting costs nothing, and a board that never answers does not slow the tests
down.

Skipped when pyserial, Pillow, numpy, scikit-learn or seaborn are missing.

    python -m unittest discover -s tests -v
"""
import csv
import importlib.util
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("MPLBACKEND", "Agg")   # the bench draws the matrix; no window

import boards  # noqa: E402
import test_results_schema as schema  # noqa: E402
from test_boards import PilFolderCase, SimulatedBoard, pil_benchmark, pil_identity  # noqa: E402

READY = (pil_benchmark is not None
         and all(importlib.util.find_spec(name) for name in ("sklearn", "seaborn", "matplotlib")))
NEEDS = "the PIL bench needs pyserial, Pillow, numpy, scikit-learn and seaborn"

# Keys the PIL bench writes since November 2026 (see docs/results_schema.md).
NEW_KEYS = {"model.fnv1a_tflite", "model.fnv1a_chip", "model.identity_verified", "model.arena_used_bytes",
            "model.arena_memory",
            "run.completed", "run.stopped_because", "run.input_shape_verified", "run.max_unanswered",
            "dataset.n_images_planned"}


@unittest.skipUnless(READY, NEEDS)
class ModelIdentityTest(PilFolderCase):
    def test_verified_run_records_the_fingerprint_of_the_board_and_of_the_file(self):
        board = self.board()
        code, log = self.run_folder(board)
        self.assertIsNone(code, log)
        metrics = self.metrics()
        fingerprint = boards.fnv1a32_file(self.model)
        self.assertEqual((metrics["model"]["fnv1a_tflite"], metrics["model"]["fnv1a_chip"],
                          metrics["model"]["identity_verified"], metrics["model"]["arena_used_bytes"]),
                         (fingerprint, fingerprint, True, 181492))
        self.assertEqual((metrics["run"]["completed"], metrics["run"]["stopped_because"],
                          metrics["run"]["input_shape_verified"], metrics["run"]["max_unanswered"]),
                         (True, None, True, 5))
        self.assertEqual((metrics["dataset"]["n_images_planned"], metrics["dataset"]["n_images"]), (4, 4))
        self.assertIn("IDENTITY VERIFIED", log)
        self.assertEqual((board.empty_packets, board.inferences), (1, 4))   # asked once, before the first image
        self.assertTrue(board.closed)

    def test_boot_lines_of_a_board_just_reset_are_enough(self):
        # the port is opened for the first time after a reset: the firmware prints its boot lines
        boot = ["--- STARTING DIAGNOSTICS ---", "BOARD:PORTENTA_H7", "AllocateTensors() OK.",
                "Setup completed successfully."] + pil_identity(self.model)
        board = SimulatedBoard("pil", board="PORTENTA_H7", boot=boot, identity=pil_identity(self.model))
        code, log = self.run_folder(board)
        self.assertIsNone(code, log)
        self.assertEqual((board.empty_packets, board.inferences), (0, 4))   # nothing had to be asked
        self.assertTrue(self.metrics()["model"]["identity_verified"])

    def test_another_model_on_the_board_stops_before_measuring(self):
        board = SimulatedBoard("pil", identity=pil_identity(self.model, fnv1a="deadbeef"))
        code, log = self.run_folder(board)
        self.assertEqual(code, 2)
        self.assertIn("MODEL MISMATCH", log)
        self.assertIn("--allow_model_mismatch", log)
        self.assertEqual((board.inferences, self.written()), (0, []))
        self.assertTrue(board.closed)

    def test_same_fingerprint_with_another_size_is_another_model(self):
        board = SimulatedBoard("pil", identity=pil_identity(self.model, model_bytes=7))
        code, log = self.run_folder(board)
        self.assertEqual(code, 2)
        self.assertIn("MODEL MISMATCH", log)
        self.assertEqual(board.inferences, 0)

    def test_allowed_mismatch_is_measured_and_marked(self):
        board = SimulatedBoard("pil", board="PORTENTA_H7", identity=pil_identity(self.model, fnv1a="deadbeef"))
        code, log = self.run_folder(board, allow_model_mismatch=True)
        self.assertIsNone(code, log)
        metrics = self.metrics()
        self.assertEqual((metrics["model"]["identity_verified"], metrics["model"]["fnv1a_chip"],
                          metrics["model"]["fnv1a_tflite"], metrics["run"]["completed"]),
                         (False, "deadbeef", boards.fnv1a32_file(self.model), True))
        self.assertIn("NOT verified", log)

    def test_firmware_without_identity_lines_is_refused(self):
        board = SimulatedBoard("pil", board="PORTENTA_H7")   # today's build: BOARD with every answer, nothing else
        code, log = self.run_folder(board)
        self.assertEqual(code, 2)
        self.assertIn("did not report the fingerprint", log)
        self.assertIn("compile_upload_arduino.py --target pil", log)
        self.assertEqual((board.empty_packets, board.inferences, self.written()), (2, 0, []))

    def test_firmware_without_identity_lines_runs_when_allowed_and_is_marked(self):
        board = SimulatedBoard("pil", board="PORTENTA_H7")
        code, log = self.run_folder(board, allow_model_mismatch=True)
        self.assertIsNone(code, log)
        metrics = self.metrics()
        self.assertEqual((metrics["model"]["identity_verified"], metrics["model"]["fnv1a_chip"],
                          metrics["model"]["arena_used_bytes"], metrics["run"]["input_shape_verified"],
                          metrics["run"]["board"], metrics["dataset"]["n_answered"]),
                         (False, None, None, False, "PORTENTA_H7", 4))

    def test_missing_model_file_is_refused(self):
        board = self.board()
        code, log = self.run_folder(board, model_path=os.path.join(self.tmp.name, "nothing_int8.tflite"))
        self.assertEqual(code, 2)
        self.assertIn("is not a file", log)
        self.assertEqual((board.inferences, self.written()), (0, []))

    def test_refused_run_leaves_an_earlier_run_as_it_was(self):
        os.makedirs(self.out)
        earlier = {}
        for name in ("metrics_sim_int8.json", "predictions_sim_int8.csv", "latency_metrics_sim_int8.csv"):
            earlier[name] = "earlier run: %s\n" % name
            Path(self.out, name).write_text(earlier[name], encoding="utf-8")
        code, _ = self.run_folder(SimulatedBoard("pil", identity=pil_identity(self.model, fnv1a="deadbeef")))
        self.assertEqual(code, 2)
        for name, text in earlier.items():
            self.assertEqual(Path(self.out, name).read_text(encoding="utf-8"), text, name)

    def test_board_that_does_not_answer_ends_the_run_before_it_starts(self):
        board = SimulatedBoard("pil", mute=True)
        code, log = self.run_folder(board)
        self.assertEqual(code, 1)
        self.assertIn("did not answer", log)
        self.assertEqual(self.written(), [])
        self.assertTrue(board.closed)

    def test_hil_firmware_is_recognised(self):
        board = SimulatedBoard("pil", mute=True, boot=["BOARD:PORTENTA_H7", "CAM_INIT:OK", "READY_HIL"])
        code, log = self.run_folder(board)
        self.assertEqual(code, 1)
        self.assertIn("HIL firmware", log)
        self.assertEqual(self.written(), [])

    def test_board_left_in_the_middle_of_a_packet_is_asked_again(self):
        # the first empty packet closes the packet an interrupted run left open: an inference on rubbish
        class MidPacket(SimulatedBoard):
            def write(self, data):
                if data == b"#@" and self.empty_packets == 0:
                    self.empty_packets += 1
                    self.answer()
                    self.inferences = 0
                    return len(data)
                return super().write(data)

        board = MidPacket("pil", board="PORTENTA_H7", identity=pil_identity(self.model))
        code, log = self.run_folder(board)
        self.assertIsNone(code, log)
        self.assertEqual((board.empty_packets, board.inferences), (2, 4))
        self.assertTrue(self.metrics()["model"]["identity_verified"])


@unittest.skipUnless(READY, NEEDS)
class InputShapeTest(PilFolderCase):
    def test_size_is_taken_from_the_firmware_when_it_is_not_given(self):
        board = self.board()
        code, log = self.run_folder(board, width=None, height=None)
        self.assertIsNone(code, log)
        run = self.metrics()["run"]
        self.assertEqual((run["input_shape"], run["input_shape_verified"]),
                         ({"width": 16, "height": 12, "channels": 1}, True))
        self.assertEqual(board.inferences, 4)

    def test_another_size_is_refused_before_measuring(self):
        # 320x320 is what the bench sent when --width and --height were left out, until November 2026
        board = self.board()
        code, log = self.run_folder(board, width=320, height=320)
        self.assertEqual(code, 2)
        self.assertIn("--width 16 --height 12", log)
        self.assertEqual((board.inferences, self.written()), (0, []))

    def test_colour_must_match_the_channels_of_the_model(self):
        board = self.board()
        code, log = self.run_folder(board, grayscale=False)
        self.assertEqual(code, 2)
        self.assertIn("no --color", log)
        self.assertEqual(board.inferences, 0)
        colour = SimulatedBoard("pil", identity=pil_identity(self.model, shape="12x16x3"))
        code, log = self.run_folder(colour)
        self.assertEqual(code, 2)
        self.assertIn("--color.", log)

    def test_firmware_that_does_not_announce_the_shape_needs_the_size(self):
        board = SimulatedBoard("pil", board="PORTENTA_H7")
        code, log = self.run_folder(board, allow_model_mismatch=True, width=None, height=None)
        self.assertEqual(code, 2)
        self.assertIn("pass --width and --height", log)
        self.assertEqual((board.inferences, self.written()), (0, []))


@unittest.skipUnless(READY, NEEDS)
class SilentBoardTest(PilFolderCase):
    N_PER_CLASS = 4

    def test_board_that_stops_answering_ends_the_run_after_n_images(self):
        board = self.board(silent_from=2)
        started = self.clock.now
        code, log = self.run_folder(board, max_unanswered=3)
        self.assertEqual(code, 3)
        self.assertIn("did NOT finish", log)
        self.assertEqual(board.packets, 4)             # one answered, three not: the other four are not sent
        metrics = self.metrics()
        self.assertFalse(metrics["run"]["completed"])
        self.assertIn("the board stopped answering: 3 images in a row", metrics["run"]["stopped_because"])
        self.assertEqual((metrics["dataset"]["n_images_planned"], metrics["dataset"]["n_images"],
                          metrics["dataset"]["n_answered"], metrics["dataset"]["n_unanswered"],
                          metrics["run"]["max_unanswered"]), (8, 4, 1, 3, 3))
        rows = self.predictions()
        self.assertEqual([r["PredLabel"] for r in rows], ["1", "", "", ""])
        # three waits of 30 s, not seven
        self.assertLess(self.clock.now - started, 4 * pil_benchmark.RESPONSE_TIMEOUT_S)

    def test_an_answer_starts_the_count_again(self):
        board = self.board(silent_on={2, 3, 5, 6})
        code, log = self.run_folder(board, max_unanswered=3)
        self.assertIsNone(code, log)
        metrics = self.metrics()
        self.assertEqual((metrics["run"]["completed"], metrics["dataset"]["n_images"],
                          metrics["dataset"]["n_answered"], metrics["dataset"]["n_unanswered"]),
                         (True, 8, 4, 4))

    def test_zero_never_gives_up(self):
        board = self.board(silent_from=1)
        code, log = self.run_folder(board, max_unanswered=0)
        self.assertIsNone(code, log)
        metrics = self.metrics()
        self.assertEqual((board.packets, metrics["run"]["completed"], metrics["dataset"]["n_unanswered"],
                          metrics["global_metrics"]), (8, True, 8, None))


@unittest.skipUnless(READY, NEEDS)
class RunCutShortTest(PilFolderCase):
    def test_ctrl_c_keeps_what_was_measured_and_marks_the_run(self):
        board = self.board(raise_at=3, failure=KeyboardInterrupt())
        code, log = self.run_folder(board)
        self.assertEqual(code, 3)
        metrics = self.metrics()
        self.assertEqual((metrics["run"]["completed"], metrics["run"]["stopped_because"]),
                         (False, "interrupted by the operator (Ctrl+C) at image 3 of 4"))
        self.assertEqual((metrics["dataset"]["n_images_planned"], metrics["dataset"]["n_images"],
                          metrics["dataset"]["n_answered"]), (4, 2, 2))
        self.assertEqual(len(self.predictions()), 2)
        self.assertEqual(metrics["latency"]["n_images_measured"], 2)
        self.assertTrue(board.closed)

    def test_image_that_cannot_be_read_ends_the_run_with_its_name_in_the_reason(self):
        Path(self.folder, "non_person", "zz_broken.png").write_bytes(b"not an image")
        board = self.board()
        code, log = self.run_folder(board)
        self.assertEqual(code, 3)
        metrics = self.metrics()
        self.assertFalse(metrics["run"]["completed"])
        reason = metrics["run"]["stopped_because"]
        self.assertIn("at image 3 of 5 (non_person/zz_broken.png)", reason)
        self.assertNotIn(os.path.basename(self.tmp.name), reason)    # no local path in a results file
        self.assertIn("zz_broken.png", log)
        self.assertEqual(len(self.predictions()), 2)
        self.assertEqual((metrics["dataset"]["n_images"], board.inferences), (2, 2))

    def test_link_lost_at_the_first_image_still_leaves_a_marked_file(self):
        board = self.board(drop_link_at=1)
        code, log = self.run_folder(board, reopen=lambda port, baud: None)
        self.assertEqual(code, 3)
        metrics = self.metrics()
        self.assertEqual((metrics["run"]["completed"], metrics["dataset"]["n_images"],
                          metrics["dataset"]["n_images_planned"], metrics["global_metrics"]),
                         (False, 0, 4, None))
        self.assertEqual(self.predictions(), [])


@unittest.skipUnless(READY, NEEDS)
class ArenaMemoryTest(PilFolderCase):
    """The run says where the firmware keeps its tensor arena, so that runs can be told apart by it."""

    def test_default_build_is_recorded_with_the_memory_of_its_board(self):
        code, log = self.run_folder(self.board())
        self.assertIsNone(code, log)
        self.assertEqual(self.metrics()["model"]["arena_memory"], "SDRAM")
        self.assertIn("Tensor arena of this build: SDRAM", log)
        code, log = self.run_folder(self.board("NICLA_VISION"), board_key="nicla")
        self.assertIsNone(code, log)
        self.assertEqual(self.metrics()["model"]["arena_memory"], "INTERNAL")

    def test_portenta_built_with_the_arena_in_internal_ram_is_recorded_as_such(self):
        board = SimulatedBoard("pil", board="PORTENTA_H7",
                               identity=pil_identity(self.model, arena_memory="INTERNAL"))
        code, log = self.run_folder(board)
        self.assertIsNone(code, log)
        metrics = self.metrics()
        self.assertEqual((metrics["run"]["board"], metrics["model"]["arena_memory"],
                          metrics["model"]["arena_used_bytes"], metrics["model"]["identity_verified"]),
                         ("PORTENTA_H7", "INTERNAL", 181492, True))
        self.assertIn("Tensor arena of this build: INTERNAL", log)

    def test_it_is_also_read_from_the_boot_lines_of_a_board_just_reset(self):
        identity = pil_identity(self.model, arena_memory="INTERNAL")
        boot = ["--- STARTING DIAGNOSTICS ---", "BOARD:PORTENTA_H7", "Tensor arena size (bytes): 376832",
                "Tensor arena address (internal RAM): 0x24001E40", "AllocateTensors() OK."] + identity
        board = SimulatedBoard("pil", board="PORTENTA_H7", boot=boot, identity=identity)
        code, log = self.run_folder(board)
        self.assertIsNone(code, log)
        self.assertEqual((board.empty_packets, self.metrics()["model"]["arena_memory"]), (0, "INTERNAL"))

    def test_firmware_that_does_not_announce_it_is_recorded_as_null(self):
        # a build with the identity lines but earlier than the ARENA_MEMORY one: verified, complete, and null
        board = SimulatedBoard("pil", board="PORTENTA_H7", identity=pil_identity(self.model, arena_memory=None))
        code, log = self.run_folder(board)
        self.assertIsNone(code, log)
        metrics = self.metrics()
        self.assertIn("arena_memory", metrics["model"])
        self.assertEqual((metrics["model"]["arena_memory"], metrics["model"]["arena_used_bytes"],
                          metrics["model"]["identity_verified"], metrics["run"]["completed"]),
                         (None, 181492, True, True))
        self.assertNotIn("Tensor arena of this build", log)

    def test_build_without_identity_lines_is_recorded_as_null(self):
        board = SimulatedBoard("pil", board="PORTENTA_H7")
        code, log = self.run_folder(board, allow_model_mismatch=True)
        self.assertIsNone(code, log)
        self.assertIsNone(self.metrics()["model"]["arena_memory"])


@unittest.skipUnless(READY, NEEDS)
class HilSubsetTest(PilFolderCase):
    """--hil_subset: only the images the test manifest flags for the HIL rig."""

    N_PER_CLASS = 3

    def manifest(self, flagged, columns=("split", "class_name", "label", "filename", "hil_subset")):
        path = os.path.join(self.tmp.name, "test_manifest.csv")
        with open(path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(columns)
            for label, cls in enumerate(("non_person", "person")):
                for i in range(self.N_PER_CLASS):
                    name = "img_%d.png" % i
                    row = {"split": "test", "class_name": cls, "label": label, "filename": name,
                           "hil_subset": int("%s/%s" % (cls, name) in flagged)}
                    writer.writerow([row[c] for c in columns])
        return path

    def test_only_the_flagged_images_are_sent(self):
        flagged = {"non_person/img_0.png", "non_person/img_2.png", "person/img_1.png"}
        board = self.board()
        code, log = self.run_folder(board, manifest_path=self.manifest(flagged))
        self.assertIsNone(code, log)
        metrics = self.metrics()
        self.assertEqual((metrics["dataset"]["manifest_subset"], metrics["dataset"]["n_images_planned"],
                          metrics["dataset"]["n_images"], metrics["run"]["completed"], board.inferences),
                         ("hil_subset==1", 3, 3, True, 3))
        self.assertEqual({r["RelPath"] for r in self.predictions()}, flagged)
        # the labels are still those of the folders, whatever was left out
        self.assertEqual({r["RelPath"]: r["TrueClass"] for r in self.predictions()},
                         {rel: rel.split("/")[0] for rel in flagged})

    def test_run_of_the_whole_folder_records_no_subset(self):
        code, log = self.run_folder(self.board())
        self.assertIsNone(code, log)
        metrics = self.metrics()
        self.assertIn("manifest_subset", metrics["dataset"])
        self.assertEqual((metrics["dataset"]["manifest_subset"], metrics["dataset"]["n_images"]), (None, 6))

    def test_count_draws_from_the_flagged_images(self):
        flagged = {"non_person/img_0.png", "non_person/img_2.png", "person/img_1.png"}
        code, log = self.run_folder(self.board(), manifest_path=self.manifest(flagged), n_samples=2)
        self.assertIsNone(code, log)
        rows = self.predictions()
        self.assertEqual(len(rows), 2)
        self.assertLess({r["RelPath"] for r in rows}, flagged)

    def test_flagged_image_that_is_not_in_the_folder_stops_before_the_port_is_opened(self):
        os.remove(os.path.join(self.folder, "person", "img_1.png"))
        board = self.board()
        code, log = self.run_folder(board, manifest_path=self.manifest({"non_person/img_0.png", "person/img_1.png"}))
        self.assertEqual(code, 1)
        self.assertIn("person/img_1.png", log)
        self.assertNotIn("Connecting", log)
        self.assertEqual((board.empty_packets, board.inferences, self.written()), (0, 0, []))

    def test_manifest_that_cannot_give_the_subset_stops_the_run(self):
        cases = {"not found": os.path.join(self.tmp.name, "no_such_manifest.csv"),
                 "has hil_subset == 1": self.manifest(set())}
        for message, path in cases.items():
            board = self.board()
            code, log = self.run_folder(board, manifest_path=path)
            self.assertEqual(code, 1, message)
            self.assertIn(message, log)
            self.assertEqual((board.inferences, self.written()), (0, []))
        board = self.board()
        code, log = self.run_folder(board, manifest_path=self.manifest({"person/img_1.png"},
                                                                       columns=("class_name", "filename")))
        self.assertEqual(code, 1)
        self.assertIn("lacks the column(s) ['hil_subset']", log)
        self.assertEqual((board.inferences, self.written()), (0, []))


@unittest.skipUnless(READY, NEEDS)
class WrittenKeysTest(PilFolderCase):
    def paths_of(self, metrics):
        paths = set()
        schema.collect_paths(metrics, "", paths)
        return paths

    def test_every_key_written_is_documented_and_the_new_ones_are_written(self):
        documented, _ = schema.documented_paths()
        code, log = self.run_folder(self.board())
        self.assertIsNone(code, log)
        complete = self.paths_of(self.metrics())
        code, _ = self.run_folder(self.board(raise_at=2, failure=KeyboardInterrupt()))
        self.assertEqual(code, 3)
        cut_short = self.paths_of(self.metrics())
        for written in (complete, cut_short):
            self.assertEqual(sorted(written - documented), [], "keys missing from the docs")
            self.assertEqual(sorted(NEW_KEYS - written), [], "keys of November 2026 the bench did not write")
            # the keys no published file has yet are documented from what the bench writes
            self.assertEqual(sorted(schema.NOT_YET_PUBLISHED_PIL_METRICS - written), [],
                             "keys listed as not yet published that the bench does not write")

    def test_new_keys_no_published_file_has_are_listed_as_such(self):
        # tests/test_results_schema.py fails when a documented key is in no published file,
        # unless the key is listed there as not yet published
        published, _ = schema.real_paths("metrics_*.json")
        self.assertEqual(NEW_KEYS - published - schema.NOT_YET_PUBLISHED_PIL_METRICS, set())


@unittest.skipUnless(READY, NEEDS)
class CommandLineTest(unittest.TestCase):
    """What the command line refuses before it opens the port."""

    def run_script(self, *args):
        return subprocess.run([sys.executable, str(REPO / "src" / "pil_benchmark.py"), "--port", "NO_SUCH_PORT",
                               *args], cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                              env=dict(os.environ, PYTHONIOENCODING="utf-8"))

    def test_model_path_that_is_not_a_file_is_refused(self):
        result = self.run_script("--model_path", "models/tflite/no_such_model_int8.tflite", "--folder", "data/sample")
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("--model_path is not a file", result.stdout)
        self.assertNotIn("Connecting", result.stdout)

    def test_negative_max_unanswered_is_refused(self):
        result = self.run_script("--model_path", str(REPO / "README.md"), "--folder", "data/sample",
                                 "--max_unanswered", "-1")
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertNotIn("Connecting", result.stdout)

    def test_help_names_the_new_options(self):
        result = self.run_script("--help")
        self.assertEqual(result.returncode, 0)
        for option in ("--max_unanswered", "--allow_model_mismatch"):
            self.assertIn(option, result.stdout)

    def test_hil_subset_needs_the_test_split_of_a_partition(self):
        # the manifest belongs to <splits_dir>/test: with an arbitrary folder there is none to read
        result = self.run_script("--model_path", str(REPO / "README.md"), "--folder", "data/sample", "--hil_subset")
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("not with --folder or --image", result.stdout)
        self.assertNotIn("Connecting", result.stdout)
        self.assertIn("--hil_subset", self.run_script("--help").stdout)


if __name__ == "__main__":
    unittest.main()
