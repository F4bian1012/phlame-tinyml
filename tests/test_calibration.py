#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Checks the rig calibrator (src/calibrate_hil_rig.py) without a board or a
camera: its measurements on patterns whose distortion is known, and a whole
--report run against a simulated board whose "camera" returns the picture the
stimulus window was asked to show.

That says the arithmetic and the plumbing are right. It does not say how the
figures behave on real frames, with blur, noise and reflections: that needs
the rig.

Skipped when pyserial, OpenCV or numpy are missing.

    python -m unittest discover -s tests -v
"""
import contextlib
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

from test_boards import load_bench  # noqa: E402
from test_hil_session import FakeClock, RigBoard, simulated_rig  # noqa: E402

hil_benchmark = load_bench("hil_benchmark")
calibrate = load_bench("calibrate_hil_rig") if hil_benchmark is not None else None
W, H = 160, 120


def moved(pattern, scale=1.0, degrees=0.0, shift=(0.0, 0.0)):
    """The pattern as a camera would see it scaled, rotated about the centre and shifted (bilinear)."""
    cv2 = hil_benchmark.cv2
    centre = ((W - 1) / 2.0, (H - 1) / 2.0)
    matrix = cv2.getRotationMatrix2D(centre, -degrees, scale)    # OpenCV counts angles the other way round
    matrix[:, 2] += shift
    return cv2.warpAffine(pattern, matrix, (W, H), flags=cv2.INTER_LINEAR)


@unittest.skipIf(calibrate is None, "calibrate_hil_rig needs pyserial, OpenCV, numpy and Pillow")
class PatternMetricsTest(unittest.TestCase):
    def test_perfect_capture_passes_every_check(self):
        geometry = calibrate.geometry_metrics(calibrate.pattern_geometry())
        self.assertTrue(geometry["ok"] and geometry["pass"], geometry)
        self.assertEqual((geometry["shift_x_px"], geometry["shift_y_px"], geometry["rotation_deg"]), (0.0, 0.0, 0.0))
        self.assertAlmostEqual(geometry["scale"], 1.0, places=3)
        self.assertEqual(geometry["missing"], [])
        self.assertEqual(calibrate.advice(geometry), "geometry within tolerance")

        gray = calibrate.gray_metrics(calibrate.pattern_gray())
        self.assertEqual(gray["levels_captured"], [float(g) for g in calibrate.GRAY_LEVELS])
        self.assertTrue(gray["monotonic"] and gray["pass"])
        self.assertEqual(gray["dynamic_range"], 255.0)

        flat = calibrate.flat_metrics([calibrate.pattern_flat()] * 3)
        self.assertEqual((flat["vignetting_ratio"], flat["exposure_sd"], flat["saturated_pct"], flat["n_captures"]),
                         (1.0, 0.0, 0.0, 3))
        self.assertTrue(flat["pass"])

        focus = calibrate.focus_metrics(calibrate.pattern_focus())
        self.assertEqual(focus["fine_to_coarse"], 1.0)
        self.assertTrue(focus["pass"])

    def test_geometry_recovers_a_known_shift_scale_and_rotation(self):
        seen = moved(calibrate.pattern_geometry(), scale=0.92, degrees=3.0, shift=(5.0, -3.0))
        g = calibrate.geometry_metrics(seen)
        self.assertTrue(g["ok"], g)
        self.assertAlmostEqual(g["scale"], 0.92, delta=0.01)
        self.assertAlmostEqual(g["rotation_deg"], 3.0, delta=0.3)
        self.assertAlmostEqual(g["shift_x_px"], 5.0, delta=0.5)
        self.assertAlmostEqual(g["shift_y_px"], -3.0, delta=0.5)
        self.assertLess(g["residual_px"], 0.5)
        self.assertFalse(g["pass"])
        advice = calibrate.advice(g)
        for expected in ("move the camera CLOSER", "to the RIGHT", "UP", "clockwise"):
            self.assertIn(expected, advice)

    def test_small_misalignment_within_the_criteria_passes(self):
        g = calibrate.geometry_metrics(moved(calibrate.pattern_geometry(), scale=1.01, degrees=0.4, shift=(1.0, -1.0)))
        self.assertTrue(g["pass"], g)

    def test_frame_that_does_not_show_the_pattern_is_reported_not_measured(self):
        import numpy as np
        dark = calibrate.geometry_metrics(np.zeros((H, W), np.uint8))
        self.assertFalse(dark["ok"])
        self.assertIn("does not see the stimulus", calibrate.advice(dark))
        cut = calibrate.geometry_metrics(moved(calibrate.pattern_geometry(), shift=(30.0, 0.0)))
        self.assertTrue(not cut["ok"] or "top-right" in cut["missing"], cut)

    def test_washed_out_and_blurred_frames_are_flagged(self):
        import numpy as np
        cv2 = hil_benchmark.cv2
        washed = (calibrate.pattern_gray().astype(np.float32) * 0.3 + 100).astype(np.uint8)
        self.assertFalse(calibrate.gray_metrics(washed)["pass"])          # range 76 of the 150 asked for
        blurred = cv2.GaussianBlur(calibrate.pattern_focus(), (0, 0), 3.0)
        focus = calibrate.focus_metrics(blurred)
        self.assertLess(focus["fine_to_coarse"], calibrate.CRITERIA["focus_ratio_min"])
        self.assertFalse(focus["pass"])
        y, x = np.mgrid[0:H, 0:W]
        falloff = 1.0 - 0.9 * (((x - W / 2) / (W / 2)) ** 2 + ((y - H / 2) / (H / 2)) ** 2) / 2
        vignetted = (calibrate.pattern_flat() * falloff).astype(np.uint8)
        flat = calibrate.flat_metrics([vignetted, vignetted])
        self.assertLess(flat["vignetting_ratio"], calibrate.CRITERIA["vignette_min"])
        self.assertFalse(flat["pass"])

    def test_correlation_of_a_frame_with_the_image_shown(self):
        import numpy as np
        rng = np.random.default_rng(3)
        picture = rng.integers(0, 256, size=(H, W), dtype=np.uint8)
        self.assertAlmostEqual(calibrate.image_ncc(picture, picture), 1.0, places=9)
        dimmer = (picture.astype(np.float32) * 0.5 + 20).astype(np.uint8)       # gain and offset do not count
        self.assertGreater(calibrate.image_ncc(picture, dimmer), 0.999)
        other = rng.integers(0, 256, size=(H, W), dtype=np.uint8)
        self.assertLess(abs(calibrate.image_ncc(picture, other)), 0.05)


@unittest.skipIf(calibrate is None, "calibrate_hil_rig needs pyserial, OpenCV, numpy and Pillow")
class LiveLanguageTest(unittest.TestCase):
    """--lang: the live mode (preview window and its lines) in another language; the report stays in English."""

    def seen(self):
        return calibrate.geometry_metrics(moved(calibrate.pattern_geometry(), scale=0.92, degrees=3.0, shift=(5.0, -3.0)))

    def test_every_language_has_every_message_with_the_same_arguments(self):
        import re
        fields = re.compile(r"%[-+0-9.]*[a-z]")
        english = calibrate.TEXT["en"]
        for lang, text in calibrate.TEXT.items():
            self.assertEqual(sorted(text), sorted(english), lang)
            for key, message in text.items():
                if isinstance(message, str):
                    self.assertEqual(fields.findall(message), fields.findall(english[key]), (lang, key))
        self.assertEqual(sorted(calibrate.TEXT["es"]["squares"]), sorted(calibrate.SQUARE_NAMES))

    def test_advice_in_spanish_says_the_same_as_in_english(self):
        g = self.seen()
        spanish = calibrate.advice(g, "es")
        for expected in ("ACERCA la cámara", "a la DERECHA", "hacia ARRIBA", "sentido horario"):
            self.assertIn(expected, spanish)
        self.assertEqual(len(spanish.split("; ")), len(calibrate.advice(g).split("; ")))
        self.assertEqual(calibrate.advice(calibrate.geometry_metrics(calibrate.pattern_geometry()), "es"),
                         "geometría dentro de la tolerancia")

    def test_frame_without_the_pattern_in_spanish_and_unchanged_in_english(self):
        import numpy as np
        dark = calibrate.geometry_metrics(np.zeros((H, W), np.uint8))
        self.assertEqual(calibrate.advice(dark), dark["hint"])          # what the report stores
        self.assertIn("la cámara no ve el estímulo", calibrate.advice(dark, "es"))
        self.assertEqual(calibrate.advice({"ok": False}, "es"), "")     # no capture at all

    def test_a_missing_square_is_named_in_the_language(self):
        g = dict(self.seen(), missing=["top-left"])
        self.assertIn("top-left", calibrate.advice(g))
        self.assertIn("arriba-izquierda", calibrate.advice(g, "es"))
        self.assertNotIn("top-left", calibrate.advice(g, "es"))

    def test_preview_picture_is_drawn_in_both_languages(self):
        frame = moved(calibrate.pattern_geometry(), scale=0.92, degrees=3.0, shift=(5.0, -3.0))
        g = calibrate.geometry_metrics(frame)
        pictures = {lang: calibrate.preview_picture(frame, g, 1, lang) for lang in ("en", "es")}
        for picture in pictures.values():
            self.assertEqual(picture.shape, (H * 6 + calibrate.BAR_H, W * 6, 3))
            self.assertGreater(int(picture[H * 6:].max()), 150)         # the bar has text
        self.assertTrue((pictures["en"][:H * 6] == pictures["es"][:H * 6]).all())   # same frame and marks
        self.assertFalse((pictures["en"][H * 6:] == pictures["es"][H * 6:]).all())  # another text

    def test_advice_is_wrapped_between_two_pieces_of_advice(self):
        lines = calibrate.wrap_advice("one; two; three", columns=9)
        self.assertEqual(lines, ["one; two;", "three"])
        self.assertEqual(calibrate.wrap_advice("a" * 30, columns=10, lines=3), ["a" * 10])
        self.assertLessEqual(len(calibrate.wrap_advice("; ".join(["piece"] * 40), columns=20, lines=3)), 3)

    def test_language_option(self):
        self.assertEqual(calibrate.parse_args(["--live"]).lang, "en")
        self.assertEqual(calibrate.parse_args(["--live", "--lang", "es"]).lang, "es")
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            calibrate.parse_args(["--live", "--lang", "xx"])


@unittest.skipIf(calibrate is None, "calibrate_hil_rig needs pyserial, OpenCV, numpy and Pillow")
class CalibrationRunTest(unittest.TestCase):
    """A --report run. The simulated camera returns what the window shows, reduced to 160x120."""

    def setUp(self):
        import numpy as np
        self.np = np
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.model = str(root / "any_int8.tflite")
        Path(self.model).write_bytes(b"model bytes; the calibrator does not look at them")
        self.image = str(root / "000000001234.png")
        rng = np.random.default_rng(5)
        self.picture = rng.integers(0, 256, size=(H, W), dtype=np.uint8)
        hil_benchmark.cv2.imwrite(self.image, self.picture)
        self.out = str(root / "calibration")
        self.clock = FakeClock()
        self.shown = []

    def camera(self):
        """What a perfectly aligned camera sees: the 640x480 window holds the stimulus four times larger."""
        return self.shown[-1][::4, ::4, 0]

    def run_calibrator(self, board, *extra):
        argv = ["--port", "SIM", "--report", "--settle", "0.2", "--repeats", "2", "--output_dir", self.out,
                "--notes", "simulated rig", *extra]
        code = None
        with contextlib.redirect_stdout(io.StringIO()) as log, \
                mock.patch.object(calibrate, "time", self.clock), \
                simulated_rig(hil_benchmark, board, self.clock, shown=self.shown):
            try:
                calibrate.main(argv)
            except SystemExit as stop:
                code = stop.code
        return code, log.getvalue()

    def report(self):
        files = sorted(os.listdir(self.out))
        self.assertEqual([os.path.splitext(f)[1] for f in files], [".json", ".png"], files)
        self.assertEqual(os.path.splitext(files[0])[0], os.path.splitext(files[1])[0])
        with open(os.path.join(self.out, files[0]), encoding="utf-8") as fh:
            text = fh.read()
        return json.loads(text), text

    def test_report_of_an_aligned_rig(self):
        board = RigBoard(self.model, self.clock, camera=self.camera)
        code, log = self.run_calibrator(board, "--images", self.image)
        self.assertIsNone(code, log)
        report, text = self.report()
        self.assertTrue(report["all_pass"], report)
        for part in ("geometry", "gray", "flat", "focus"):
            self.assertTrue(report[part]["pass"], part)
        self.assertEqual(report["flat"]["n_captures"], 2)
        self.assertEqual(report["images"], [{"image": "000000001234.png", "ncc": 1.0}])
        self.assertEqual(report["images_ncc_mean"], 1.0)
        self.assertEqual((report["board"], report["port"], report["settle_s"], report["stimulus_monitor"]),
                         ("PORTENTA_H7", "SIM", 0.2, None))
        self.assertEqual(report["frame_size"], {"width": 160, "height": 120})
        self.assertEqual(report["criteria"], calibrate.CRITERIA)
        self.assertNotIn(self.tmp.name.replace("\\", "/"), text.replace("\\\\", "/"))    # no local paths
        # 4 patterns, the flat field once more, and the test image: 6 captures
        self.assertEqual(board.triggers, 6)
        self.assertEqual(len(self.shown), 6)
        self.assertTrue(board.closed)
        self.assertFalse(board.dump)             # the frame dump is turned off again at the end
        self.assertIn("overall  : PASS", log)

    def test_test_image_is_shown_from_its_file_and_compared_with_what_the_model_receives(self):
        board = RigBoard(self.model, self.clock, camera=self.camera)
        self.run_calibrator(board, "--images", self.image)
        self.assertTrue(self.np.array_equal(self.shown[-1][::4, ::4, 0], self.picture))

    def test_misaligned_camera_is_reported_as_a_warning_not_an_error(self):
        board = RigBoard(self.model, self.clock,
                         camera=lambda: moved(self.camera(), scale=0.9, degrees=2.0, shift=(4.0, 0.0)))
        code, log = self.run_calibrator(board)
        self.assertIsNone(code, log)             # the criteria are reported, never enforced
        report, _ = self.report()
        self.assertFalse(report["all_pass"])
        self.assertFalse(report["geometry"]["pass"])
        self.assertIn("move the camera CLOSER", report["geometry"]["advice"])
        self.assertIn("geometry : WARN", log)

    def test_firmware_of_another_board_stops_the_calibration(self):
        board = RigBoard(self.model, self.clock, board="NICLA_VISION", camera=self.camera)
        code, log = self.run_calibrator(board)
        self.assertEqual(code, 2)
        self.assertIn("--board nicla", log)
        self.assertEqual(board.triggers, 0)
        self.assertTrue(board.closed)

    def test_same_board_named_on_the_command_line_goes_to_its_own_folder(self):
        args = calibrate.parse_args(["--report", "--board", "nicla"])
        self.assertEqual(args.output_dir, os.path.join("results", "nicla", "hil", "calibration"))
        self.assertEqual(calibrate.parse_args(["--live"]).output_dir, os.path.join("results", "hil", "calibration"))
        self.assertEqual(args.manifest, os.path.join("data", "splits", "test_manifest.csv"))

    def test_live_or_report_must_be_chosen(self):
        with contextlib.redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit) as ctx:
            calibrate.parse_args(["--port", "SIM"])
        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("--live and/or --report", err.getvalue())

    def test_image_can_be_named_as_in_the_test_manifest(self):
        manifest = os.path.join(self.tmp.name, "test_manifest.csv")
        with open(manifest, "w", encoding="utf-8", newline="") as fh:
            fh.write("split,class_name,label,filename,src_path,dst_path,hil_subset\n")
            fh.write("test,person,1,000000001234.png,%s,%s,1\n" % (self.image, self.image))
        self.assertEqual(calibrate.resolve_image("000000001234.png", manifest), self.image)
        self.assertEqual(calibrate.resolve_image(self.image, manifest), self.image)
        with contextlib.redirect_stdout(io.StringIO()) as log, self.assertRaises(SystemExit) as ctx:
            calibrate.resolve_image("not_there.png", manifest)
        self.assertEqual(ctx.exception.code, 1)
        self.assertIn("Image not found", log.getvalue())


if __name__ == "__main__":
    unittest.main()
