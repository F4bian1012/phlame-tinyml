#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Checks src/find_near_duplicates.py on a small synthetic partition whose answer
is known: the test split holds an exact copy of a training image, a resized
and re-encoded copy, a mirrored and rotated copy, and images that are in
neither train nor val. No dataset, no TensorFlow (the candidate search uses
--embed pixels); skipped when numpy or OpenCV are not installed.

    python -m unittest discover -s tests -v
"""
import contextlib
import csv
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import find_near_duplicates as fnd  # noqa: E402

np, cv2 = fnd.np, fnd.cv2
READY = np is not None and cv2 is not None
W, H = fnd.W_IMG, fnd.H_IMG


def photo(seed):
    """A synthetic 'photo': smooth background plus shapes, different for every seed."""
    rng = np.random.default_rng(seed)
    img = cv2.resize(rng.uniform(40, 215, (6, 8)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    img = np.clip(img, 0, 255).astype(np.uint8)
    for _ in range(14):
        x, y = int(rng.integers(0, W)), int(rng.integers(0, H))
        shade = int(rng.integers(0, 256))
        kind = int(rng.integers(0, 3))
        if kind == 0:
            cv2.rectangle(img, (x, y), (x + int(rng.integers(8, 40)), y + int(rng.integers(8, 30))), shade, -1)
        elif kind == 1:
            cv2.circle(img, (x, y), int(rng.integers(5, 18)), shade, -1)
        else:
            cv2.line(img, (x, y), (int(rng.integers(0, W)), int(rng.integers(0, H))), shade, int(rng.integers(1, 4)))
    return cv2.GaussianBlur(img, (3, 3), 0)


def save(path, img, quality=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if quality else []
    ok, enc = cv2.imencode(os.path.splitext(path)[1], img, params)
    assert ok
    enc.tofile(path)


def build_partition(root):
    """train/val/test with two classes; returns {test file: source file or None}."""
    rows = {"train": [], "val": [], "test": []}
    seed = 0
    originals = {}
    for split, per_class in (("train", 9), ("val", 3)):
        for cls in ("a", "b"):
            for k in range(per_class):
                seed += 1
                name = "%s_%s_%02d.png" % (split, cls, k)
                originals[name] = photo(seed)
                save(os.path.join(root, split, cls, name), originals[name])
                rows[split].append((cls, name))
    truth = {}

    def add_test(cls, name, img, source, quality=None):
        save(os.path.join(root, "test", cls, name), img, quality)
        rows["test"].append((cls, name))
        truth[name] = source

    add_test("a", "exact_copy.png", originals["train_a_03.png"], "train_a_03.png")
    bigger = cv2.resize(originals["train_b_05.png"], (2 * W, 2 * H), interpolation=cv2.INTER_CUBIC)
    add_test("b", "resized_reencoded.jpg", bigger, "train_b_05.png", quality=70)
    M = cv2.getRotationMatrix2D((W / 2, H / 2), 9, 1.0)
    turned = cv2.warpAffine(np.ascontiguousarray(originals["val_a_01.png"][:, ::-1]), M, (W, H), borderValue=0)
    add_test("a", "mirrored_rotated.jpg", turned, "val_a_01.png", quality=85)
    for k in range(5):
        seed += 1
        cls = "ab"[k % 2]
        add_test(cls, "new_%s_%02d.png" % (cls, k), photo(1000 + seed), None)

    for split, items in rows.items():
        with open(os.path.join(root, "%s_manifest.csv" % split), "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["split", "class_name", "label", "filename", "src_path", "dst_path", "hil_subset"])
            for cls, name in items:
                writer.writerow([split, cls, "ab".index(cls), name, "", "",
                                 "1" if name == "exact_copy.png" else ("0" if split == "test" else "")])
    return truth


def run(*argv):
    with contextlib.redirect_stdout(io.StringIO()) as log:
        fnd.main(list(argv))
    return log.getvalue()


FAST = ("--embed", "pixels", "--topk", "3", "--topk_features", "5", "--controls", "8", "--seed", "3")


@unittest.skipUnless(READY, "find_near_duplicates needs numpy and opencv-python")
class DetectorOnKnownPartitionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.root = os.path.join(cls.tmp.name, "splits")
        cls.truth = build_partition(cls.root)
        cls.log = run("--splits_dir", cls.root, *FAST)
        with open(os.path.join(cls.root, "near_duplicates.csv"), encoding="utf-8", newline="") as fh:
            cls.rows = {r["test_file"]: r for r in csv.DictReader(fh)}
        with open(os.path.join(cls.root, "near_duplicates_summary.json"), encoding="utf-8") as fh:
            cls.summary = json.load(fh)

    def test_controls_are_separated_for_both_measures(self):
        cal = self.summary["calibration"]
        self.assertFalse(cal["ncc"]["overlap"], cal["ncc"])
        self.assertFalse(cal["feature_inliers"]["overlap"], cal["feature_inliers"])
        self.assertTrue(self.summary["rule"]["features_used"])

    def test_every_copy_is_found_with_its_source(self):
        for name, source in self.truth.items():
            if source is None:
                continue
            row = self.rows[name]
            self.assertEqual(row["verdict"], "near_duplicate", row)
            self.assertEqual(row["match_file"], source, row)
            self.assertEqual(row["same_label"], "1")
        self.assertEqual(self.rows["exact_copy.png"]["exact"], "1")
        self.assertEqual(self.rows["resized_reencoded.jpg"]["exact"], "0")
        self.assertEqual(self.rows["mirrored_rotated.jpg"]["mirrored"], "1")
        self.assertEqual(self.rows["mirrored_rotated.jpg"]["match_split"], "val")

    def test_new_images_are_not_flagged(self):
        for name, source in self.truth.items():
            if source is None:
                self.assertNotEqual(self.rows[name]["verdict"], "near_duplicate", self.rows[name])

    def test_summary_counts_match_the_table(self):
        s = self.summary
        self.assertEqual((s["n_test"], s["n_trainval"]), (8, 24))
        self.assertEqual(s["exact_pixel_duplicates"], 1)
        self.assertEqual(s["verdicts"]["near_duplicate"], 3)
        self.assertEqual(sum(s["verdicts"].values()), 8)
        self.assertEqual(s["near_duplicate_in_hil_subset"], 1)
        self.assertEqual(s["near_duplicate_match_split"], {"train": 2, "val": 1})
        self.assertEqual(s["near_duplicate_label_mismatch"], 0)
        self.assertEqual(s["parameters"]["region"], "full")

    def test_outputs_hold_names_and_numbers_only(self):
        self.assertEqual(sorted(os.listdir(self.root)),
                         sorted(list(fnd.OUTPUT_NAMES) + ["test", "train", "val", "test_manifest.csv",
                                                          "train_manifest.csv", "val_manifest.csv"]))
        for name in fnd.OUTPUT_NAMES:
            text = Path(self.root, name).read_text(encoding="utf-8")
            self.assertNotIn(self.tmp.name.replace("\\", "/"), text.replace("\\", "/"), name)

    def test_existing_outputs_are_not_replaced_without_overwrite(self):
        before = Path(self.root, "near_duplicates.csv").read_bytes()
        with contextlib.redirect_stdout(io.StringIO()) as log, self.assertRaises(SystemExit) as ctx:
            fnd.main(["--splits_dir", self.root, *FAST])
        self.assertEqual(ctx.exception.code, 1)
        self.assertIn("--overwrite", log.getvalue())
        self.assertEqual(Path(self.root, "near_duplicates.csv").read_bytes(), before)

    def test_same_input_gives_the_same_bytes(self):
        again = os.path.join(self.tmp.name, "again")
        run("--splits_dir", self.root, "--output_dir", again, *FAST)
        for name in fnd.OUTPUT_NAMES:
            self.assertEqual(Path(again, name).read_bytes(), Path(self.root, name).read_bytes(), name)

    def test_center_region_is_an_option(self):
        out = os.path.join(self.tmp.name, "center")
        run("--splits_dir", self.root, "--output_dir", out, "--region", "center", *FAST)
        with open(os.path.join(out, "near_duplicates.csv"), encoding="utf-8", newline="") as fh:
            rows = {r["test_file"]: r for r in csv.DictReader(fh)}
        self.assertEqual(rows["exact_copy.png"]["verdict"], "near_duplicate")
        self.assertEqual(rows["exact_copy.png"]["match_file"], "train_a_03.png")


@unittest.skipUnless(READY, "find_near_duplicates needs numpy and opencv-python")
class DetectorPiecesTest(unittest.TestCase):
    def test_a_partition_that_is_not_on_disk_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            for split in ("train", "val", "test"):
                with open(os.path.join(tmp, "%s_manifest.csv" % split), "w", encoding="utf-8", newline="") as fh:
                    fh.write("split,class_name,label,filename\n%s,a,0,missing.png\n" % split)
            with contextlib.redirect_stdout(io.StringIO()) as log, self.assertRaises(SystemExit) as ctx:
                fnd.main(["--splits_dir", tmp, *FAST])
            self.assertEqual(ctx.exception.code, 1)
            self.assertIn("not on disk", log.getvalue())
            self.assertEqual(sorted(os.listdir(tmp)), ["test_manifest.csv", "train_manifest.csv", "val_manifest.csv"])

    def test_verdict_needs_both_measures_when_features_are_used(self):
        rule = {"ncc": 0.85, "ncc_review": 0.70, "use_features": True, "inliers": 40, "inliers_review": 12}
        self.assertEqual(fnd.verdict_of(0.95, 90, rule), "near_duplicate")
        self.assertEqual(fnd.verdict_of(0.95, 3, rule), "review")       # same layout, nothing in common
        self.assertEqual(fnd.verdict_of(0.40, 60, rule), "review")      # same scene, other viewpoint
        self.assertEqual(fnd.verdict_of(0.75, 0, rule), "review")
        self.assertEqual(fnd.verdict_of(0.40, 5, rule), "distinct")
        self.assertEqual(fnd.verdict_of(float("nan"), 0, rule), "distinct")
        self.assertEqual(fnd.verdict_of(1.0, 0, rule, exact=True), "near_duplicate")   # flat image: no keypoints
        ncc_alone = dict(rule, use_features=False)
        self.assertEqual(fnd.verdict_of(0.95, 0, ncc_alone), "near_duplicate")
        self.assertEqual(fnd.verdict_of(0.75, 99, ncc_alone), "review")
        self.assertEqual(fnd.verdict_of(0.40, 99, ncc_alone), "distinct")

    def test_overlapping_controls_are_reported(self):
        with contextlib.redirect_stdout(io.StringIO()) as log:
            threshold, info = fnd.calibrate([0.5, 0.9, 0.95], [0.2, 0.6, 0.7], None, "ncc")
        self.assertTrue(info["overlap"])
        self.assertIn("overlap", log.getvalue())
        with contextlib.redirect_stdout(io.StringIO()):
            threshold, info = fnd.calibrate([0.9, 0.95, 0.99], [0.2, 0.3, 0.5], None, "ncc")
        self.assertFalse(info["overlap"])
        self.assertTrue(0.5 < threshold < 0.9)

    def test_both_measures_tell_a_copy_from_another_photo(self):
        where = fnd.region_mask("full")
        a, other = photo(501), photo(502)
        copy = fnd.augment(a, np.random.default_rng(7))
        feats = fnd.Features(where)
        described = (feats.describe(a), feats.describe(fnd.mirror(a)))
        self.assertGreater(fnd.align_score(a, copy, where)["ncc"], 0.9)
        self.assertLess(fnd.align_score(a, other, where)["ncc"], 0.8)
        self.assertGreater(feats.score(*described, feats.describe(copy)), 25)
        self.assertLess(feats.score(*described, feats.describe(other)), 12)


import independence_report as report  # noqa: E402


@unittest.skipUnless(report.np is not None, "independence_report needs numpy")
class IndependenceReportTest(unittest.TestCase):
    TABLE = [("t1.jpg", "near_duplicate"), ("t2.jpg", "near_duplicate"), ("t3.jpg", "review"),
             ("t4.jpg", "distinct"), ("t5.jpg", "distinct"), ("t6.jpg", "distinct"), ("t7.jpg", "distinct")]
    # image, true, predicted: the two flagged images are right, one of the others is wrong
    PREDICTIONS = [("t1.jpg", 0, 0), ("t2.jpg", 1, 1), ("t3.jpg", 0, 0), ("t4.jpg", 1, 1), ("t5.jpg", 0, 1),
                   ("t6.jpg", 1, 1)]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.splits = os.path.join(self.tmp.name, "splits_demo")
        self.results = os.path.join(self.tmp.name, "results")
        os.makedirs(self.splits)
        with open(os.path.join(self.splits, "near_duplicates.csv"), "w", encoding="utf-8", newline="") as fh:
            fh.write("test_file,match_file,verdict\n")
            for name, verdict in self.TABLE:
                fh.write("%s,train_%s,%s\n" % (name, name, verdict))
        os.makedirs(os.path.join(self.results, "mil"))
        with open(os.path.join(self.results, "mil", "predictions_Demo.csv"), "w", encoding="utf-8", newline="") as fh:
            fh.write("Image;RelPath;TrueLabel;TrueClass;PredLabel;PredClass;Correct\n")
            for name, true, pred in self.PREDICTIONS:
                fh.write("%s;x/%s;%d;c%d;%d;c%d;%d\n" % (name, name, true, true, pred, pred, int(true == pred)))
            fh.write("t7.jpg;x/t7.jpg;0;c0;;;\n")                # sent to the board, no answer: not counted
        # a result file of another partition kept in the same folder: must be left out
        with open(os.path.join(self.results, "mil", "predictions_Other.csv"), "w", encoding="utf-8", newline="") as fh:
            fh.write("Image;RelPath;TrueLabel;TrueClass;PredLabel;PredClass;Correct\nelsewhere.jpg;x;0;c0;0;c0;1\n")
        self.output = os.path.join(self.tmp.name, "independence.csv")

    def run_report(self, *extra):
        with contextlib.redirect_stdout(io.StringIO()) as log:
            report.main(["--splits_dir", self.splits, "--results_dir", self.results, "--output", self.output,
                         "--bootstrap", "300", *extra])
        with open(self.output, encoding="utf-8", newline="") as fh:
            return {r["cleaning"]: r for r in csv.DictReader(fh)}, log.getvalue()

    def test_scores_match_a_hand_computed_case(self):
        accuracy, macro = report.scores([[2, 1], [0, 3]])        # rows = true label
        self.assertAlmostEqual(float(accuracy), 5 / 6)
        self.assertAlmostEqual(float(macro), (2 * 2 / (3 + 2) + 2 * 3 / (3 + 4)) / 2)
        _, collapsed = report.scores([[5, 0], [5, 0]])           # one class never predicted
        self.assertAlmostEqual(float(collapsed), (2 * 5 / (5 + 10) + 0) / 2)

    def test_both_cleanings_are_reported_and_other_partitions_left_out(self):
        rows, log = self.run_report()
        self.assertEqual(set(rows), {"confirmed", "strict"})
        self.assertIn("1 result files used, 1 left out", log)
        full = rows["confirmed"]
        self.assertEqual((full["level"], full["model"], full["n_full"]), ("MIL", "Demo", "6"))
        self.assertAlmostEqual(float(full["accuracy_full"]), 5 / 6, places=4)
        self.assertEqual((rows["confirmed"]["n_removed"], rows["confirmed"]["n_kept"]), ("2", "4"))
        self.assertAlmostEqual(float(rows["confirmed"]["accuracy_kept"]), 3 / 4, places=4)
        self.assertEqual((rows["strict"]["n_removed"], rows["strict"]["n_kept"]), ("3", "3"))
        self.assertAlmostEqual(float(rows["strict"]["accuracy_kept"]), 2 / 3, places=4)
        self.assertLessEqual(float(rows["strict"]["delta_f1_macro_ci95_low"]), float(rows["strict"]["delta_f1_macro"]))
        self.assertGreaterEqual(float(rows["strict"]["delta_f1_macro_ci95_high"]), float(rows["strict"]["delta_f1_macro"]))

    def test_a_person_s_check_overrides_the_detector(self):
        with open(os.path.join(self.splits, "near_duplicates_review.csv"), "w", encoding="utf-8", newline="") as fh:
            fh.write("test_file,match_file,checked,note\n"
                     "t1.jpg,train_t1.jpg,different,same layout only\n"
                     "t3.jpg,train_t3.jpg,same_scene,taken seconds apart\n")
        rows, _ = self.run_report()
        self.assertEqual(rows["confirmed"]["n_removed"], "2")     # t2 (detector) and t3 (person); t1 is kept
        self.assertEqual(rows["strict"]["n_removed"], "3")        # the strict cleaning never keeps a suspect
        with open(os.path.join(self.splits, "near_duplicates_review.csv"), "a", encoding="utf-8", newline="") as fh:
            fh.write("t5.jpg,train_t5.jpg,maybe,\n")
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as ctx:
            report.main(["--splits_dir", self.splits, "--results_dir", self.results, "--output", self.output,
                         "--overwrite"])
        self.assertEqual(ctx.exception.code, 1)

    def test_existing_report_is_not_replaced_without_overwrite(self):
        self.run_report()
        before = Path(self.output).read_bytes()
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as ctx:
            report.main(["--splits_dir", self.splits, "--results_dir", self.results, "--output", self.output])
        self.assertEqual(ctx.exception.code, 1)
        self.assertEqual(Path(self.output).read_bytes(), before)
        rows, _ = self.run_report("--overwrite")
        self.assertEqual(Path(self.output).read_bytes(), before)  # same seed, same bytes


if __name__ == "__main__":
    unittest.main()
