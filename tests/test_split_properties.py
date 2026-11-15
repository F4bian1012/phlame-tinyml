#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Checks properties of the partition (src/split_dataset.py) that
tests/test_own_dataset.py does not cover: the split is stratified class by
class, the subset marked for the camera rig is drawn from the test partition
in proportion to its classes, every source image is copied once and unchanged,
and a request that makes no sense writes nothing.

The expected counts are worked out by hand in the comments. Standard library
only: the "images" are small files with distinct contents.

    python -m unittest discover -s tests -v
"""
import collections
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SPLIT = str(REPO / "src" / "split_dataset.py")

# Three classes of different sizes, none of which gives a count that ends in .5
SIZES = {"apple": 40, "pear": 20, "zebra": 60}
# train 0.70 / val 0.15 / test 0.15, class by class:
#   apple 40 -> 28 / 6 / 6      pear 20 -> 14 / 3 / 3      zebra 60 -> 42 / 9 / 9
EXPECTED = {"train": {"apple": 28, "pear": 14, "zebra": 42},
            "val": {"apple": 6, "pear": 3, "zebra": 9},
            "test": {"apple": 6, "pear": 3, "zebra": 9}}


def make_dataset(root):
    for cls, n in SIZES.items():
        folder = Path(root) / cls
        folder.mkdir(parents=True)
        for i in range(n):
            (folder / ("img_%03d.jpg" % i)).write_bytes(("%s image %d" % (cls, i)).encode())


def run_split(cwd, *args):
    return subprocess.run([sys.executable, SPLIT, *args], cwd=cwd,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def manifests(folder):
    rows = {}
    for split in ("train", "val", "test"):
        with open(os.path.join(folder, "%s_manifest.csv" % split), encoding="utf-8", newline="") as fh:
            rows[split] = list(csv.DictReader(fh))
    return rows


def per_class(rows, only_rig=False):
    return dict(collections.Counter(r["class_name"] for r in rows if not only_rig or r["hil_subset"] == "1"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class StratifiedSplitTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cwd = self.tmp.name
        make_dataset(os.path.join(self.cwd, "in"))

    def split(self, out, *extra):
        result = run_split(self.cwd, "--input_dir", "in", "--output_dir", out, *extra)
        self.assertEqual(result.returncode, 0, result.stdout)
        return manifests(os.path.join(self.cwd, out))

    def test_every_class_is_split_in_the_same_proportions(self):
        rows = self.split("out", "--hil_subset", "0")
        for split in ("train", "val", "test"):
            self.assertEqual(per_class(rows[split]), EXPECTED[split], split)
        with open(os.path.join(self.cwd, "out", "split_report.json"), encoding="utf-8") as fh:
            report = json.load(fh)
        self.assertEqual(report["totals"], {"train": 84, "val": 18, "test": 18, "dataset": 120})
        # labels follow the alphabetical order of the folders
        self.assertEqual({r["class_name"]: r["label"] for split in rows for r in rows[split]},
                         {"apple": "0", "pear": "1", "zebra": "2"})

    def test_rig_subset_is_drawn_from_the_test_partition_in_proportion(self):
        # 6 of the 18 test images, by class 6 : 3 : 9 -> 2 : 1 : 3
        rows = self.split("out", "--hil_subset", "6")
        self.assertEqual(per_class(rows["test"], only_rig=True), {"apple": 2, "pear": 1, "zebra": 3})
        for split in ("train", "val"):
            self.assertEqual(per_class(rows[split], only_rig=True), {}, split)

    def test_rig_subset_larger_than_the_test_partition_takes_all_of_it_and_zero_takes_none(self):
        rows = self.split("all", "--hil_subset", "500")
        self.assertEqual(per_class(rows["test"], only_rig=True), EXPECTED["test"])
        rows = self.split("none", "--hil_subset", "0")
        self.assertEqual(per_class(rows["test"], only_rig=True), {})

    def test_rig_subset_does_not_change_which_images_go_where(self):
        with_rig = self.split("a", "--hil_subset", "6")
        without = self.split("b", "--hil_subset", "0")
        for split in ("train", "val", "test"):
            self.assertEqual([(r["class_name"], r["filename"]) for r in with_rig[split]],
                             [(r["class_name"], r["filename"]) for r in without[split]], split)

    def test_every_image_is_copied_once_and_unchanged(self):
        rows = self.split("out")
        seen = []
        for split in rows:
            for row in rows[split]:
                source = os.path.join(self.cwd, row["src_path"])
                copy = os.path.join(self.cwd, row["dst_path"])
                self.assertEqual(digest(copy), digest(source), row["dst_path"])
                self.assertEqual(Path(row["dst_path"]).parts[-3:-1], (split, row["class_name"]))
                seen.append(row["src_path"])
        expected = sorted("in/%s/img_%03d.jpg" % (cls, i) for cls, n in SIZES.items() for i in range(n))
        self.assertEqual(sorted(seen), expected)
        # and nothing else is in the folders of the partition
        copied = [p for p in (Path(self.cwd) / "out").rglob("*.jpg")]
        self.assertEqual(len(copied), 120)

    def test_another_seed_gives_another_partition_of_the_same_sizes(self):
        first = self.split("a", "--seed", "1")
        other = self.split("b", "--seed", "2")
        for split in ("train", "val", "test"):
            self.assertEqual(per_class(first[split]), per_class(other[split]))
        self.assertNotEqual(sorted((r["class_name"], r["filename"]) for r in first["test"]),
                            sorted((r["class_name"], r["filename"]) for r in other["test"]))


class RefusedSplitTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cwd = self.tmp.name
        make_dataset(os.path.join(self.cwd, "in"))

    def test_fractions_that_do_not_sum_to_one_are_refused(self):
        result = run_split(self.cwd, "--input_dir", "in", "--output_dir", "out",
                           "--train", "0.7", "--val", "0.2", "--test", "0.2")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must sum to 1.0", result.stdout)
        self.assertFalse(os.path.exists(os.path.join(self.cwd, "out")))

    def test_missing_input_folder_is_refused(self):
        result = run_split(self.cwd, "--input_dir", "nowhere", "--output_dir", "out")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not exist", result.stdout)
        self.assertFalse(os.path.exists(os.path.join(self.cwd, "out")))

    def test_a_class_without_images_is_refused(self):
        (Path(self.cwd) / "in" / "empty").mkdir()
        result = run_split(self.cwd, "--input_dir", "in", "--output_dir", "out")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("contains no images", result.stdout)
        self.assertFalse(os.path.exists(os.path.join(self.cwd, "out", "split_report.json")))

    def test_dry_run_reports_and_writes_nothing(self):
        result = run_split(self.cwd, "--input_dir", "in", "--output_dir", "out", "--dry_run")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("nothing was written", result.stdout)
        self.assertEqual(list((Path(self.cwd) / "out").rglob("*")) if (Path(self.cwd) / "out").exists() else [], [])


if __name__ == "__main__":
    unittest.main()
