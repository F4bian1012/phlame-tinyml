#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Checks that the pipeline can be used with a dataset other than the published
ones without colliding with them: the partition takes its classes from the
folders, every run can carry its own name, existing outputs are not replaced
silently, and the label order travels with the model.

Standard library only (no TensorFlow): the split is run on empty files with
image extensions, which is all split_dataset.py needs.

    python -m unittest discover -s tests -v
"""
import argparse
import csv
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import artifacts  # noqa: E402

SPLIT = str(REPO / "src" / "split_dataset.py")


def make_dataset(root, classes, per_class=20):
    for cls in classes:
        folder = Path(root) / cls
        folder.mkdir(parents=True)
        for i in range(per_class):
            (folder / ("img_%03d.jpg" % i)).write_bytes(b"")


def run_split(cwd, *args):
    return subprocess.run([sys.executable, SPLIT, *args], cwd=cwd,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def read_manifest(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def read_report(folder):
    with open(os.path.join(folder, "split_report.json"), encoding="utf-8") as fh:
        return json.load(fh)


class SplitOwnDatasetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cwd = self.tmp.name
        make_dataset(os.path.join(self.cwd, "in"), ["zebra", "apple"])

    def test_classes_come_from_the_folders_in_alphabetical_order(self):
        r = run_split(self.cwd, "--input_dir", "in", "--output_dir", "out", "--hil-subset", "4")
        self.assertEqual(r.returncode, 0, r.stdout)
        report = read_report(os.path.join(self.cwd, "out"))
        self.assertEqual(report["class_names"], ["apple", "zebra"])
        self.assertEqual(report["label_map"], {"apple": 0, "zebra": 1})
        self.assertEqual(report["totals"], {"train": 28, "val": 6, "test": 6, "dataset": 40})

        rows = {sp: read_manifest(os.path.join(self.cwd, "out", "%s_manifest.csv" % sp))
                for sp in ("train", "val", "test")}
        keys = [(r_["class_name"], r_["filename"]) for sp in rows for r_ in rows[sp]]
        self.assertEqual(len(keys), len(set(keys)), "an image appears in two partitions")
        self.assertEqual(sum(r_["hil_subset"] == "1" for r_ in rows["test"]), 4)
        for r_ in rows["test"]:
            self.assertTrue(os.path.isfile(os.path.join(self.cwd, r_["dst_path"])))

    def test_same_seed_gives_the_same_partition(self):
        run_split(self.cwd, "--input_dir", "in", "--output_dir", "a", "--seed", "7")
        run_split(self.cwd, "--input_dir", "in", "--output_dir", "b", "--seed", "7")
        for sp in ("train", "val", "test"):
            a = [(r_["class_name"], r_["filename"], r_["hil_subset"])
                 for r_ in read_manifest(os.path.join(self.cwd, "a", "%s_manifest.csv" % sp))]
            b = [(r_["class_name"], r_["filename"], r_["hil_subset"])
                 for r_ in read_manifest(os.path.join(self.cwd, "b", "%s_manifest.csv" % sp))]
            self.assertEqual(a, b)

    def test_existing_output_is_not_replaced_without_force(self):
        self.assertEqual(run_split(self.cwd, "--input_dir", "in", "--output_dir", "out").returncode, 0)
        again = run_split(self.cwd, "--input_dir", "in", "--output_dir", "out")
        self.assertEqual(again.returncode, 1)
        self.assertIn("already exists", again.stdout)
        self.assertEqual(run_split(self.cwd, "--input_dir", "in", "--output_dir", "out", "--force").returncode, 0)

    def test_explicit_class_file_must_match_the_folders(self):
        names = os.path.join(self.cwd, "names.txt")
        Path(names).write_text("Class1\nClass2\n", encoding="utf-8")
        r = run_split(self.cwd, "--input_dir", "in", "--output_dir", "out", "--class_names_path", names)
        self.assertEqual(r.returncode, 1)
        self.assertIn("Class1", r.stdout)
        self.assertIn("apple", r.stdout)
        self.assertFalse(os.path.exists(os.path.join(self.cwd, "out", "split_report.json")))

    def test_explicit_class_file_sets_the_label_order(self):
        names = os.path.join(self.cwd, "names.txt")
        Path(names).write_text("zebra\napple\n", encoding="utf-8")
        r = run_split(self.cwd, "--input_dir", "in", "--output_dir", "out", "--class_names_path", names)
        self.assertEqual(r.returncode, 0, r.stdout)
        report = read_report(os.path.join(self.cwd, "out"))
        self.assertEqual(report["label_map"], {"zebra": 0, "apple": 1})


class RunNameTest(unittest.TestCase):
    def test_stem_without_tag_is_the_published_name(self):
        self.assertEqual(artifacts.model_stem("MobileNetV2", 32, 20, 0.0001, 160, 120),
                         "MobileNetV2+32+20+0.0001+160+120")

    def test_tag_goes_after_the_architecture(self):
        self.assertEqual(artifacts.model_stem("MobileNetV2", 32, 20, 0.0001, 160, 120, tag="VWW"),
                         "MobileNetV2_VWW+32+20+0.0001+160+120")
        self.assertEqual(artifacts.training_curve_name("ResNet8", 32, 20, 0.0001, 160, 120, tag="mine"),
                         "ResNet8_mine_training_history+32+20+0.0001+160+120.png")

    def test_tag_keeps_the_fields_the_quantizer_reads(self):
        parts = artifacts.model_stem("MobileNetV2", 32, 20, 0.0001, 160, 120, tag="my-data_2").split("+")
        self.assertEqual(len(parts), 6)
        self.assertEqual(parts[-2:], ["160", "120"])

    def test_invalid_tags_are_rejected(self):
        for bad in ("with+plus", "two words", "", "_leading", "ñandú"):
            with self.assertRaises(argparse.ArgumentTypeError, msg=bad):
                artifacts.valid_tag(bad)
        self.assertEqual(artifacts.valid_tag("VWW"), "VWW")


class OverwriteProtectionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.existing = os.path.join(self.tmp.name, "metrics.json")
        Path(self.existing).write_text("{}", encoding="utf-8")
        self.missing = os.path.join(self.tmp.name, "other.json")

    def test_stops_when_an_output_exists(self):
        out = io.StringIO()
        with redirect_stdout(out), self.assertRaises(SystemExit) as ctx:
            artifacts.refuse_to_overwrite([self.missing, self.existing], overwrite=False, hint="use --tag")
        self.assertEqual(ctx.exception.code, 1)
        self.assertIn(self.existing, out.getvalue())
        self.assertIn("--overwrite", out.getvalue())
        self.assertEqual(Path(self.existing).read_text(encoding="utf-8"), "{}")

    def test_passes_with_overwrite_or_when_nothing_exists(self):
        artifacts.refuse_to_overwrite([self.existing], overwrite=True)
        artifacts.refuse_to_overwrite([self.missing], overwrite=False)


class ClassNamesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.keras = os.path.join(self.tmp.name, "MobileNetV2_demo+32+20+0.0001+160+120.keras")
        self.tflite = os.path.join(self.tmp.name, "MobileNetV2_demo+32+20+0.0001+160+120_int8.tflite")

    def test_label_order_travels_with_the_model(self):
        written = artifacts.write_class_names(self.keras, ["non_person", "person"])
        self.assertEqual(written, artifacts.classes_path_for(self.keras))
        copied = artifacts.copy_class_names(self.keras, self.tflite)
        self.assertEqual(Path(copied).read_text(encoding="utf-8"), "non_person\nperson\n")
        self.assertEqual(artifacts.resolve_class_names_path(self.tflite), copied)

    def test_resolution_order(self):
        explicit = os.path.join(self.tmp.name, "explicit.txt")
        self.assertEqual(artifacts.resolve_class_names_path(self.keras, explicit), explicit)
        # a model without its own file (e.g. the published ones) uses the repository-wide list
        self.assertEqual(artifacts.resolve_class_names_path(self.keras), artifacts.LEGACY_CLASS_NAMES)
        self.assertIsNone(artifacts.copy_class_names(self.keras, self.tflite))


class PublishedPartitionsTest(unittest.TestCase):
    """The published partitions sit in named folders; data/splits is left free for the user."""

    def test_default_split_folder_holds_no_versioned_file(self):
        tracked = subprocess.run(["git", "ls-files", "data/splits"], cwd=REPO,
                                 stdout=subprocess.PIPE, text=True).stdout.split()
        self.assertEqual(tracked, [])

    def test_manifests_point_inside_their_own_folder(self):
        for name in ("splits_plates", "splits_vww"):
            folder = REPO / "data" / name
            for sp in ("train", "val", "test"):
                rows = read_manifest(folder / ("%s_manifest.csv" % sp))
                self.assertTrue(rows, "%s/%s_manifest.csv is empty" % (name, sp))
                for r_ in rows:
                    self.assertTrue(r_["dst_path"].startswith("data/%s/%s/" % (name, sp)), r_["dst_path"])


if __name__ == "__main__":
    unittest.main()
