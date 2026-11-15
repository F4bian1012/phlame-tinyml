#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two things about models/checksums.sha256 and the result files it is checked against.

Which result files count. A user who follows the usage steps of the README
writes the results of their own model next to the published ones. Those runs
are not published results: in a git checkout the checks look at the files git
tracks, so they keep passing.

Who writes the list. `verify_models.py --update_list --set <set>` rewrites the
entries of one set from the model files its published results name, instead of
the list being edited by hand after a campaign.

Standard library and git only.

    python -m unittest discover -s tests -v
"""
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = str(REPO / "src" / "verify_models.py")
sys.path.insert(0, str(REPO / "src"))

import verify_models  # noqa: E402

HEADER = "# SHA-256 of every model file behind a published result (sha256sum format).\n"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_result(root, level, run, model_rel, data):
    path = Path(root, "results", level, "metrics_%s.json" % run)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"level": level.upper(), "model": {"path": model_rel, "sha256": digest(data)}}),
                    encoding="utf-8")
    return path


def write_model(root, rel, data):
    path = Path(root, rel)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def run_script(root, *args):
    return subprocess.run([sys.executable, SCRIPT, "--root", str(root), *args],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")


class Tree(unittest.TestCase):
    """A small repository-like tree: two COCO models with results, one plates model, and a list that
    still names an older COCO model."""

    KERAS = "models/checkpoints/Tiny_COCO+1+2+0.1+8+8.keras"
    TFLITE = "models/tflite/Tiny_COCO+1+2+0.1+8+8_int8.tflite"
    OLD = "models/checkpoints/Tiny_COCO+1+1+0.1+8+8.keras"
    PLATES = "models/tflite/Tiny+1+1+0.1+8+8_int8.tflite"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        write_model(self.root, self.KERAS, b"new float model")
        write_model(self.root, self.TFLITE, b"new int8 model")
        write_model(self.root, self.PLATES, b"plates int8 model")
        write_result(self.root, "mil", "Tiny_COCO+1+2+0.1+8+8", self.KERAS, b"new float model")
        write_result(self.root, "sil", "Tiny_COCO+1+2+0.1+8+8_int8", self.TFLITE, b"new int8 model")
        write_result(self.root, "pil", "Tiny_COCO+1+2+0.1+8+8_int8", self.TFLITE, b"new int8 model")
        write_result(self.root, "sil", "Tiny+1+1+0.1+8+8_int8", self.PLATES, b"plates int8 model")
        self.list = self.root / "models" / "checksums.sha256"
        self.list.write_text(HEADER + "%s  %s\n%s  %s\n" % (digest(b"old float model"), self.OLD,
                                                            digest(b"plates int8 model"), self.PLATES),
                             encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def entries(self):
        return verify_models.read_manifest(str(self.list))


class UpdateListTest(Tree):
    def test_the_set_is_rewritten_from_its_results(self):
        result = run_script(self.root, "--update_list", "--set", "coco")
        self.assertEqual(result.returncode, 0, result.stdout)
        entries = self.entries()
        self.assertEqual(list(entries), sorted(entries))
        self.assertEqual(entries, {self.KERAS: digest(b"new float model"), self.TFLITE: digest(b"new int8 model"),
                                   self.PLATES: digest(b"plates int8 model")})
        self.assertIn("DROPPED  " + self.OLD, result.stdout)
        self.assertTrue(self.list.read_text(encoding="utf-8").startswith(HEADER))
        self.assertEqual(run_script(self.root, "--results").returncode, 0)
        self.assertEqual(run_script(self.root, "--require_all").returncode, 0)

    def test_a_model_file_that_is_not_the_measured_one_stops_it(self):
        write_model(self.root, self.TFLITE, b"another file under the same name")
        before = self.list.read_text(encoding="utf-8")
        result = run_script(self.root, "--update_list", "--set", "coco")
        self.assertEqual(result.returncode, 1)
        self.assertIn("is not the file its results were obtained with", result.stdout)
        self.assertEqual(self.list.read_text(encoding="utf-8"), before)

    def test_it_needs_one_set_and_results_of_it(self):
        self.assertEqual(run_script(self.root, "--update_list").returncode, 2)            # --set all
        before = self.list.read_text(encoding="utf-8")
        result = run_script(self.root, "--update_list", "--set", "vww")
        self.assertEqual(result.returncode, 1)
        self.assertIn("no published result names a model of set 'vww'", result.stdout)
        self.assertEqual(self.list.read_text(encoding="utf-8"), before)


@unittest.skipIf(shutil.which("git") is None, "needs git")
class OwnRunsTest(Tree):
    """In a git checkout, a run that git does not track is the user's own."""

    def git(self, *args):
        subprocess.run(["git", "-C", str(self.root), "-c", "user.name=t", "-c", "user.email=t@example.org", *args],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def setUp(self):
        super().setUp()
        self.assertEqual(run_script(self.root, "--update_list", "--set", "coco").returncode, 0)
        self.git("init", "-q")
        self.git("add", "results", "models/checksums.sha256")
        self.git("commit", "-q", "-m", "published")

    def test_an_untracked_run_of_another_model_is_left_out(self):
        self.assertEqual(run_script(self.root, "--results").returncode, 0)
        write_result(self.root, "mil", "Mine_demo+1+1+0.1+8+8", "models/checkpoints/Mine_demo+1+1+0.1+8+8.keras",
                     b"my own model")
        result = run_script(self.root, "--results")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertNotIn("NOT LISTED", result.stdout)
        found = sorted(ref[0] for ref in verify_models.result_references(str(self.root)))
        self.assertNotIn("results/mil/metrics_Mine_demo+1+1+0.1+8+8.json", found)
        self.assertEqual(len(found), 4)

    def test_once_git_tracks_it_the_run_is_checked(self):
        write_result(self.root, "mil", "Mine_demo+1+1+0.1+8+8", "models/checkpoints/Mine_demo+1+1+0.1+8+8.keras",
                     b"my own model")
        self.git("add", "results")
        result = run_script(self.root, "--results")
        self.assertEqual(result.returncode, 1)
        self.assertIn("NOT LISTED", result.stdout)


if __name__ == "__main__":
    unittest.main()
