#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Checks that the scripts that use TensorFlow read their command line before
they import it: --help and a wrong option must answer at once, not after the
several seconds TensorFlow takes to load.

Each script is run in a child process whose `tensorflow` is a stand-in that
fails when imported, so the test does not depend on timing and runs the same
with or without TensorFlow installed. It also checks that asking a training
script for its help creates nothing in the working directory.

Standard library only.

    python -m unittest discover -s tests -v
"""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"

# Scripts that use TensorFlow, and an option each one declares.
SCRIPTS = {
    "mil_benchmark.py": "--model_path",
    "sil_benchmark.py": "--kernels",
    "quantize_int8_basic.py": "--calib_samples",
    "train_mobilenet.py": "--base_model",
    "train_resnet.py": "--dropout_rate",
    "train_squeezenet.py": "--batch_norm",
    "prune_model.py": "--sparsity",
}
TRAINERS = ("train_mobilenet.py", "train_resnet.py", "train_squeezenet.py")
# Options a script requires: without them argparse complains about those first.
REQUIRED = {"mil_benchmark.py": ("--model_path", "m.keras"), "sil_benchmark.py": ("--model_path", "m.tflite"),
            "prune_model.py": ("--model_path", "m.keras", "--sparsity", "0.5", "--technique", "global")}
MARK = "TENSORFLOW-WAS-IMPORTED"


class HelpWithoutTensorFlowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        root = Path(cls.tmp.name)
        # A package named tensorflow that comes before the installed one and fails on import.
        (root / "standin" / "tensorflow").mkdir(parents=True)
        (root / "standin" / "tensorflow" / "__init__.py").write_text(
            'raise ImportError("%s")\n' % MARK, encoding="utf-8")
        cls.env = dict(os.environ, PYTHONPATH=str(root / "standin"))
        cls.root = root

    def run_script(self, script, *args):
        cwd = self.root / ("cwd_%s_%d" % (script[:-3], len(list(self.root.iterdir()))))
        cwd.mkdir()
        result = subprocess.run([sys.executable, str(SRC / script), *args], cwd=cwd, env=self.env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return result, sorted(p.name for p in cwd.iterdir())

    def test_the_stand_in_stops_a_process_that_imports_tensorflow(self):
        result = subprocess.run([sys.executable, "-c", "import tensorflow"], env=self.env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(MARK, result.stdout)

    def test_help_does_not_import_tensorflow(self):
        for script, option in SCRIPTS.items():
            with self.subTest(script=script):
                result, _ = self.run_script(script, "--help")
                self.assertEqual(result.returncode, 0, result.stdout[-1500:])
                self.assertNotIn(MARK, result.stdout)
                self.assertIn("usage:", result.stdout)
                self.assertIn(option, result.stdout)

    def test_a_wrong_option_is_refused_without_importing_tensorflow(self):
        for script in SCRIPTS:
            with self.subTest(script=script):
                result, _ = self.run_script(script, *REQUIRED.get(script, ()), "--no_such_option")
                self.assertEqual(result.returncode, 2, result.stdout[-1500:])
                self.assertNotIn(MARK, result.stdout)
                self.assertIn("--no_such_option", result.stdout)

    def test_a_wrong_value_is_refused_without_importing_tensorflow(self):
        result, _ = self.run_script("sil_benchmark.py", "--model_path", "m.tflite", "--kernels", "fastest")
        self.assertEqual(result.returncode, 2, result.stdout[-1500:])
        self.assertNotIn(MARK, result.stdout)
        self.assertIn("invalid choice", result.stdout)

    def test_help_of_a_training_script_creates_nothing(self):
        for script in TRAINERS:
            with self.subTest(script=script):
                result, left = self.run_script(script, "--help")
                self.assertEqual(result.returncode, 0, result.stdout[-1500:])
                self.assertEqual(left, [], "%s --help left files in the working directory" % script)

    def test_a_run_still_reaches_tensorflow(self):
        # With a complete command line the script goes on and imports TensorFlow:
        # the stand-in stops it there, which shows the import was only moved, not lost.
        for script, args in (("mil_benchmark.py", ("--model_path", "no_such_model.keras")),
                             ("quantize_int8_basic.py", ("--model_path", "no_such_model.keras")),
                             ("train_resnet.py", ("--epochs", "1"))):
            with self.subTest(script=script):
                result, _ = self.run_script(script, *args)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(MARK, result.stdout)


if __name__ == "__main__":
    unittest.main()
