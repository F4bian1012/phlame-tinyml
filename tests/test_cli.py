#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Checks the command-line conventions shared by the scripts: every multi-word
option is accepted with '_' or '-', and the four ladder benches expose the
same options for the same things.

Standard library only (no TensorFlow, no board).

    python -m unittest discover -s tests -v
"""
import ast
import io
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
sys.path.insert(0, str(SRC))

import cli  # noqa: E402


def make_parser():
    parser = cli.ArgumentParser()
    parser.add_argument("--model_path")
    parser.add_argument("--output-dir", default="out")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--count", type=int)
    parser.add_argument("image", nargs="?")
    return parser


class EitherSpellingTest(unittest.TestCase):
    def test_underscore_and_hyphen_are_the_same_option(self):
        for argv in (["--model_path", "m.tflite", "--output-dir", "x", "--dry-run"],
                     ["--model-path", "m.tflite", "--output_dir", "x", "--dry_run"]):
            args = make_parser().parse_args(argv)
            self.assertEqual((args.model_path, args.output_dir, args.dry_run), ("m.tflite", "x", True))

    def test_equals_form(self):
        args = make_parser().parse_args(["--model-path=a_b-c.tflite", "--output_dir=res/x-y_z"])
        self.assertEqual(args.model_path, "a_b-c.tflite")
        self.assertEqual(args.output_dir, "res/x-y_z")

    def test_values_are_left_alone(self):
        args = make_parser().parse_args(["--model_path", "my-model_v2.tflite", "some-image_1.png"])
        self.assertEqual(args.model_path, "my-model_v2.tflite")
        self.assertEqual(args.image, "some-image_1.png")

    def test_unknown_options_still_fail(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            make_parser().parse_args(["--no-such-option", "1"])

    def test_defaults_and_single_word_options_unchanged(self):
        args = make_parser().parse_args(["--count", "3"])
        self.assertEqual((args.count, args.output_dir, args.dry_run), (3, "out", False))


def declared_options(script):
    """Option strings a script passes to add_argument, read from its source (no import)."""
    tree = ast.parse((SRC / script).read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument":
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("--"):
                    found.add(arg.value.replace("-", "_").strip("_"))
    return found


class LadderOptionsTest(unittest.TestCase):
    BENCHES = ("mil_benchmark.py", "sil_benchmark.py", "pil_benchmark.py", "hil_benchmark.py")

    def test_every_script_uses_the_shared_parser(self):
        # train_common.py only adds options to the parser a training script hands it.
        for script in sorted(SRC.glob("*.py")):
            text = script.read_text(encoding="utf-8")
            if "add_argument(" in text and script.name not in ("cli.py", "train_common.py"):
                self.assertIn("cli.ArgumentParser(", text, script.name)
                self.assertNotIn("argparse.ArgumentParser(", text, script.name)

    def test_the_four_benches_share_their_common_options(self):
        for script in self.BENCHES:
            options = declared_options(script)
            for common in ("model_path", "splits_dir", "output_dir", "overwrite"):
                self.assertIn(common, options, "%s lacks --%s" % (script, common))

    def test_the_two_board_benches_share_the_serial_options(self):
        for script in ("pil_benchmark.py", "hil_benchmark.py"):
            options = declared_options(script)
            for common in ("port", "baud", "count", "delay", "gap", "folder"):
                self.assertIn(common, options, "%s lacks --%s" % (script, common))


class TrainingOptionsTest(unittest.TestCase):
    def test_the_three_training_scripts_share_their_run_options(self):
        for script in ("train_mobilenet.py", "train_resnet.py", "train_squeezenet.py"):
            options = declared_options(script)
            for common in ("seed", "tag", "overwrite", "splits_dir", "epochs", "learning_rate"):
                self.assertIn(common, options, "%s lacks --%s" % (script, common))

    def test_the_recipe_options_are_declared_once_for_the_three(self):
        self.assertTrue({"lr_schedule", "augmentation"} <= declared_options("train_common.py"))
        for script in ("train_mobilenet.py", "train_resnet.py", "train_squeezenet.py"):
            self.assertIn("train_common.add_recipe_arguments(parser)",
                          (SRC / script).read_text(encoding="utf-8"), script)


class FailureExitCodeTest(unittest.TestCase):
    """A step that reports an error must not end with exit code 0, or a chained
    pipeline would carry on with the files of an earlier run."""

    def run_script(self, script, *args):
        with tempfile.TemporaryDirectory() as cwd:
            return subprocess.run([sys.executable, str(SRC / script), *args], cwd=cwd,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def test_header_converter_fails_on_a_missing_model(self):
        r = self.run_script("tflite_to_c.py", "no_such_model.tflite")
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("does not exist", r.stdout)

    def test_scripts_that_report_an_error_exit_with_it(self):
        # Read from the source, without importing TensorFlow: after printing an
        # error these scripts used to return normally.
        for script in ("quantize_int8_basic.py", "mil_benchmark.py", "sil_benchmark.py",
                       "train_mobilenet.py", "train_resnet.py", "train_squeezenet.py", "prune_model.py"):
            lines = (SRC / script).read_text(encoding="utf-8").splitlines()
            for i, line in enumerate(lines[:-1]):
                if "print(" in line and "rror" in line and lines[i + 1].strip() in ("return", "return None"):
                    self.fail("%s:%d prints an error and returns normally" % (script, i + 1))


if __name__ == "__main__":
    unittest.main()
