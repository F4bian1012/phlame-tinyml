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
import sys
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
        for script in sorted(SRC.glob("*.py")):
            text = script.read_text(encoding="utf-8")
            if "add_argument(" in text and script.name != "cli.py":
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


if __name__ == "__main__":
    unittest.main()
