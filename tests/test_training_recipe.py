#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Checks the part of the training recipe that can be checked without training:
the learning rate of every epoch, what the training record says about a run,
and that the three training scripts offer the recipe options and keep the
previous behaviour as their default.

train_common.py imports TensorFlow only when a training starts, so this runs
in a fraction of a second.

    python -m unittest discover -s tests -v
"""
import ast
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
sys.path.insert(0, str(SRC))

import train_common  # noqa: E402

TRAINERS = ("train_mobilenet.py", "train_resnet.py", "train_squeezenet.py")


class LearningRateTest(unittest.TestCase):
    def test_tensorflow_is_not_needed_to_describe_a_recipe(self):
        self.assertNotIn("tensorflow", sys.modules)

    def test_stepped_schedule_is_the_one_of_mlperf_tiny(self):
        # train_vww.py: 20 epochs at 1e-3, 10 at 5e-4, 20 at 2.5e-4
        rates = train_common.learning_rates("stepped", 0.001, 50)
        self.assertEqual(rates, [0.001] * 20 + [0.0005] * 10 + [0.00025] * 20)

    def test_stepped_schedule_keeps_its_proportions_at_other_lengths(self):
        rates = train_common.learning_rates("stepped", 0.01, 10)
        self.assertEqual(rates, [0.01] * 4 + [0.005] * 2 + [0.0025] * 4)
        self.assertEqual(train_common.learning_rates("stepped", 0.01, 1), [0.01])

    def test_constant_schedule(self):
        self.assertEqual(train_common.learning_rates("constant", 0.0001, 3), [0.0001] * 3)


class TrainingRecordTest(unittest.TestCase):
    def test_default_recipe_is_described_as_it_always_ran(self):
        s = train_common.recipe_settings("constant", 0.0001, 20, "standard", {"rotation_fraction_of_turn": 0.2})
        self.assertEqual(s["lr_schedule"], "constant")
        self.assertEqual(s["learning_rates"], [0.0001])
        self.assertIn("patience 5", s["early_stopping"])
        self.assertIsNone(s["reduce_lr_on_plateau"])
        self.assertEqual(s["augmentation_parameters"], {"rotation_fraction_of_turn": 0.2})

    def test_literature_recipe_is_recorded_in_full(self):
        s = train_common.recipe_settings("stepped", 0.001, 50, "light", {"unused": 1}, plateau=True)
        self.assertEqual(s["learning_rates"], [0.001, 0.0005, 0.00025])
        self.assertIsNone(s["early_stopping"])
        self.assertIsNone(s["reduce_lr_on_plateau"])
        self.assertEqual(s["augmentation_parameters"],
                         {"horizontal_flip": True, "rotation_degrees": 10, "shift_fraction": 0.05,
                          "zoom_fraction": 0.1})

    def test_plateau_rule_is_recorded_only_when_it_applies(self):
        s = train_common.recipe_settings("constant", 0.0001, 30, "standard", {}, plateau=True)
        self.assertIn("factor 0.5", s["reduce_lr_on_plateau"])


def keyword_defaults(script, function):
    """{parameter: default} of a function, read from the source (no import)."""
    tree = ast.parse((SRC / script).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function:
            names = [a.arg for a in node.args.args]
            defaults = [ast.literal_eval(d) for d in node.args.defaults]
            return dict(zip(names[len(names) - len(defaults):], defaults))
    raise AssertionError("%s has no function %s" % (script, function))


RECIPE_OPTIONS = ("--kernel_initializer", "--batch_norm", "--trainable_backbone", "--weights")


def option_defaults(script):
    """{option: default} of the recipe options a script declares by itself, read from the
    source. A flag (action='store_true', off unless given) is reported as 'store_true'."""
    found = {}
    for node in ast.walk(ast.parse((SRC / script).read_text(encoding="utf-8"))):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument" and node.args):
            continue
        name = ast.literal_eval(node.args[0])
        if name in RECIPE_OPTIONS:
            keywords = {k.arg: ast.literal_eval(k.value) for k in node.keywords if k.arg in ("default", "action")}
            found[name] = keywords.get("action") or keywords.get("default")
    return found


class DefaultsTest(unittest.TestCase):
    """Without options a training script must train as it did before the recipe options existed."""

    def test_recipe_defaults_of_the_three_scripts(self):
        for script in TRAINERS:
            d = keyword_defaults(script, "main")
            self.assertEqual((d["lr_schedule"], d["augmentation"]), ("constant", "standard"), script)
        d = keyword_defaults("train_mobilenet.py", "main")
        self.assertEqual((d["weights"], d["trainable_backbone"]), ("imagenet", False))

    def test_squeezenet_keeps_the_network_without_batch_normalisation_by_default(self):
        for function in ("build_squeezenet", "create_squeezenet_model", "main", "conv_relu", "fire_module"):
            d = keyword_defaults("train_squeezenet.py", function)
            self.assertEqual((d["kernel_initializer"], d["batch_norm"]), ("glorot_uniform", False), function)

    def test_squeezenet_options_are_off_on_the_command_line(self):
        self.assertEqual(option_defaults("train_squeezenet.py"),
                         {"--kernel_initializer": "glorot_uniform", "--batch_norm": "store_true"})

    def test_mobilenet_options_are_off_on_the_command_line(self):
        self.assertEqual(option_defaults("train_mobilenet.py"),
                         {"--trainable_backbone": "store_true", "--weights": "imagenet"})

    def test_command_line_defaults(self):
        import argparse
        parser = argparse.ArgumentParser()
        train_common.add_recipe_arguments(parser)
        args = parser.parse_args([])
        self.assertEqual((args.lr_schedule, args.augmentation), ("constant", "standard"))
        args = parser.parse_args(["--lr_schedule", "stepped", "--augmentation", "light"])
        self.assertEqual((args.lr_schedule, args.augmentation), ("stepped", "light"))


if __name__ == "__main__":
    unittest.main()
