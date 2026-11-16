"""prune_model.py: what it refuses before loading TensorFlow, and that it prunes the
convolutions of a nested model.

    python -m unittest tests.test_prune_model -v

The checks of the command line run without TensorFlow: the script loads it only
after them. The pruning itself needs TensorFlow and is skipped without it; it
runs in a child process, so TensorFlow is never loaded into the test process.
The model is a few layers with random weights and nothing is trained.
"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
SCRIPT = str(SRC / "prune_model.py")
HAVE_TF = importlib.util.find_spec("tensorflow") is not None
ENV = dict(os.environ, TF_CPP_MIN_LOG_LEVEL="3", PYTHONIOENCODING="utf-8")


def run(*args, cwd=None):
    return subprocess.run([sys.executable, SCRIPT, *args], cwd=cwd, env=ENV, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, encoding="utf-8")


class CommandLineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.model = os.path.join(self.tmp.name, "Tiny_demo+1+1+0.1+8+8.keras")
        Path(self.model).write_bytes(b"not a model: the script must stop before reading it")

    def test_missing_model_ends_with_exit_code_1(self):
        result = run("--model_path", os.path.join(self.tmp.name, "absent.keras"), "--sparsity", "0.5",
                     "--technique", "global")
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("does not exist", result.stdout)

    def test_sparsity_out_of_range_ends_with_exit_code_1(self):
        result = run("--model_path", self.model, "--sparsity", "1.0", "--technique", "layer")
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("range [0.0, 1.0)", result.stdout)

    def test_an_existing_pruned_model_is_not_replaced(self):
        pruned = Path(self.tmp.name, "Tiny_demo+1+1+0.1+8+8_pruned_global_0.5.keras")
        pruned.write_bytes(b"an earlier pruned model")
        result = run("--model_path", self.model, "--sparsity", "0.5", "--technique", "global")
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("would replace existing files", result.stdout)
        self.assertIn(pruned.name, result.stdout)
        self.assertEqual(pruned.read_bytes(), b"an earlier pruned model")

    def test_the_checks_do_not_load_tensorflow(self):
        # Importing the module and parsing a command line must not import TensorFlow.
        code = ("import sys; sys.path.insert(0, %r); import prune_model; "
                "prune_model.parse_args(['--model_path', 'x', '--sparsity', '0.5', '--technique', 'global']); "
                "print('tensorflow' in sys.modules)" % str(SRC))
        result = subprocess.run([sys.executable, "-c", code], env=ENV, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8")
        self.assertEqual(result.stdout.strip().splitlines()[-1], "False", result.stdout)


# Builds a model whose backbone is a nested model, as the MobileNets of train_mobilenet.py are,
# prunes it both ways and through the command line, and prints what happened as one JSON line.
CHILD = r"""
import json, os, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
root = sys.argv[2]
import tensorflow as tf
import prune_model

def build():
    tf.keras.utils.set_random_seed(7)
    inner_in = tf.keras.Input((8, 8, 1))
    x = tf.keras.layers.Conv2D(4, 3, padding="same", name="inner_conv_a")(inner_in)
    x = tf.keras.layers.DepthwiseConv2D(3, padding="same", name="inner_depthwise")(x)
    x = tf.keras.layers.Conv2D(8, 1, name="inner_conv_b")(x)
    backbone = tf.keras.Model(inner_in, x, name="backbone")
    outer_in = tf.keras.Input((8, 8, 1))
    y = backbone(outer_in)
    y = tf.keras.layers.GlobalAveragePooling2D()(y)
    y = tf.keras.layers.Dense(2, activation="softmax", name="head")(y)
    return tf.keras.Model(outer_in, y)

def zeros(model, name):
    layer = model.get_layer("backbone").get_layer(name) if name.startswith("inner") else model.get_layer(name)
    w = layer.get_weights()[0]
    return float(np.mean(w == 0))

out = {}
model = build()
out["names"] = [layer.name for layer in prune_model.prunable_layers(model)]
out["before"] = prune_model.calculate_sparsity(model)
biases = [w.copy() for layer in prune_model.prunable_layers(model) for w in layer.get_weights()[1:]]
depthwise = model.get_layer("backbone").get_layer("inner_depthwise").get_weights()[0].copy()

prune_model.apply_global_unstructured_pruning(model, 0.5)
out["global"] = prune_model.calculate_sparsity(model)
out["global_inner"] = zeros(model, "inner_conv_a")
out["biases_kept"] = all(np.array_equal(a, b) for a, b in zip(
    biases, [w for layer in prune_model.prunable_layers(model) for w in layer.get_weights()[1:]]))
out["depthwise_kept"] = bool(np.array_equal(
    depthwise, model.get_layer("backbone").get_layer("inner_depthwise").get_weights()[0]))

model = build()
prune_model.apply_layer_unstructured_pruning(model, 0.25)
out["layer"] = {name: zeros(model, name) for name in ("inner_conv_a", "inner_conv_b", "head")}

# through the command line: the file it writes, and the model that file holds
model = build()
path = os.path.join(root, "Nested_demo+1+1+0.1+8+8.keras")
model.save(path)
try:
    prune_model.main(["--model_path", path, "--sparsity", "0.5", "--technique", "global"])
    out["exit"] = 0
except SystemExit as stop:
    out["exit"] = stop.code
saved = os.path.join(root, "Nested_demo+1+1+0.1+8+8_pruned_global_0.5.keras")
out["saved"] = os.path.isfile(saved)
if out["saved"]:
    out["reloaded"] = prune_model.calculate_sparsity(tf.keras.models.load_model(saved))
print("RESULT " + json.dumps(out))
"""


@unittest.skipUnless(HAVE_TF, "pruning a model needs TensorFlow")
class NestedModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        child = subprocess.run([sys.executable, "-c", CHILD, str(SRC), cls.tmp.name], env=ENV,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
        lines = [line for line in child.stdout.splitlines() if line.startswith("RESULT ")]
        if child.returncode != 0 or not lines:
            raise AssertionError("the child process failed:\n" + child.stdout[-3000:])
        cls.out = json.loads(lines[-1][len("RESULT "):])

    def test_the_layers_of_a_nested_model_are_found(self):
        # the two convolutions inside the backbone and the head; the depthwise layer is not pruned
        self.assertEqual(self.out["names"], ["inner_conv_a", "inner_conv_b", "head"])
        self.assertEqual(self.out["before"], 0.0)

    def test_global_pruning_reaches_the_backbone(self):
        # 36 + 32 + 16 = 84 kernel weights, half of them under the one threshold
        self.assertAlmostEqual(self.out["global"], 0.5, delta=0.02)
        self.assertGreater(self.out["global_inner"], 0.0)

    def test_layer_pruning_zeroes_the_fraction_of_each_layer(self):
        for name, fraction in self.out["layer"].items():
            self.assertAlmostEqual(fraction, 0.25, delta=0.07, msg=name)

    def test_biases_and_depthwise_kernels_are_left_as_they_are(self):
        self.assertTrue(self.out["biases_kept"])
        self.assertTrue(self.out["depthwise_kept"])

    def test_the_command_line_writes_the_pruned_model(self):
        self.assertEqual(self.out["exit"], 0)
        self.assertTrue(self.out["saved"])
        self.assertAlmostEqual(self.out["reloaded"], 0.5, delta=0.02)


if __name__ == "__main__":
    unittest.main()
