#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Checks the choice of TFLite kernels in the SIL level (sil_benchmark.py --kernels).

The default interpreter hands the graph to the XNNPACK delegate, which refuses
some valid INT8 files; the other kernels accept them. The test builds a small
INT8 model, runs the SIL script on a few images with each kind of kernels and
checks that the choice is written to the metrics file, and that a file the
interpreter cannot prepare ends the run with a message instead of a traceback.

Needs TensorFlow, scikit-learn and seaborn (the packages of the SIL level);
skipped when TensorFlow is not installed. TensorFlow only runs in child
processes, so this file does not load it into the process of the other tests.

    python -m unittest discover -s tests -v
"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"

HAVE_TF = importlib.util.find_spec("tensorflow") is not None
W, H = 16, 12
ENV = dict(os.environ, TF_CPP_MIN_LOG_LEVEL="3", PYTHONIOENCODING="utf-8")

# A small convolutional classifier for 16x12 grayscale images, quantized to INT8.
BUILD_MODEL = """
import sys
import numpy as np
import tensorflow as tf
tf.keras.utils.set_random_seed(0)
model = tf.keras.Sequential([
    tf.keras.layers.Input((12, 16, 1)),
    tf.keras.layers.Conv2D(4, 3, activation="relu"),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(2, activation="softmax"),
])
rng = np.random.default_rng(0)
def representative():
    for _ in range(8):
        yield [rng.integers(0, 256, size=(1, 12, 16, 1)).astype(np.float32)]
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
open(sys.argv[1], "wb").write(converter.convert())
"""

# make_interpreter() with an interpreter that refuses the file, and the table of choices.
REFUSING_INTERPRETER = """
import sys
sys.path.insert(0, sys.argv[1])
from unittest import mock
import sil_benchmark

class Refusing:
    def __init__(self, **kwargs):
        pass
    def allocate_tensors(self):
        raise RuntimeError("failed to create XNNPACK runtime")

print("CHOICES", sorted(sil_benchmark.KERNELS))
print("RESOLVERS", all(hasattr(sil_benchmark.tf.lite.experimental.OpResolverType, name)
                       for name in sil_benchmark.KERNELS.values()))
with mock.patch.object(sil_benchmark.tf.lite, "Interpreter", Refusing):
    sil_benchmark.make_interpreter(sys.argv[2], "default")
print("NOT REACHED")
"""


def python(code, *args, cwd=None):
    return subprocess.run([sys.executable, "-c", code, *args], cwd=cwd, env=ENV,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def make_test_split(folder):
    rng = np.random.default_rng(1)
    for cls in ("non_person", "person"):
        (Path(folder) / cls).mkdir(parents=True)
        for i in range(2):
            Image.fromarray(rng.integers(0, 256, size=(H, W), dtype=np.uint8), "L").save(
                Path(folder) / cls / ("img_%d.png" % i))


@unittest.skipUnless(HAVE_TF, "the SIL level needs TensorFlow")
class SilKernelsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.model = cls.root / "Tiny_test+1+1+0.1+16+12_int8.tflite"
        built = python(BUILD_MODEL, str(cls.model))
        if built.returncode != 0 or not cls.model.is_file():
            cls.tmp.cleanup()
            raise RuntimeError("could not build the test model:\n" + built.stdout[-2000:])
        (cls.root / "Tiny_test+1+1+0.1+16+12_int8.classes.txt").write_text("non_person\nperson\n", encoding="utf-8")
        make_test_split(cls.root / "test")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_sil(self, out, *extra):
        return subprocess.run(
            [sys.executable, str(SRC / "sil_benchmark.py"), "--model_path", str(self.model),
             "--data_dir", str(self.root / "test"), "--width", str(W), "--height", str(H),
             "--output_dir", str(out), *extra],
            cwd=self.root, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def metrics(self, out):
        with open(Path(out) / ("metrics_%s.json" % self.model.stem), encoding="utf-8") as fh:
            return json.load(fh)

    def test_default_run_records_the_default_kernels(self):
        out = self.root / "out_default"
        result = self.run_sil(out)
        self.assertEqual(result.returncode, 0, result.stdout[-2000:])
        self.assertEqual(self.metrics(out)["model"]["kernels"], "default")

    def test_each_kind_of_kernels_runs_and_is_recorded(self):
        for kernels in ("builtin", "reference"):
            with self.subTest(kernels=kernels):
                out = self.root / ("out_" + kernels)
                result = self.run_sil(out, "--kernels", kernels)
                self.assertEqual(result.returncode, 0, result.stdout[-2000:])
                metrics = self.metrics(out)
                self.assertEqual(metrics["model"]["kernels"], kernels)
                self.assertEqual(metrics["dataset"]["n_samples"], 4)

    def test_unknown_kernels_are_refused_by_the_command_line(self):
        result = self.run_sil(self.root / "out_bad", "--kernels", "fastest")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid choice", result.stdout)

    def test_a_file_the_interpreter_cannot_prepare_ends_with_a_message(self):
        result = python(REFUSING_INTERPRETER, str(SRC), str(self.model), cwd=self.root)
        self.assertEqual(result.returncode, 1, result.stdout[-2000:])
        self.assertIn("CHOICES ['builtin', 'default', 'reference']", result.stdout)
        self.assertIn("RESOLVERS True", result.stdout)
        self.assertIn("--kernels builtin", result.stdout)
        self.assertIn("failed to create XNNPACK runtime", result.stdout)
        self.assertNotIn("NOT REACHED", result.stdout)
        self.assertNotIn("Traceback", result.stdout)

    def test_this_file_does_not_load_tensorflow_into_the_test_process(self):
        self.assertNotIn("sil_benchmark", sys.modules)


if __name__ == "__main__":
    unittest.main()
