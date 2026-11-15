#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Checks src/count_macs.py: the multiply-accumulate operations (MACs) it counts
for a .tflite model.

  - the formulas, on shapes written by hand: a convolution, a grouped one, a
    depthwise convolution, a fully connected layer;
  - what is left out is named: operators without MACs are listed with the
    elements they produce, and an operator the script does not know, or one
    whose shape is not static, is reported instead of being counted as zero;
  - a small model built for the test: the count read from the INT8 file equals
    the count worked out by hand from the layers (5,008 MACs) and the count
    made layer by layer on the Keras model, which never sees the .tflite;
  - the table: one row per model, the file name and its SHA-256, no local path;
    exit code 3 when a model has an operator that could not be classified;
  - the published table of the COCO models is internally consistent and names
    model files the checksum list knows.

Expected values are computed in the test from the layer definitions, not taken
from the script. TensorFlow only runs in one child process; that part is
skipped when TensorFlow is not installed.

    python -m unittest discover -s tests -v
"""
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
sys.path.insert(0, str(SRC))

import count_macs  # noqa: E402

HAVE_TF = importlib.util.find_spec("tensorflow") is not None
ENV = dict(os.environ, TF_CPP_MIN_LOG_LEVEL="3", PYTHONIOENCODING="utf-8")
SCRIPT = str(SRC / "count_macs.py")

# The layers of the test model, 12 rows x 16 columns x 1 channel in:
#   Conv2D(4, 3x3, stride 2, same)   -> 6 x 8 x 4     6*8*4 * 3*3*1 = 1728
#   DepthwiseConv2D(3x3, same)       -> 6 x 8 x 4     6*8*4 * 3*3   = 1728
#   Conv2D(8, 1x1)                   -> 6 x 8 x 8     6*8*8 * 1*1*4 = 1536
#   GlobalAveragePooling2D, Dense(2) -> 2             2 * 8         =   16
HAND = {"conv2d_macs": 6 * 8 * 4 * 3 * 3 * 1 + 6 * 8 * 8 * 1 * 1 * 4, "depthwise_conv2d_macs": 6 * 8 * 4 * 3 * 3,
        "fully_connected_macs": 2 * 8}

# Runs in a child process: builds the two models, counts the MACs of the first one layer by
# layer from the Keras model, and runs the script twice in that same process.
CHILD = """
import contextlib, io, json, sys
import numpy as np
import tensorflow as tf
sys.path.insert(0, sys.argv[1])
root = sys.argv[2]
tf.keras.utils.set_random_seed(0)

def to_int8(model, path, shape):
    rng = np.random.default_rng(0)
    def representative():
        for _ in range(8):
            yield [rng.integers(0, 256, size=(1,) + shape).astype(np.float32)]
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    with contextlib.redirect_stdout(io.StringIO()):
        data = converter.convert()
    open(path, "wb").write(data)

L = tf.keras.layers
good = tf.keras.Sequential([
    L.Input((12, 16, 1)),
    L.Conv2D(4, 3, strides=2, padding="same", activation="relu"),
    L.DepthwiseConv2D(3, padding="same", activation="relu"),
    L.Conv2D(8, 1, activation="relu"),
    L.GlobalAveragePooling2D(),
    L.Dense(2, activation="softmax"),
])
to_int8(good, root + "/Good_int8.tflite", (12, 16, 1))
# a transposed convolution: products are accumulated, but the script does not count them
other = tf.keras.Sequential([
    L.Input((4, 4, 1)),
    L.Conv2DTranspose(2, 2, strides=2),
    L.GlobalAveragePooling2D(),
    L.Dense(2, activation="softmax"),
])
to_int8(other, root + "/Other_int8.tflite", (4, 4, 1))

keras_count = {"conv2d_macs": 0, "depthwise_conv2d_macs": 0, "fully_connected_macs": 0}
for layer in good.layers:
    out = layer.output.shape
    if isinstance(layer, L.DepthwiseConv2D):
        keras_count["depthwise_conv2d_macs"] += int(out[1] * out[2] * out[3]) * layer.kernel_size[0] * layer.kernel_size[1]
    elif isinstance(layer, L.Conv2D):
        keras_count["conv2d_macs"] += (int(out[1] * out[2] * out[3]) * layer.kernel_size[0] * layer.kernel_size[1]
                                       * int(layer.input.shape[-1]))
    elif isinstance(layer, L.Dense):
        keras_count["fully_connected_macs"] += layer.units * int(layer.input.shape[-1])

import count_macs
runs = {}
for name, argv in (("one", [root + "/Good_int8.tflite", "--output", root + "/one.csv"]),
                   ("pattern", [root + "/*_int8.tflite", "--output", root + "/both.csv"])):
    code, log = 0, io.StringIO()
    sys.argv = ["count_macs.py"] + argv
    with contextlib.redirect_stdout(log):
        try:
            count_macs.main()
        except SystemExit as stop:
            code = stop.code
    runs[name] = {"code": code, "log": log.getvalue()}
print("RESULT " + json.dumps({"keras": keras_count, "runs": runs}))
"""


class FormulaTest(unittest.TestCase):
    def test_convolution(self):
        # 3x3 kernel over 3 input channels, 16 filters, output 60 x 80
        macs = count_macs.operator_macs("CONV_2D", [[1, 120, 160, 3], [16, 3, 3, 3], [16]], [1, 60, 80, 16])
        self.assertEqual(macs, 60 * 80 * 16 * 3 * 3 * 3)

    def test_grouped_convolution_counts_the_channels_of_its_filter(self):
        # 8 input channels in 2 groups: each filter sees 4 of them
        macs = count_macs.operator_macs("CONV_2D", [[1, 10, 10, 8], [6, 3, 3, 4], [6]], [1, 10, 10, 6])
        self.assertEqual(macs, 10 * 10 * 6 * 3 * 3 * 4)

    def test_depthwise_convolution(self):
        macs = count_macs.operator_macs("DEPTHWISE_CONV_2D", [[1, 60, 80, 16], [1, 3, 3, 16], [16]], [1, 30, 40, 16])
        self.assertEqual(macs, 30 * 40 * 16 * 3 * 3)

    def test_fully_connected(self):
        self.assertEqual(count_macs.operator_macs("FULLY_CONNECTED", [[1, 1280], [2, 1280], [2]], [1, 2]), 2 * 1280)
        # applied to every position of a sequence of 5
        self.assertEqual(count_macs.operator_macs("FULLY_CONNECTED", [[1, 5, 8], [3, 8], [3]], [1, 5, 3]), 5 * 3 * 8)

    def test_shape_that_is_not_static_gives_no_count(self):
        self.assertIsNone(count_macs.operator_macs("CONV_2D", [[-1, 12, 16, 1], [4, 3, 3, 1]], [-1, 6, 8, 4]))
        self.assertIsNone(count_macs.operator_macs("CONV_2D", [[1, 12, 16, 1], []], [1, 6, 8, 4]))
        self.assertIsNone(count_macs.operator_macs("FULLY_CONNECTED", [[1, 8]], [1, 2]))


class CountTest(unittest.TestCase):
    OPS = [
        {"name": "CONV_2D", "inputs": [[1, 12, 16, 1], [4, 3, 3, 1], [4]], "outputs": [[1, 6, 8, 4]]},
        {"name": "DEPTHWISE_CONV_2D", "inputs": [[1, 6, 8, 4], [1, 3, 3, 4], [4]], "outputs": [[1, 6, 8, 4]]},
        {"name": "ADD", "inputs": [[1, 6, 8, 4], [1, 6, 8, 4]], "outputs": [[1, 6, 8, 4]]},
        {"name": "MEAN", "inputs": [[1, 6, 8, 4], [2]], "outputs": [[1, 4]]},
        {"name": "FULLY_CONNECTED", "inputs": [[1, 4], [2, 4], [2]], "outputs": [[1, 2]]},
        {"name": "SOFTMAX", "inputs": [[1, 2]], "outputs": [[1, 2]]},
    ]

    def test_totals_per_type_and_what_is_left_out(self):
        row = count_macs.count_macs(self.OPS)
        self.assertEqual((row["conv2d_ops"], row["conv2d_macs"]), (1, 6 * 8 * 4 * 9))
        self.assertEqual((row["depthwise_conv2d_ops"], row["depthwise_conv2d_macs"]), (1, 6 * 8 * 4 * 9))
        self.assertEqual((row["fully_connected_ops"], row["fully_connected_macs"]), (1, 8))
        self.assertEqual(row["macs_total"], 2 * 6 * 8 * 4 * 9 + 8)
        self.assertEqual(row["n_operators"], 6)
        self.assertEqual(row["other_operators"], "ADD:1;MEAN:1;SOFTMAX:1")
        self.assertEqual(row["other_output_elements"], 6 * 8 * 4 + 4 + 2)
        self.assertEqual(row["unclassified_operators"], "")

    def test_unknown_operator_is_named_not_counted_as_zero(self):
        ops = self.OPS + [{"name": "TRANSPOSE_CONV", "inputs": [[4], [2, 2, 2, 1], [1, 4, 4, 1]], "outputs": [[1, 8, 8, 2]]},
                          {"name": "CUSTOM:MyOp", "inputs": [[1, 2]], "outputs": [[1, 2]]}]
        row = count_macs.count_macs(ops)
        self.assertEqual(row["unclassified_operators"], "CUSTOM:MyOp:1;TRANSPOSE_CONV:1")
        self.assertEqual(row["macs_total"], 2 * 6 * 8 * 4 * 9 + 8)     # the known part is still counted

    def test_counted_operator_without_a_static_shape_is_reported(self):
        ops = [{"name": "CONV_2D", "inputs": [[-1, 12, 16, 1], [4, 3, 3, 1], [4]], "outputs": [[-1, 6, 8, 4]]}]
        row = count_macs.count_macs(ops)
        self.assertEqual((row["conv2d_ops"], row["macs_total"]), (0, 0))
        self.assertEqual(row["unclassified_operators"], "CONV_2D (shape not static):1")

    def test_further_subgraphs_are_reported(self):
        self.assertIn("1 further subgraph(s) not read", count_macs.count_macs(self.OPS, n_subgraphs=2)["unclassified_operators"])

    def test_no_operator_is_both_counted_and_free(self):
        self.assertFalse(set(count_macs.COUNTED) & count_macs.NO_MACS)


class CommandLineTest(unittest.TestCase):
    """Calls that end before a model is read: they do not need TensorFlow."""

    def run_script(self, *args, cwd=None):
        return subprocess.run([sys.executable, SCRIPT, *args], cwd=cwd, env=ENV, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8")

    def test_missing_file_and_empty_pattern(self):
        with tempfile.TemporaryDirectory() as tmp:
            for argument in (os.path.join(tmp, "nothing.tflite"), os.path.join(tmp, "*.tflite")):
                result = self.run_script(argument, "--output", os.path.join(tmp, "macs.csv"))
                self.assertEqual(result.returncode, 1, result.stdout)
                self.assertNotIn("Traceback", result.stdout)
                self.assertFalse(os.path.exists(os.path.join(tmp, "macs.csv")))

    def test_existing_table_is_not_replaced_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            model, table = Path(tmp, "a_int8.tflite"), Path(tmp, "macs.csv")
            model.write_bytes(b"not read")
            table.write_text("earlier table\n", encoding="utf-8")
            result = self.run_script(str(model), "--output", str(table))
            self.assertEqual(result.returncode, 1, result.stdout)
            self.assertIn("--overwrite", result.stdout)
            self.assertEqual(table.read_text(encoding="utf-8"), "earlier table\n")

    def test_two_files_with_the_same_name_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            for folder in ("a", "b"):
                Path(tmp, folder).mkdir()
                Path(tmp, folder, "m_int8.tflite").write_bytes(b"x")
            result = self.run_script(os.path.join(tmp, "a", "m_int8.tflite"), os.path.join(tmp, "b", "m_int8.tflite"),
                                     "--output", os.path.join(tmp, "macs.csv"))
            self.assertEqual(result.returncode, 1, result.stdout)
            self.assertIn("same name", result.stdout)


@unittest.skipUnless(HAVE_TF, "reading a .tflite needs TensorFlow")
class ModelFileTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.root = Path(cls.tmp.name)
        child = subprocess.run([sys.executable, "-c", CHILD, str(SRC), cls.root.as_posix()], cwd=cls.root, env=ENV,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
        lines = [line for line in child.stdout.splitlines() if line.startswith("RESULT ")]
        if child.returncode != 0 or not lines:
            raise RuntimeError("the child process failed:\n" + child.stdout[-3000:])
        cls.result = json.loads(lines[-1][len("RESULT "):])

    def table(self, name):
        with open(self.root / name, encoding="utf-8", newline="") as fh:
            return {row["model_file"]: row for row in csv.DictReader(fh)}

    def test_count_of_the_file_equals_the_hand_count_and_the_keras_count(self):
        self.assertEqual(self.result["runs"]["one"]["code"], 0, self.result["runs"]["one"]["log"])
        row = self.table("one.csv")["Good_int8.tflite"]
        from_file = {key: int(row[key]) for key in HAND}
        self.assertEqual(from_file, HAND)
        self.assertEqual(from_file, self.result["keras"])
        self.assertEqual(int(row["macs_total"]), 5008)
        self.assertEqual((row["conv2d_ops"], row["depthwise_conv2d_ops"], row["fully_connected_ops"]), ("2", "1", "1"))
        self.assertEqual(row["unclassified_operators"], "")
        self.assertIn("MEAN:1", row["other_operators"])
        self.assertIn("SOFTMAX:1", row["other_operators"])
        self.assertEqual(row["input_shape"], "12x16x1")

    def test_row_is_tied_to_the_file_and_holds_no_path(self):
        row = self.table("one.csv")["Good_int8.tflite"]
        data = (self.root / "Good_int8.tflite").read_bytes()
        self.assertEqual((row["sha256"], int(row["bytes"])), (hashlib.sha256(data).hexdigest(), len(data)))
        text = (self.root / "one.csv").read_text(encoding="utf-8")
        self.assertNotIn(self.root.name, text)
        self.assertEqual(text.splitlines()[0], ",".join(count_macs.CSV_COLUMNS))

    def test_operator_that_cannot_be_classified_is_reported_with_exit_code_3(self):
        run = self.result["runs"]["pattern"]
        self.assertEqual(run["code"], 3, run["log"])
        rows = self.table("both.csv")
        self.assertEqual(sorted(rows), ["Good_int8.tflite", "Other_int8.tflite"])       # the pattern found both
        self.assertEqual(rows["Other_int8.tflite"]["unclassified_operators"], "TRANSPOSE_CONV:1")
        self.assertIn("TRANSPOSE_CONV", run["log"])
        self.assertIn("lower bounds", run["log"])
        self.assertEqual(int(rows["Good_int8.tflite"]["macs_total"]), 5008)             # the other row is unaffected


class PublishedTableTest(unittest.TestCase):
    TABLE = REPO / "results" / "tables" / "macs_COCO.csv"

    @classmethod
    def setUpClass(cls):
        with open(cls.TABLE, encoding="utf-8", newline="") as fh:
            cls.rows = list(csv.DictReader(fh))

    def test_total_is_the_sum_of_its_parts_and_nothing_is_unclassified(self):
        self.assertEqual(len(self.rows), 7)
        for row in self.rows:
            parts = sum(int(row[key]) for key in ("conv2d_macs", "depthwise_conv2d_macs", "fully_connected_macs"))
            self.assertEqual(int(row["macs_total"]), parts, row["model_file"])
            self.assertEqual(row["unclassified_operators"], "", row["model_file"])
            self.assertEqual(row["input_shape"], "120x160x1")

    def test_every_row_names_a_model_file_of_the_checksum_list(self):
        listed = {}
        with open(REPO / "models" / "checksums.sha256", encoding="utf-8") as fh:
            for line in fh:
                if line.strip() and not line.startswith("#"):
                    digest, rel = line.rstrip("\r\n").split("  ", 1)
                    listed[rel.rsplit("/", 1)[-1]] = digest
        for row in self.rows:
            self.assertEqual(listed.get(row["model_file"]), row["sha256"], row["model_file"])

    def test_table_holds_file_names_only(self):
        text = self.TABLE.read_text(encoding="utf-8")
        for fragment in ("\\", "/", "Users"):
            self.assertNotIn(fragment, text)


if __name__ == "__main__":
    unittest.main()
