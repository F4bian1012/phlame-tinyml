#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
count_macs.py - Multiply-accumulate operations (MACs) of a .tflite model, per
operator type and in total, read from the model file.

    python src/count_macs.py models/tflite/MobileNetV2_demo+32+20+0.0001+160+120_int8.tflite
    python src/count_macs.py "models/tflite/*_COCO+*_int8.tflite" --output results/tables/macs_COCO.csv

A MAC is one multiplication whose product is added to a running sum: the unit
of work of a convolution or of a fully connected layer. The count is a
property of the graph (shapes of the filters and of the outputs), so it says
how much arithmetic a model asks for whatever the chip that runs it; divided
into the inference cycles a board measures, it gives cycles per MAC.

What is counted, for one inference (batch of one image):

  CONV_2D            output elements x kernel height x kernel width x input
                     channels of the filter (the filter is stored as
                     [out, kh, kw, in/groups])
  DEPTHWISE_CONV_2D  output elements x kernel height x kernel width (the
                     filter is stored as [1, kh, kw, out])
  FULLY_CONNECTED    output elements x input features of the weights ([units,
                     features])

  Every kernel position is counted for every output element, also where a
  'same' padding puts part of the kernel outside the image: a kernel may skip
  those products, the count does not.

What is not counted (zero MACs; these operators are listed with the number of
elements they produce, so the size of what is left out can be seen):

  the bias addition of the three operators above; element-wise arithmetic
  (ADD, SUB, MUL, DIV: one operation per element, nothing accumulated);
  pooling and MEAN (sums and comparisons, no products); activations (RELU,
  RELU6, HARD_SWISH, LOGISTIC, TANH, SOFTMAX); QUANTIZE and DEQUANTIZE; and
  the operators that only move data (RESHAPE, PAD, CONCATENATION, ...).

An operator the script knows neither as counted nor as free of MACs is never
counted as zero in silence: it is named in the table (unclassified_operators)
and on the screen, and the script ends with exit code 3.

The graph is read from the flatbuffer itself, with the schema TensorFlow ships
(tensorflow.lite.python.schema_py_generated), not through the interpreter:
the file is described as it is, whichever kernels or delegate would run it
(the default interpreter refuses to prepare some valid INT8 files), and no
private method of the interpreter is needed. Nothing is executed.

Output: a CSV with one row per model (--output; default results/tables/macs.csv),
with the SHA-256 of each file so the count stays tied to the exact model.
"""

import csv
import glob
import hashlib
import os
import sys

import artifacts
import cli

# Operators with products that are accumulated: the MACs.
COUNTED = ("CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED")

# Operators that do no multiply-accumulate: element-wise arithmetic, pooling and
# reductions, activations, (de)quantization and data movement.
NO_MACS = frozenset((
    "ADD", "SUB", "MUL", "DIV", "MINIMUM", "MAXIMUM", "NEG", "ABS", "SQRT", "RSQRT", "SQUARED_DIFFERENCE",
    "AVERAGE_POOL_2D", "MAX_POOL_2D", "MEAN", "SUM", "REDUCE_MAX", "REDUCE_MIN",
    "RELU", "RELU6", "RELU_N1_TO_1", "RELU_0_TO_1", "LEAKY_RELU", "PRELU", "HARD_SWISH", "LOGISTIC", "TANH",
    "SOFTMAX", "LOG_SOFTMAX",
    "QUANTIZE", "DEQUANTIZE", "CAST",
    "RESHAPE", "SQUEEZE", "EXPAND_DIMS", "PAD", "PADV2", "MIRROR_PAD", "CONCATENATION", "PACK", "UNPACK",
    "SLICE", "STRIDED_SLICE", "SPLIT", "SPLIT_V", "TRANSPOSE", "SHAPE", "RESIZE_NEAREST_NEIGHBOR",
    "RESIZE_BILINEAR", "ARG_MAX", "ARG_MIN",
))

CSV_COLUMNS = ("model_file", "sha256", "bytes", "input_shape", "n_operators",
               "conv2d_ops", "conv2d_macs", "depthwise_conv2d_ops", "depthwise_conv2d_macs",
               "fully_connected_ops", "fully_connected_macs", "macs_total",
               "other_operators", "other_output_elements", "unclassified_operators")
COLUMN_OF = {"CONV_2D": "conv2d", "DEPTHWISE_CONV_2D": "depthwise_conv2d", "FULLY_CONNECTED": "fully_connected"}


def sha256_file(path, chunk=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _product(shape):
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def operator_macs(name, input_shapes, output_shape):
    """MACs of one CONV_2D, DEPTHWISE_CONV_2D or FULLY_CONNECTED, or None when its shapes
    do not allow the count (a missing or dynamic dimension).

    input_shapes are the shapes of the operator's inputs in the order of the
    file: the data, then the filter or the weights, then the bias.
    """
    if len(input_shapes) < 2 or not output_shape or not input_shapes[1]:
        return None
    weights = input_shapes[1]
    if any(int(d) <= 0 for d in list(output_shape) + list(weights)):
        return None
    outputs = _product(output_shape)
    if name == "CONV_2D":                      # filter [out, kh, kw, in/groups]
        return outputs * _product(weights[1:]) if len(weights) == 4 else None
    if name == "DEPTHWISE_CONV_2D":            # filter [1, kh, kw, out]
        return outputs * int(weights[1]) * int(weights[2]) if len(weights) == 4 else None
    if name == "FULLY_CONNECTED":              # weights [units, features]
        return outputs * int(weights[-1]) if len(weights) == 2 else None
    return None


def read_graph(path):
    """The operators of the first subgraph of a .tflite file, as dictionaries with the
    operator name and the shapes of its inputs and outputs; plus the input shape of the
    model and the number of subgraphs."""
    # Imported here: --help and a wrong call do not need TensorFlow.
    from tensorflow.lite.python import schema_py_generated as schema

    with open(path, "rb") as fh:
        buffer = fh.read()
    root = getattr(schema.Model, "GetRootAsModel", None) or schema.Model.GetRootAs
    model = root(buffer, 0)
    names = {value: key for key, value in vars(schema.BuiltinOperator).items()
             if not key.startswith("_") and isinstance(value, int)}
    subgraph = model.Subgraphs(0)

    def shape_of(index):
        if index < 0:                           # an optional input that is not given
            return None
        tensor = subgraph.Tensors(index)
        return [int(d) for d in tensor.ShapeAsNumpy()] if tensor.ShapeLength() else []

    operators = []
    for j in range(subgraph.OperatorsLength()):
        op = subgraph.Operators(j)
        code = model.OperatorCodes(op.OpcodeIndex())
        # Files written before the code was widened keep it in the old field.
        builtin = max(code.BuiltinCode(), code.DeprecatedBuiltinCode())
        name = names.get(builtin, "BUILTIN_%d" % builtin)
        if name == "CUSTOM" and code.CustomCode():
            name = "CUSTOM:" + code.CustomCode().decode("utf-8", "replace")
        operators.append({
            "name": name,
            "inputs": [shape_of(int(i)) for i in op.InputsAsNumpy()] if op.InputsLength() else [],
            "outputs": [shape_of(int(i)) for i in op.OutputsAsNumpy()] if op.OutputsLength() else [],
        })
    input_shape = shape_of(int(subgraph.Inputs(0))) if subgraph.InputsLength() else []
    return operators, input_shape, model.SubgraphsLength()


def count_macs(operators, n_subgraphs=1):
    """MACs of a list of operators (as read_graph gives them), per counted type and in total."""
    row = {"n_operators": len(operators), "macs_total": 0, "other_output_elements": 0}
    for name in COUNTED:
        row[COLUMN_OF[name] + "_ops"] = 0
        row[COLUMN_OF[name] + "_macs"] = 0
    other, unclassified = {}, {}
    for op in operators:
        name = op["name"]
        if name in COUNTED:
            macs = operator_macs(name, op["inputs"], op["outputs"][0] if op["outputs"] else None)
            if macs is None:
                unclassified[name + " (shape not static)"] = unclassified.get(name + " (shape not static)", 0) + 1
                continue
            row[COLUMN_OF[name] + "_ops"] += 1
            row[COLUMN_OF[name] + "_macs"] += macs
            row["macs_total"] += macs
        elif name in NO_MACS:
            other[name] = other.get(name, 0) + 1
            row["other_output_elements"] += sum(_product(shape) for shape in op["outputs"]
                                                if shape and all(d > 0 for d in shape))
        else:
            unclassified[name] = unclassified.get(name, 0) + 1
    if n_subgraphs > 1:
        unclassified["%d further subgraph(s) not read" % (n_subgraphs - 1)] = 1
    row["other_operators"] = ";".join("%s:%d" % item for item in sorted(other.items()))
    row["unclassified_operators"] = ";".join("%s:%d" % item for item in sorted(unclassified.items()))
    return row


def describe(path):
    """The CSV row of one model file."""
    operators, input_shape, n_subgraphs = read_graph(path)
    row = count_macs(operators, n_subgraphs)
    row["model_file"] = os.path.basename(path)     # the file name only: no local path in the table
    row["sha256"] = sha256_file(path)
    row["bytes"] = os.path.getsize(path)
    row["input_shape"] = "x".join(str(d) for d in input_shape[1:]) if len(input_shape) > 1 else ""
    return row


def parse_args():
    parser = cli.ArgumentParser(description="Count the multiply-accumulate operations (MACs) of .tflite models, "
                                            "per operator type and in total, from the model files.")
    parser.add_argument("models", nargs="+",
                        help="One or more .tflite files; a pattern such as \"models/tflite/*_int8.tflite\" is expanded")
    parser.add_argument("--output", default=os.path.join("results", "tables", "macs.csv"),
                        help="CSV to write, one row per model (default: results/tables/macs.csv)")
    parser.add_argument("--overwrite", action="store_true", help="Replace the CSV if it already exists")
    return parser.parse_args()


def main():
    args = parse_args()
    paths = []
    for pattern in args.models:
        found = sorted(glob.glob(pattern)) if any(c in pattern for c in "*?[") else [pattern]
        if not found:
            print("ERROR: no file matches %s" % pattern)
            sys.exit(1)
        paths.extend(found)
    for path in paths:
        if not os.path.isfile(path):
            print("ERROR: model not found: %s" % path)
            sys.exit(1)
    names = [os.path.basename(path) for path in paths]
    if len(set(names)) != len(names):
        print("ERROR: two of the files have the same name; the table identifies a model by its file name")
        sys.exit(1)
    artifacts.refuse_to_overwrite([args.output], args.overwrite, hint="write somewhere else with --output")

    rows = [describe(path) for path in paths]
    rows.sort(key=lambda row: (row["macs_total"], row["model_file"]))

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in CSV_COLUMNS})

    print("%-58s %14s %14s %14s %14s" % ("model", "CONV_2D", "DEPTHWISE", "FULLY_CONN.", "MACs total"))
    for row in rows:
        print("%-58s %14d %14d %14d %14d" % (row["model_file"][:58], row["conv2d_macs"],
                                             row["depthwise_conv2d_macs"], row["fully_connected_macs"],
                                             row["macs_total"]))
    print("Wrote %s (%d model%s)" % (args.output, len(rows), "" if len(rows) == 1 else "s"))
    problems = [row for row in rows if row["unclassified_operators"]]
    for row in problems:
        print("[WARN] %s has operators that were not counted and are not known to be free of MACs: %s"
              % (row["model_file"], row["unclassified_operators"]))
    if problems:
        print("[WARN] The totals of those models are lower bounds.")
        sys.exit(3)


if __name__ == "__main__":
    main()
