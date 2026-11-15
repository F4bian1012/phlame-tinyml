#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Checks src/evaluation.py, the part mil_benchmark.py and sil_benchmark.py share:
the label of every image from the model output, the metrics against values
worked out by hand, the three result files and the order of their keys, the
class names that travel with a model, and the refusals (existing results,
missing test split).

The module runs in a child process, so that matplotlib, seaborn and
scikit-learn are not loaded into the process of the other tests, and with a
stand-in `tensorflow` that fails when imported, which shows the module does
not need it. Skipped when scikit-learn, seaborn or matplotlib is missing.

    python -m unittest discover -s tests -v
"""
import ast
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
READY = all(importlib.util.find_spec(name) is not None for name in ("sklearn", "seaborn", "matplotlib", "numpy"))

# Six invented images of three classes. Worked out by hand:
#   class a: 1 right of 2, one image of c also called a  -> precision 1/2, recall 1/2, F1 1/2
#   class b: 2 right of 2, one image of a also called b  -> precision 2/3, recall 1,   F1 4/5
#   class c: 1 right of 2, nothing else called c         -> precision 1,   recall 1/2, F1 2/3
Y_TRUE = [0, 0, 1, 1, 2, 2]
Y_PRED = [0, 1, 1, 1, 2, 0]
REL = ["a/1.png", "a/2.png", "b/1.png", "b/2.png", "c/1.png", "c/2.png"]
NAMES = ["a", "b", "c"]
MODEL_BYTES = b"invented model file"

DRIVER = r"""
import contextlib, io, json, os, sys
sys.path.insert(0, sys.argv[1])
import numpy as np
import evaluation

root = sys.argv[2]
out = {"tensorflow_loaded": "tensorflow" in sys.modules}


def exit_code(function, *args):
    log = io.StringIO()
    try:
        with contextlib.redirect_stdout(log):
            function(*args)
    except SystemExit as exc:
        return [exc.code, log.getvalue()]
    return [None, log.getvalue()]


out["class_name"] = [evaluation.class_name(i, ["a", "b"]) for i in (0, 1, 2, -1, None)]
out["json_safe"] = [evaluation.json_safe(np.int64(3)), evaluation.json_safe(np.float64(0.5)),
                    evaluation.json_safe(np.array([[1, 2]]))]
try:
    evaluation.json_safe(object())
    out["json_safe_other"] = "accepted"
except TypeError:
    out["json_safe_other"] = "TypeError"
out["sha256_of_a_folder"] = evaluation.sha256_file(root)
out["model_name"] = evaluation.model_name(os.path.join("some", "folder", "Net_T+1+2+0.1+8+6_int8.tflite"))

softmax = np.array([[0.1, 0.9], [0.8, 0.2], [0.5, 0.5]])
sigmoid = np.array([[0.9], [0.2], [0.5]])
out["labels_softmax"] = evaluation.predicted_labels(softmax, 2).tolist()
out["labels_sigmoid_two_classes"] = evaluation.predicted_labels(sigmoid, 2).tolist()
out["labels_single_class"] = evaluation.predicted_labels(sigmoid, 1).tolist()

model = os.path.join(root, "Net_T+1+2+0.1+8+6.keras")
with contextlib.redirect_stdout(io.StringIO()):
    # no file beside the model, and the working directory has no models/class_names.txt
    out["names_without_a_file"] = evaluation.read_class_names(model)[0]
with open(os.path.join(root, "Net_T+1+2+0.1+8+6.classes.txt"), "w") as fh:
    fh.write("a\nb\n\nc\n")
log = io.StringIO()
with contextlib.redirect_stdout(log):
    names, path = evaluation.read_class_names(model)
    out["names_beside_the_model"] = [names, os.path.basename(path)]
    out["names_matching_the_split"] = evaluation.class_names_of_this_split(names, path, os.path.join(root, "split_abc"))
    out["names_of_another_split"] = evaluation.class_names_of_this_split(names, path, os.path.join(root, "split_xy"))
out["names_log"] = log.getvalue()

out["no_split"] = exit_code(evaluation.stop_without_test_split, "data/splits/test", 160, 120, "data/splits")

y_true, y_pred = np.array(json.loads(sys.argv[3])), np.array(json.loads(sys.argv[4]))
rel, names = json.loads(sys.argv[5]), json.loads(sys.argv[6])
log = io.StringIO()
with contextlib.redirect_stdout(log):
    evaluation.write_level_results(
        level="MIL", model_path=model, output_dir=os.path.join(root, "mil"), y_true=y_true, y_pred=y_pred,
        rel_paths=rel, class_names=names, dataset={"data_dir": "d"}, image_size=(8, 6),
        software={"python": "x"}, report_title="REPORT", plot_title="Title")
    evaluation.write_level_results(
        level="SIL", model_path=model, output_dir=os.path.join(root, "sil"), y_true=y_true, y_pred=y_pred,
        rel_paths=rel, class_names=names, dataset={"splits_dir": "s", "data_dir": "d"}, image_size=(8, 6),
        software={"python": "x"}, report_title="REPORT", plot_title="Title", model_extra={"kernels": "reference"})
out["report_log"] = log.getvalue()
out["refused"] = exit_code(evaluation.refuse_to_replace_results, os.path.join(root, "mil"), model, False)
out["allowed_with_overwrite"] = exit_code(evaluation.refuse_to_replace_results, os.path.join(root, "mil"), model, True)
out["allowed_elsewhere"] = exit_code(evaluation.refuse_to_replace_results, os.path.join(root, "other"), model, False)
out["tensorflow_loaded_at_the_end"] = "tensorflow" in sys.modules
print("RESULT " + json.dumps(out))
"""


@unittest.skipUnless(READY, "evaluation.py needs numpy, matplotlib, scikit-learn and seaborn")
class EvaluationModuleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.root = Path(cls.tmp.name)
        (cls.root / "Net_T+1+2+0.1+8+6.keras").write_bytes(MODEL_BYTES)
        for folder in ("split_abc/a", "split_abc/b", "split_abc/c", "split_xy/x", "split_xy/y", "cwd"):
            (cls.root / folder).mkdir(parents=True)
        # A package named tensorflow that fails on import: the module must not ask for it.
        (cls.root / "standin" / "tensorflow").mkdir(parents=True)
        (cls.root / "standin" / "tensorflow" / "__init__.py").write_text(
            'raise ImportError("TENSORFLOW-WAS-IMPORTED")\n', encoding="utf-8")
        env = dict(os.environ, PYTHONPATH=str(cls.root / "standin"), MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
        ran = subprocess.run(
            [sys.executable, "-c", DRIVER, str(SRC), str(cls.root), json.dumps(Y_TRUE), json.dumps(Y_PRED),
             json.dumps(REL), json.dumps(NAMES)],
            cwd=cls.root / "cwd", env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8")
        lines = [line for line in ran.stdout.splitlines() if line.startswith("RESULT ")]
        if ran.returncode != 0 or not lines:
            raise RuntimeError("the evaluation module failed:\n" + ran.stdout[-3000:])
        cls.out = json.loads(lines[-1][len("RESULT "):])

    def metrics(self, level):
        with open(self.root / level / "metrics_Net_T+1+2+0.1+8+6.json", encoding="utf-8") as fh:
            return json.load(fh)

    def test_tensorflow_is_not_needed(self):
        self.assertFalse(self.out["tensorflow_loaded"])
        self.assertFalse(self.out["tensorflow_loaded_at_the_end"])

    def test_small_helpers(self):
        self.assertEqual(self.out["class_name"], ["a", "b", "Invalid_2", "", ""])
        self.assertEqual(self.out["json_safe"], [3, 0.5, [[1, 2]]])
        self.assertEqual(self.out["json_safe_other"], "TypeError")
        self.assertIsNone(self.out["sha256_of_a_folder"])
        self.assertEqual(self.out["model_name"], "Net_T+1+2+0.1+8+6_int8")

    def test_label_of_every_image(self):
        self.assertEqual(self.out["labels_softmax"], [1, 0, 0])               # a tie goes to the first class
        self.assertEqual(self.out["labels_sigmoid_two_classes"], [1, 0, 0])   # 0.5 is not above the threshold
        self.assertEqual(self.out["labels_single_class"], [1, 0, 0])

    def test_class_names_travel_with_the_model(self):
        self.assertEqual(self.out["names_without_a_file"], [])
        self.assertEqual(self.out["names_beside_the_model"], [["a", "b", "c"], "Net_T+1+2+0.1+8+6.classes.txt"])
        self.assertEqual(self.out["names_matching_the_split"], ["a", "b", "c"])
        self.assertEqual(self.out["names_of_another_split"], [])
        self.assertIn("[WARN]", self.out["names_log"])
        self.assertIn("will be inferred from the directory", self.out["names_log"])

    def test_missing_test_split_stops_with_the_command_that_creates_it(self):
        code, log = self.out["no_split"]
        self.assertEqual(code, 1)
        self.assertIn("test split not found at data/splits/test", log)
        self.assertIn("python src/split_dataset.py --input_dir data/processed/160x120 --output_dir data/splits", log)

    def test_metrics_equal_the_values_worked_out_by_hand(self):
        m = self.metrics("mil")
        g = m["global_metrics"]
        self.assertAlmostEqual(g["accuracy"], 4 / 6, places=12)
        self.assertAlmostEqual(g["recall_weighted"], 4 / 6, places=12)
        self.assertAlmostEqual(g["precision_weighted"], (1 / 2 + 2 / 3 + 1) / 3, places=12)
        self.assertAlmostEqual(g["f1_macro"], (1 / 2 + 4 / 5 + 2 / 3) / 3, places=12)
        self.assertAlmostEqual(g["f1_weighted"], g["f1_macro"], places=12)      # equal support
        self.assertEqual(m["confusion_matrix"]["matrix"], [[1, 1, 0], [0, 2, 0], [1, 0, 1]])
        self.assertEqual(m["confusion_matrix"]["labels"], NAMES)
        self.assertAlmostEqual(m["per_class_metrics"]["b"]["f1-score"], 4 / 5, places=12)
        self.assertEqual(m["per_class_metrics"]["c"]["support"], 2)
        self.assertEqual(self.metrics("sil")["global_metrics"], g)
        self.assertIn("Accuracy:             0.6667", self.out["report_log"])

    def test_keys_and_their_order(self):
        top = ["level", "generated_utc", "model", "dataset", "global_metrics", "per_class_metrics",
               "confusion_matrix", "software", "artifacts"]
        mil, sil = self.metrics("mil"), self.metrics("sil")
        self.assertEqual((list(mil), list(sil)), (top, top))
        self.assertEqual((mil["level"], sil["level"]), ("MIL", "SIL"))
        self.assertEqual(list(mil["model"]), ["path", "name", "sha256", "bytes"])
        self.assertEqual(list(sil["model"]), ["path", "name", "sha256", "bytes", "kernels"])
        self.assertEqual(list(mil["dataset"]), ["data_dir", "n_samples", "class_names", "image_size"])
        self.assertEqual(list(sil["dataset"]), ["splits_dir", "data_dir", "n_samples", "class_names", "image_size"])
        self.assertEqual(mil["dataset"]["n_samples"], 6)
        self.assertEqual(mil["dataset"]["image_size"], {"width": 8, "height": 6})
        self.assertEqual(mil["software"], {"python": "x"})
        self.assertEqual(mil["model"]["sha256"], hashlib.sha256(MODEL_BYTES).hexdigest())
        self.assertEqual(mil["model"]["bytes"], len(MODEL_BYTES))
        self.assertEqual(list(mil["artifacts"]), ["confusion_matrix_png", "predictions_csv"])

    def test_predictions_file(self):
        for level in ("mil", "sil"):
            with open(self.root / level / "predictions_Net_T+1+2+0.1+8+6.csv", encoding="utf-8", newline="") as fh:
                rows = list(csv.reader(fh, delimiter=";"))
            self.assertEqual(rows[0], ["Image", "RelPath", "TrueLabel", "TrueClass", "PredLabel", "PredClass", "Correct"])
            self.assertEqual(rows[1], ["1.png", "a/1.png", "0", "a", "0", "a", "1"])
            self.assertEqual(rows[2], ["2.png", "a/2.png", "0", "a", "1", "b", "0"])
            self.assertEqual(rows[6], ["2.png", "c/2.png", "2", "c", "0", "a", "0"])
            self.assertEqual(len(rows), 7)

    def test_confusion_matrix_plot_is_a_png(self):
        for level in ("mil", "sil"):
            data = (self.root / level / "confusion_matrix_Net_T+1+2+0.1+8+6.png").read_bytes()
            self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")

    def test_existing_results_are_not_replaced_without_overwrite(self):
        code, log = self.out["refused"]
        self.assertEqual(code, 1)
        self.assertIn("would replace existing files", log)
        self.assertIn("metrics_Net_T+1+2+0.1+8+6.json", log)
        self.assertEqual(self.out["allowed_with_overwrite"][0], None)
        self.assertEqual(self.out["allowed_elsewhere"][0], None)


class OneEvaluationForTheTwoLevelsTest(unittest.TestCase):
    """Read from the source: the two scripts must go through the shared module."""

    LEVELS = ("mil_benchmark.py", "sil_benchmark.py")

    def calls(self, script):
        found = set()
        for node in ast.walk(ast.parse((SRC / script).read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and isinstance(node.func.value, ast.Name):
                found.add("%s.%s" % (node.func.value.id, node.func.attr))
        return found

    def test_both_levels_write_their_results_through_the_module(self):
        for script in self.LEVELS:
            calls = self.calls(script)
            for shared in ("evaluation.write_level_results", "evaluation.predicted_labels",
                           "evaluation.read_class_names", "evaluation.class_names_of_this_split",
                           "evaluation.refuse_to_replace_results", "evaluation.stop_without_test_split"):
                self.assertIn(shared, calls, "%s does not call %s" % (script, shared))

    def test_neither_level_computes_metrics_or_writes_files_by_itself(self):
        own = {"json.dump", "csv.writer", "plt.savefig", "sns.heatmap", "hashlib.sha256"}
        for script in self.LEVELS:
            text = (SRC / script).read_text(encoding="utf-8")
            self.assertEqual(sorted(own & self.calls(script)), [], script)
            for name in ("classification_report", "confusion_matrix(", "accuracy_score", "def _class_name",
                         "def _sha256_file", "def _json_safe"):
                self.assertNotIn(name, text, "%s still has its own %s" % (script, name))


if __name__ == "__main__":
    unittest.main()
