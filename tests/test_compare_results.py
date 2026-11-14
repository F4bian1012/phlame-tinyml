#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Checks compare_results.py, the script that says whether a reproduced run gives
the published result: identical runs pass whatever the timestamp and the path
separators are, and a single different prediction or a different model fails.

Standard library only.

    python -m unittest discover -s tests -v
"""
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = str(REPO / "src" / "compare_results.py")

METRICS = {
    "level": "SIL",
    "generated_utc": "2026-11-03T12:00:00+00:00",
    "model": {"path": "models/tflite/m_int8.tflite", "name": "m_int8", "sha256": "ab" * 32, "bytes": 1000},
    "dataset": {"data_dir": "data/splits_coco\\test", "n_samples": 4, "class_names": ["non_person", "person"]},
    "global_metrics": {"accuracy": 0.75, "f1_macro": 0.7333333333333334},
    "per_class_metrics": {"non_person": {"precision": 1.0, "recall": 0.5}},
    "confusion_matrix": {"labels": ["non_person", "person"], "matrix": [[1, 1], [0, 2]]},
    "artifacts": {"predictions_csv": "results\\sil\\predictions_m_int8.csv"},
}
ROWS = [("a.jpg", "non_person", 0, 0), ("b.jpg", "non_person", 0, 1), ("c.jpg", "person", 1, 1), ("d.jpg", "person", 1, 1)]


def write_run(root, metrics=METRICS, rows=ROWS, sep="\\"):
    folder = Path(root) / "sil"
    folder.mkdir(parents=True)
    (folder / "metrics_m_int8.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    lines = ["Image;RelPath;TrueLabel;TrueClass;PredLabel;PredClass;Correct"]
    for name, cls, true, pred in rows:
        lines.append("%s;%s%s%s;%d;%s;%d;x;%d" % (name, cls, sep, name, true, cls, pred, int(true == pred)))
    (folder / "predictions_m_int8.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare(candidate, reference):
    return subprocess.run([sys.executable, SCRIPT, candidate, reference],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


class CompareResultsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.published = os.path.join(self.tmp.name, "published")
        self.mine = os.path.join(self.tmp.name, "mine")
        write_run(self.published)

    def test_same_run_on_another_day_and_system_is_identical(self):
        metrics = copy.deepcopy(METRICS)
        metrics["generated_utc"] = "2026-12-01T08:00:00+00:00"
        metrics["dataset"]["data_dir"] = "data/splits_coco/test"
        metrics["artifacts"]["predictions_csv"] = "results/smoke_test/sil/predictions_m_int8.csv"
        write_run(self.mine, metrics=metrics, sep="/")
        r = compare(self.mine, self.published)
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertIn("identical  sil/metrics_m_int8.json", r.stdout)
        self.assertIn("4 images compared one by one", r.stdout)

    def test_one_different_prediction_is_reported(self):
        rows = list(ROWS)
        rows[1] = ("b.jpg", "non_person", 0, 0)
        write_run(self.mine, rows=rows)
        r = compare(self.mine, self.published)
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("1 images have a different prediction", r.stdout)
        self.assertIn("non_person/b.jpg", r.stdout)

    def test_another_model_file_or_other_metrics_are_reported(self):
        metrics = copy.deepcopy(METRICS)
        metrics["model"]["sha256"] = "cd" * 32
        metrics["global_metrics"]["accuracy"] = 0.5
        write_run(self.mine, metrics=metrics)
        r = compare(self.mine, self.published)
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("model.sha256", r.stdout)
        self.assertIn("global_metrics", r.stdout)

    def test_nothing_to_compare_is_a_failure(self):
        os.makedirs(self.mine)
        self.assertEqual(compare(self.mine, self.published).returncode, 1)
        write_run(self.mine)
        r = compare(self.mine, os.path.join(self.tmp.name, "empty"))
        self.assertEqual(r.returncode, 1)
        self.assertIn("NOT FOUND", r.stdout)

    def test_published_results_are_identical_to_themselves(self):
        r = compare(str(REPO / "results"), str(REPO / "results"))
        self.assertEqual(r.returncode, 0, r.stdout[-600:])


if __name__ == "__main__":
    unittest.main()
