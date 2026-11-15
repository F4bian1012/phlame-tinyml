#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Checks src/build_benchmark_table.py on a small invented benchmark: the figures
it copies from the metrics files, the image-by-image counts it computes from
the prediction files, and what it does when a level is missing.

Standard library only.

    python -m unittest discover -s tests -v
"""
import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = str(REPO / "src" / "build_benchmark_table.py")

IMAGES = ["non_person/a.png", "non_person/b.png", "person/c.png", "person/d.png"]
TRUE = [0, 0, 1, 1]


def write_level(folder, name, predicted, extra=None, answered=None):
    """A metrics JSON and a predictions CSV for one level, with the given predictions."""
    folder.mkdir(parents=True, exist_ok=True)
    right = sum(1 for p, t in zip(predicted, TRUE) if p == t)
    metrics = {"dataset": {"n_samples": len(TRUE)},
               "global_metrics": {"accuracy": right / len(TRUE), "f1_macro": right / len(TRUE) - 0.01},
               "model": {"sha256": "abc", "bytes": 2048}}
    for key, value in (extra or {}).items():
        metrics.setdefault(key, {}).update(value)
    (folder / ("metrics_%s.json" % name)).write_text(json.dumps(metrics), encoding="utf-8")
    with open(folder / ("predictions_%s.csv" % name), "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter=";")
        writer.writerow(["Image", "RelPath", "TrueLabel", "TrueClass", "PredLabel", "PredClass", "Correct"])
        for i, (rel, t, p) in enumerate(zip(IMAGES, TRUE, predicted)):
            if answered is not None and i >= answered:
                p = ""
            # one file uses the Windows separator, as the result files written on Windows do
            writer.writerow([rel.split("/")[1], rel.replace("/", "\\") if i == 0 else rel, t, "", p, "", ""])


def pil_extra(board, inf_us, n_answered=4):
    return {"dataset": {"n_images": 4, "n_answered": n_answered},
            "run": {"board": board, "n_reconnects": 0},
            "latency": {"per_field": {"US_INF": {"mean": inf_us}, "US_TOTAL": {"mean": inf_us + 3000.0},
                                      "CYC_INF": {"mean": inf_us * 400.0}}}}


class BenchmarkTableTest(unittest.TestCase):
    RUN = "Tiny_T+1+1+0.1+8+8"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.results = Path(self.tmp.name) / "results"
        self.out = Path(self.tmp.name) / "tables"
        write_level(self.results / "mil", self.RUN, [0, 0, 1, 1])
        write_level(self.results / "sil", self.RUN + "_int8", [0, 1, 1, 1], {"model": {"kernels": "reference"}})
        write_level(self.results / "pil", self.RUN + "_int8", [0, 1, 1, 1], pil_extra("PORTENTA_H7", 400000.0))
        write_level(self.results / "nicla" / "pil", self.RUN + "_int8", [0, 1, 1, 0], pil_extra("NICLA_VISION", 320000.0))
        write_level(self.results / "sil_kernels" / "default", self.RUN + "_int8", [0, 0, 1, 1],
                    {"model": {"kernels": "default"}})
        # a run with another tag, which must be left out
        write_level(self.results / "mil", "Tiny_OTHER+1+1+0.1+8+8", [0, 0, 1, 1])

    def tearDown(self):
        self.tmp.cleanup()

    def run_script(self, *args):
        # The output is read as UTF-8, so the script is asked to write it: on Windows it would
        # use the code page of the console, and the table has characters outside ASCII.
        return subprocess.run([sys.executable, SCRIPT, "--results_dir", str(self.results),
                               "--output_dir", str(self.out), *args],
                              env=dict(os.environ, PYTHONIOENCODING="utf-8"),
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")

    def row(self):
        with open(self.out / "benchmark_T.csv", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 1)
        return rows[0]

    def test_figures_and_counts(self):
        result = self.run_script("--tag", "T")
        self.assertEqual(result.returncode, 0, result.stdout)
        row = self.row()
        self.assertEqual((row["run"], row["architecture"], row["n_test"]), (self.RUN, "Tiny", "4"))
        self.assertEqual((float(row["mil_accuracy"]), float(row["sil_accuracy"])), (1.0, 0.75))
        self.assertEqual((row["tflite_bytes"], row["sil_kernels"]), ("2048", "reference"))
        self.assertEqual(row["mil_eq_sil"], "3/4")
        self.assertEqual((row["sil_eq_pil_portenta"], row["sil_eq_pil_nicla"]), ("4/4", "3/4"))
        self.assertEqual(row["portenta_eq_nicla"], "3/4")
        self.assertEqual((float(row["pil_portenta_inf_ms"]), float(row["pil_nicla_total_ms"])), (400.0, 323.0))
        self.assertEqual((row["sil_default_eq_sil"], row["sil_default_eq_pil_portenta"]), ("3/4", "3/4"))
        self.assertEqual(row["pil_portenta_same_file_as_sil"], "True")
        text = (self.out / "benchmark_T.md").read_text(encoding="utf-8")
        self.assertIn("| `Tiny` | 2.0 KiB | 4 | 1.0000 / 0.9900 | 0.7500 / 0.7400 | 3 / 4 |", text)
        self.assertIn("SIL `default`", text)
        self.assertNotIn("OTHER", text)

    def test_images_a_board_did_not_answer_are_not_compared(self):
        write_level(self.results / "pil", self.RUN + "_int8", [0, 1, 1, 1], pil_extra("PORTENTA_H7", 400000.0, 3),
                    answered=3)
        self.assertEqual(self.run_script("--tag", "T").returncode, 0)
        row = self.row()
        self.assertEqual((row["sil_eq_pil_portenta"], row["pil_portenta_n_answered"]), ("3/3", "3"))
        self.assertIn("(3 answered)", (self.out / "benchmark_T.md").read_text(encoding="utf-8"))

    def test_levels_that_were_not_run_are_left_empty(self):
        for name in ("pil", "nicla", "sil_kernels"):
            for path in sorted((self.results / name).rglob("*"), reverse=True):
                path.unlink() if path.is_file() else path.rmdir()
        result = self.run_script("--tag", "T")
        self.assertEqual(result.returncode, 0, result.stdout)
        row = self.row()
        self.assertNotIn("pil_portenta_accuracy", row)
        self.assertEqual(row["mil_eq_sil"], "3/4")
        text = (self.out / "benchmark_T.md").read_text(encoding="utf-8")
        self.assertNotIn("other TFLite kernels", text)

    def test_a_board_run_with_another_model_file_is_reported(self):
        write_level(self.results / "pil", self.RUN + "_int8", [0, 1, 1, 1],
                    dict(pil_extra("PORTENTA_H7", 400000.0), model={"sha256": "another"}))
        result = self.run_script("--tag", "T")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("names another .tflite than the SIL run", result.stdout)

    def test_unknown_tag_and_unknown_board(self):
        self.assertEqual(self.run_script("--tag", "NOPE").returncode, 1)
        self.assertEqual(self.run_script("--tag", "T", "--boards", "portenta,uno").returncode, 2)


class AttributionTest(unittest.TestCase):
    """Two models whose SIL predictions differ; each PIL run must agree with its own."""

    A, B = "Alpha_T+1+1+0.1+8+8", "Beta_T+1+1+0.1+8+8"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.results = Path(self.tmp.name) / "results"
        self.out = Path(self.tmp.name) / "tables"
        for run, sil, size in ((self.A, [0, 0, 1, 1], 1024), (self.B, [1, 1, 1, 0], 4096)):
            write_level(self.results / "mil", run, sil)
            write_level(self.results / "sil", run + "_int8", sil, {"model": {"kernels": "reference", "bytes": size}})
        write_level(self.results / "pil", self.A + "_int8", [0, 0, 1, 1], pil_extra("PORTENTA_H7", 100000.0))
        write_level(self.results / "nicla" / "pil", self.A + "_int8", [0, 0, 1, 0], pil_extra("NICLA_VISION", 90000.0))

    def tearDown(self):
        self.tmp.cleanup()

    run_script = BenchmarkTableTest.run_script

    def attribution(self):
        with open(self.out / "benchmark_T_attribution.csv", encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))

    def test_each_run_is_compared_with_the_sil_of_every_model(self):
        result = self.run_script("--tag", "T")
        self.assertEqual(result.returncode, 0, result.stdout)
        rows = self.attribution()
        self.assertEqual([(r["architecture"], r["board"]) for r in rows], [("Alpha", "portenta"), ("Alpha", "nicla")])
        self.assertEqual((rows[0]["same_as_sil_Alpha"], rows[0]["same_as_sil_Beta"], rows[0]["attributed"]),
                         ("4", "1", "True"))
        self.assertEqual((rows[1]["own"], rows[1]["best_other"], rows[1]["best_other_architecture"]),
                         ("3", "2", "Beta"))
        text = (self.out / "benchmark_T.md").read_text(encoding="utf-8")
        self.assertIn("## Which model each PIL run agrees with", text)
        self.assertIn("| `Alpha` | Portenta H7 | 4 | **4** | 1 | yes |", text)

    def test_a_board_that_carried_the_other_model_is_reported(self):
        # the run is filed under Alpha, but its predictions are those of Beta
        write_level(self.results / "pil", self.A + "_int8", [1, 1, 1, 0], pil_extra("PORTENTA_H7", 100000.0))
        result = self.run_script("--tag", "T")
        self.assertEqual(result.returncode, 3, result.stdout)
        self.assertIn("agrees more with the SIL of Beta", result.stdout)
        self.assertEqual(self.attribution()[0]["attributed"], "False")
        self.assertIn("**NO**", (self.out / "benchmark_T.md").read_text(encoding="utf-8"))

    def test_without_pil_runs_there_is_no_attribution_table(self):
        for folder in ("pil", "nicla"):
            for path in sorted((self.results / folder).rglob("*"), reverse=True):
                path.unlink() if path.is_file() else path.rmdir()
        self.assertEqual(self.run_script("--tag", "T").returncode, 0)
        self.assertFalse((self.out / "benchmark_T_attribution.csv").exists())
        self.assertNotIn("Which model each PIL run", (self.out / "benchmark_T.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
