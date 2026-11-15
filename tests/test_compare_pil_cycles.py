#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Checks compare_pil_cycles.py, the script that puts the on-chip latency of
several PIL runs of one model side by side, on invented runs: the means are
taken over the images every run measured, the differences are relative to the
first run, the arena memory is the one the run recorded (or the place of the
board's default build, marked as not recorded), and runs that cannot be
compared are refused.

The invented files have the columns and keys pil_benchmark.py writes.
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
SCRIPT = str(REPO / "src" / "compare_pil_cycles.py")
sys.path.insert(0, str(REPO / "src"))

import compare_pil_cycles  # noqa: E402

RUN = "Tiny_COCO+1+2+0.1+8+8_int8"
HEADER = "Image;TEMP_C;CYC_PRE;CYC_INF;CYC_POST;CYC_TOTAL;US_PRE;US_INF;US_POST;US_TOTAL"
CLOCK_MHZ = 400


def write_run(folder, cycles, board=None, arena_memory=False, sha256="ab" * 32, run=RUN, unanswered=()):
    """A PIL run as pil_benchmark.py leaves it: the latency CSV and, when board is given, the metrics JSON.

    cycles is {image: (pre, inf, post)}; the total and the microseconds follow
    from them at 400 MHz. arena_memory=False is a file without that key."""
    os.makedirs(folder, exist_ok=True)
    lines = [HEADER]
    for image, (pre, inf, post) in cycles.items():
        values = [pre, inf, post, pre + inf + post]
        lines.append(";".join([image, "40"] + [str(v) for v in values]
                              + ["%.3f" % (v / CLOCK_MHZ) for v in values]))
    lines += ["%s;;;;;;;;;" % image for image in unanswered]
    Path(folder, "latency_metrics_%s.csv" % run).write_text("\n".join(lines) + "\n", encoding="utf-8")
    if board is not None:
        model = {"name": run, "sha256": sha256, "arena_used_bytes": 181492}
        if arena_memory is not False:
            model["arena_memory"] = arena_memory
        Path(folder, "metrics_%s.json" % run).write_text(
            json.dumps({"level": "PIL", "model": model, "run": {"board": board}}), encoding="utf-8")
    return folder


def run_script(*args):
    return subprocess.run([sys.executable, SCRIPT, *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", env=dict(os.environ, PYTHONIOENCODING="utf-8"))


class Runs(unittest.TestCase):
    """Three runs of one model: the whole split on two boards, and a subset on the first board with
    its arena somewhere else."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = self.tmp.name
        # published-like files: they record the board and not the arena memory
        self.sdram = write_run(os.path.join(root, "pil"), board="PORTENTA_H7", cycles={
            "a.png": (1000, 100000, 20), "b.png": (1200, 100400, 20), "c.png": (1400, 99600, 20),
            "d.png": (5000, 500000, 80)})
        self.nicla = write_run(os.path.join(root, "nicla", "pil"), board="NICLA_VISION", cycles={
            "a.png": (900, 80000, 10), "b.png": (1100, 80400, 10), "c.png": (1300, 79600, 10),
            "d.png": (4000, 400000, 40)})
        # a run of b and c only, and an image the board did not answer
        self.internal = write_run(os.path.join(root, "experiment"), board="PORTENTA_H7", arena_memory="INTERNAL",
                                  cycles={"b.png": (1150, 80300, 10), "c.png": (1350, 79700, 10)},
                                  unanswered=["a.png"])
        self.out = os.path.join(root, "tables")

    def table(self):
        with open(os.path.join(self.out, "pil_cycles_%s.csv" % RUN), encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))


class ComparisonTest(Runs):
    def test_means_are_taken_over_the_images_every_run_measured(self):
        rows, model = compare_pil_cycles.compare([self.sdram, self.nicla, self.internal], RUN)
        self.assertEqual(model, RUN)
        self.assertEqual([r["n_images_common"] for r in rows], [2, 2, 2])          # b and c
        self.assertEqual([r["n_images_run"] for r in rows], [4, 4, 2])             # a of the third had no answer
        # d, which only two runs have and is five times slower, does not enter any mean
        self.assertEqual([r["CYC_INF_mean"] for r in rows], [100000.0, 80000.0, 80000.0])
        self.assertEqual([r["CYC_PRE_mean"] for r in rows], [1300.0, 1200.0, 1250.0])
        self.assertEqual([r["CYC_POST_mean"] for r in rows], [20.0, 10.0, 10.0])
        self.assertEqual([r["CYC_TOTAL_mean"] for r in rows], [101320.0, 81210.0, 81260.0])
        self.assertAlmostEqual(rows[0]["US_INF_mean"], 250.0, places=9)
        self.assertAlmostEqual(rows[2]["US_TOTAL_mean"], 203.15, places=9)

    def test_differences_are_relative_to_the_first_run(self):
        rows, _ = compare_pil_cycles.compare([self.sdram, self.nicla, self.internal], RUN)
        self.assertEqual([r["CYC_INF_vs_first_pct"] for r in rows], [0.0, -20.0, -20.0])
        self.assertAlmostEqual(rows[2]["CYC_PRE_vs_first_pct"], 100.0 * (1250 - 1300) / 1300, places=9)
        self.assertAlmostEqual(rows[1]["US_TOTAL_vs_first_pct"], 100.0 * (81210 - 101320) / 101320, places=6)
        # with another reference the sign changes
        rows, _ = compare_pil_cycles.compare([self.nicla, self.sdram], RUN)
        self.assertEqual([r["CYC_INF_vs_first_pct"] for r in rows], [0.0, 25.0])
        self.assertEqual(rows[0]["n_images_common"], 4)

    def test_arena_memory_is_the_recorded_one_or_the_default_of_the_board(self):
        rows, _ = compare_pil_cycles.compare([self.sdram, self.nicla, self.internal], RUN)
        self.assertEqual([(r["board"], r["arena_memory"], r["arena_memory_recorded"]) for r in rows],
                         [("PORTENTA_H7", "SDRAM", False), ("NICLA_VISION", "INTERNAL", False),
                          ("PORTENTA_H7", "INTERNAL", True)])
        # a file that says null (the firmware announced nothing) is like one without the key
        write_run(self.sdram, {"b.png": (1, 2, 3)}, board="PORTENTA_H7", arena_memory=None)
        rows, _ = compare_pil_cycles.compare([self.sdram, self.internal], RUN)
        self.assertEqual((rows[0]["arena_memory"], rows[0]["arena_memory_recorded"]), ("SDRAM", False))

    def test_run_without_a_metrics_file_is_compared_with_its_board_unknown(self):
        bare = write_run(os.path.join(self.tmp.name, "bare"), {"b.png": (1300, 100000, 20)})
        rows, _ = compare_pil_cycles.compare([self.internal, bare], RUN)
        self.assertEqual((rows[1]["board"], rows[1]["arena_memory"], rows[1]["arena_memory_recorded"]),
                         (None, None, False))
        self.assertEqual(rows[1]["n_images_common"], 1)


class OutputTest(Runs):
    def test_script_writes_the_two_tables(self):
        result = run_script(self.sdram, self.nicla, self.internal, "--model_path",
                            "models/tflite/%s.tflite" % RUN, "--output_dir", self.out)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(sorted(os.listdir(self.out)), ["pil_cycles_%s.csv" % RUN, "pil_cycles_%s.md" % RUN])
        rows = self.table()
        self.assertEqual(len(rows), 3)
        self.assertEqual([(r["board"], r["arena_memory"], r["arena_memory_recorded"], r["n_images_run"],
                           r["n_images_common"]) for r in rows],
                         [("PORTENTA_H7", "SDRAM", "False", "4", "2"), ("NICLA_VISION", "INTERNAL", "False", "4", "2"),
                          ("PORTENTA_H7", "INTERNAL", "True", "2", "2")])
        for phase in compare_pil_cycles.PHASES:
            for unit in ("CYC", "US"):
                for column in ("%s_%s_mean" % (unit, phase), "%s_%s_vs_first_pct" % (unit, phase)):
                    self.assertIn(column, rows[0])
        self.assertEqual([float(r["CYC_INF_mean"]) for r in rows], [100000.0, 80000.0, 80000.0])
        self.assertEqual([float(r["CYC_INF_vs_first_pct"]) for r in rows], [0.0, -20.0, -20.0])
        # each run is named as it was given on the command line
        self.assertEqual(rows[2]["run"], self.internal.replace(os.sep, "/"))

        text = Path(self.out, "pil_cycles_%s.md" % RUN).read_text(encoding="utf-8")
        self.assertIn("2 images that all the runs measured", text)
        self.assertIn("| Portenta H7 | SDRAM (board default) | 4 | 2 | 1,300 | 100,000 | 20 | 101,320 |", text)
        self.assertIn("| Nicla Vision | INTERNAL (board default) | 4 | 2 | 1,200 | 80,000 | 10 | 81,210 |", text)
        self.assertIn("| Portenta H7 | INTERNAL | 2 | 2 | 1,250 | 80,000 | 10 | 81,260 |", text)
        self.assertIn("| 3.250 | 250.000 | 0.050 | 253.300 |", text)                     # microseconds, first run
        self.assertIn("| -3.85% | -20.00% | -50.00% | -19.80% |", text)                   # third run against the first
        self.assertIn("do not record where the arena was", text)
        self.assertEqual(result.stdout.count("| Portenta H7 |"), 2)                       # the table is also printed

    def test_a_run_can_be_given_as_one_of_its_files(self):
        result = run_script(os.path.join(self.sdram, "metrics_%s.json" % RUN),
                            os.path.join(self.internal, "latency_metrics_%s.csv" % RUN), "--output_dir", self.out)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual([r["arena_memory"] for r in self.table()], ["SDRAM", "INTERNAL"])

    def test_a_folder_with_one_run_needs_no_model_and_one_with_several_does(self):
        result = run_script(self.sdram, self.internal, "--output_dir", self.out)
        self.assertEqual(result.returncode, 0, result.stdout)
        write_run(self.sdram, {"b.png": (1, 2, 3)}, board="PORTENTA_H7", run="Other_COCO+1+2+0.1+8+8_int8")
        # the file pil_benchmark.py --image appends to is not a run
        Path(self.internal, "latency_metrics_single_%s.csv" % RUN).write_text(HEADER + "\n", encoding="utf-8")
        result = run_script(self.sdram, self.internal, "--output_dir", self.out)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("holds 2 PIL runs", result.stdout)
        self.assertIn("--model_path", result.stdout)
        result = run_script(self.sdram, self.internal, "--model_path", RUN + ".tflite", "--output_dir", self.out)
        self.assertEqual(result.returncode, 0, result.stdout)


class RefusedTest(Runs):
    def refused(self, *args, code=1):
        result = run_script(*args, "--output_dir", self.out)
        self.assertEqual(result.returncode, code, result.stdout)
        self.assertFalse(os.path.exists(self.out), "a table was written")
        return result.stdout

    def test_runs_of_different_model_files(self):
        write_run(self.internal, {"b.png": (1, 2, 3)}, board="PORTENTA_H7", arena_memory="INTERNAL", sha256="cd" * 32)
        self.assertIn("not of the same model file", self.refused(self.sdram, self.internal))

    def test_runs_without_an_image_in_common(self):
        other = write_run(os.path.join(self.tmp.name, "other"), {"z.png": (1, 2, 3)}, board="PORTENTA_H7")
        self.assertIn("no image in common", self.refused(self.sdram, other))

    def test_a_single_run(self):
        self.assertIn("at least two runs", self.refused(self.sdram, code=2))

    def test_a_run_that_is_not_there(self):
        self.assertIn("does not exist", self.refused(self.sdram, os.path.join(self.tmp.name, "nothing")))
        empty = os.path.join(self.tmp.name, "empty")
        os.makedirs(empty)
        self.assertIn("no latency_metrics_%s.csv" % RUN, self.refused(self.sdram, empty, "--model_path", RUN))

    def test_a_file_that_is_not_a_pil_latency_file(self):
        # the latency file of the HIL bench has other columns
        Path(self.internal, "latency_metrics_%s.csv" % RUN).write_text(
            "idx,utc_iso,image,cyc_inf,us_inf\n1,2026-11-15T00:00:00+00:00,b.png,80000,200.0\n", encoding="utf-8")
        self.assertIn("is not a latency file of pil_benchmark.py", self.refused(self.sdram, self.internal))

    def test_two_images_with_the_same_name(self):
        path = Path(self.internal, "latency_metrics_%s.csv" % RUN)
        path.write_text(path.read_text(encoding="utf-8") + "b.png;40;1;2;3;6;0.1;0.2;0.3;0.6\n", encoding="utf-8")
        self.assertIn("more than one row for b.png", self.refused(self.sdram, self.internal))


if __name__ == "__main__":
    unittest.main()
