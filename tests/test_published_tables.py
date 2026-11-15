#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Checks that the published tables are what src/build_ladder_tables.py builds
from the result files in git, and that the bootstrap is reproducible. There
are two sets: those of the COCO benchmark in results/tables/coco/, which the
script builds without options, and those of the vehicle benchmark in
results/tables/, which it builds with --benchmark vehicle
(tests/test_ladder_benchmarks.py checks which is which).

  - A rebuild of the vehicle benchmark gives the published tables: the eight
    CSV files, the LaTeX and the Markdown files and the summary, leaving out
    the date, the commit and the library versions (tests/tables_check.py says
    exactly what is compared).
  - Two rebuilds with the same seed give the same files; another seed changes
    the bootstrap intervals and nothing else in the two main tables.
  - tables_check.py reports a changed value and ignores the fields it is meant
    to ignore.

The rebuilds need numpy and scipy (not TensorFlow) and take about 15 seconds
in all; they are skipped when either package is missing.

    python -m unittest discover -s tests -v
"""
import csv
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import tables_check  # noqa: E402

SCRIPT = str(REPO / "src" / "build_ladder_tables.py")
PUBLISHED = REPO / "results" / "tables"
BENCHMARK = ("--benchmark", "vehicle")   # the one option that names the benchmark of those tables
READY = all(importlib.util.find_spec(name) for name in ("numpy", "scipy"))


def rebuild(out_dir, *options, benchmark=BENCHMARK):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    return subprocess.run([sys.executable, SCRIPT, *benchmark, "--output_dir", str(out_dir), *options], cwd=REPO,
                          env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


@unittest.skipUnless(READY, "build_ladder_tables.py needs numpy and scipy")
class RebuiltTablesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name) / "tables"
        cls.rebuilt = rebuild(cls.out)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_the_rebuild_gives_the_published_tables(self):
        self.assertEqual(self.rebuilt.returncode, 0, self.rebuilt.stdout[-3000:])
        found, _rounded = tables_check.differences(str(PUBLISHED), str(self.out))
        self.assertEqual(found, [], "\n".join(found[:20]))

    def test_the_tables_were_built_from_the_result_files_in_git(self):
        self.assertEqual(self.rebuilt.returncode, 0, self.rebuilt.stdout[-3000:])
        with open(PUBLISHED / "ladder_summary.json", encoding="utf-8") as fh:
            published = json.load(fh)
        with open(self.out / "ladder_summary.json", encoding="utf-8") as fh:
            rebuilt = json.load(fh)
        self.assertGreater(len(published["inputs_sha256"]), 20)
        self.assertEqual(rebuilt["inputs_sha256"], published["inputs_sha256"])
        self.assertEqual(rebuilt["parameters"], published["parameters"])


@unittest.skipUnless(READY, "build_ladder_tables.py needs numpy and scipy")
class RebuiltCocoTablesTest(unittest.TestCase):
    """The tables of the COCO benchmark, which the script builds without naming a benchmark."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name) / "tables"
        cls.rebuilt = rebuild(cls.out, benchmark=())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_the_rebuild_gives_the_published_tables(self):
        self.assertEqual(self.rebuilt.returncode, 0, self.rebuilt.stdout[-3000:])
        found, _rounded = tables_check.differences(str(PUBLISHED / "coco"), str(self.out))
        self.assertEqual(found, [], "\n".join(found[:20]))

    def test_the_tables_were_built_from_the_result_files_in_git(self):
        self.assertEqual(self.rebuilt.returncode, 0, self.rebuilt.stdout[-3000:])
        with open(PUBLISHED / "coco" / "ladder_summary.json", encoding="utf-8") as fh:
            published = json.load(fh)
        with open(self.out / "ladder_summary.json", encoding="utf-8") as fh:
            rebuilt = json.load(fh)
        self.assertGreater(len(published["inputs_sha256"]), 20)
        self.assertEqual(rebuilt["inputs_sha256"], published["inputs_sha256"])
        self.assertEqual(rebuilt["parameters"], published["parameters"])

    def test_they_have_the_repeatability_sessions(self):
        rows = read_csv(PUBLISHED / "coco" / "table4_repeatability.csv")
        self.assertEqual([row["run"] for row in rows], ["original", "rep1", "rep2", "rep3", "rep4"])
        self.assertEqual([row["role"] for row in rows[1:]],
                         ["strict replicate"] * 3 + ["presentation-order control"])


@unittest.skipUnless(READY, "build_ladder_tables.py needs numpy and scipy")
class BootstrapSeedTest(unittest.TestCase):
    """With fewer resamples than the published tables use, so that three rebuilds stay short."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dirs = {name: Path(cls.tmp.name) / name for name in ("first", "again", "other")}
        cls.runs = {
            "first": rebuild(cls.dirs["first"], "--bootstrap", "300", "--seed", "5"),
            "again": rebuild(cls.dirs["again"], "--bootstrap", "300", "--seed", "5"),
            "other": rebuild(cls.dirs["other"], "--bootstrap", "300", "--seed", "6"),
        }

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_same_seed_gives_the_same_tables(self):
        for name, run in self.runs.items():
            self.assertEqual(run.returncode, 0, "%s: %s" % (name, run.stdout[-2000:]))
        found, rounded = tables_check.differences(str(self.dirs["first"]), str(self.dirs["again"]))
        self.assertEqual(found, [], "\n".join(found[:20]))
        self.assertEqual(rounded, 0, "two runs on the same machine must not need the tolerance")

    def test_another_seed_changes_the_intervals_and_nothing_else(self):
        for table in ("table1_ladder.csv", "table2_decomposition.csv"):
            first, other = read_csv(self.dirs["first"] / table), read_csv(self.dirs["other"] / table)
            self.assertEqual(len(first), len(other))
            intervals_that_differ = 0
            for row_a, row_b in zip(first, other):
                for column in row_a:
                    if column.endswith(("_ci_lo", "_ci_hi")):
                        intervals_that_differ += (row_a[column] != row_b[column])
                    else:
                        # estimates, discordant counts and exact p-values do not come from the bootstrap
                        self.assertEqual(row_a[column], row_b[column], "%s, %s" % (table, column))
            self.assertGreater(intervals_that_differ, 0, table)

    def test_the_seed_and_the_resamples_are_recorded(self):
        with open(self.dirs["other"] / "ladder_summary.json", encoding="utf-8") as fh:
            parameters = json.load(fh)["parameters"]
        self.assertEqual((parameters["bootstrap"], parameters["seed"]), (300, 6))


class TablesCheckTest(unittest.TestCase):
    """The comparison itself, on copies of the published folder (standard library only)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.copy = Path(self.tmp.name) / "tables"
        shutil.copytree(PUBLISHED, self.copy)

    def edit(self, name, old, new, count=1):
        path = self.copy / name
        text = path.read_text(encoding="utf-8")
        self.assertGreaterEqual(text.count(old), 1, "%s not in %s" % (old, name))
        path.write_text(text.replace(old, new, count), encoding="utf-8")

    def check(self):
        return tables_check.differences(str(PUBLISHED), str(self.copy))

    def first_f1(self):
        return read_csv(PUBLISHED / "table1_ladder.csv")[0]["MIL_f1"]

    def test_a_copy_is_the_same(self):
        self.assertEqual(self.check(), ([], 0))

    def test_a_changed_value_is_reported_with_its_table_and_column(self):
        value = self.first_f1()
        self.edit("table1_ladder.csv", value, "%.15f" % (float(value) + 1e-6))
        found, _ = self.check()
        self.assertEqual(len(found), 1, found)
        self.assertIn("table1_ladder.csv row 2, MIL_f1", found[0])

    def test_a_last_digit_is_tolerated_and_counted(self):
        value = self.first_f1()
        self.edit("table1_ladder.csv", value, repr(float(value) * (1 + 1e-13)))
        self.assertEqual(self.check(), ([], 1))

    def test_date_commit_and_library_versions_are_left_out(self):
        summary = self.copy / "ladder_summary.json"
        data = json.loads(summary.read_text(encoding="utf-8"))
        data["generated_utc"] = "2030-01-01T00:00:00+00:00"
        data["git_commit"] = "0" * 40
        data["environment"] = {"python": "9.9", "numpy": "9.9", "scipy": "9.9", "platform": "another"}
        summary.write_text(json.dumps(data, indent=2), encoding="utf-8")
        markdown = self.copy / "ladder_tables.md"
        text = markdown.read_text(encoding="utf-8")
        stamped = tables_check.STAMP.sub(" on 2030-01-01T00:00:00+00:00 (commit 0000000000)", text)
        self.assertNotEqual(stamped, text, "the Markdown file has no 'Generated by ... on <date> (commit ...)' line")
        markdown.write_text(stamped, encoding="utf-8")
        self.assertEqual(self.check(), ([], 0))

    def test_a_changed_text_table_a_changed_summary_and_a_missing_file_are_reported(self):
        self.edit("ladder_tables.tex", "\\toprule", "\\midrule")
        summary = self.copy / "ladder_summary.json"
        data = json.loads(summary.read_text(encoding="utf-8"))
        data["n_test"] += 1
        summary.write_text(json.dumps(data, indent=2), encoding="utf-8")
        (self.copy / "table8_representativeness.csv").unlink()
        found, _ = self.check()
        self.assertEqual(len(found), 3, found)
        self.assertTrue(any(line.startswith("ladder_tables.tex line") for line in found))
        self.assertTrue(any("n_test" in line for line in found))
        self.assertTrue(any("table8_representativeness.csv: missing in the rebuilt folder" in line for line in found))

    def test_command_line_exit_codes(self):
        script = str(Path(__file__).resolve().parent / "tables_check.py")

        def run(*args):
            return subprocess.run([sys.executable, script, *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8")

        same = run(str(PUBLISHED), str(self.copy))
        self.assertEqual(same.returncode, 0, same.stdout)
        self.assertIn("every value is identical", same.stdout)
        value = self.first_f1()
        self.edit("table1_ladder.csv", value, "0.5")
        other = run(str(PUBLISHED), str(self.copy))
        self.assertEqual(other.returncode, 1)
        self.assertIn("DIFFERENT  table1_ladder.csv row 2, MIL_f1", other.stdout)
        self.assertEqual(run(str(PUBLISHED), str(self.copy / "nowhere")).returncode, 2)
        self.assertEqual(run(str(PUBLISHED)).returncode, 2)


if __name__ == "__main__":
    unittest.main()
