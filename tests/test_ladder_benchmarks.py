#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Checks that src/build_ladder_tables.py builds the tables of a benchmark other
than the vehicle one with one command line.

Two benchmarks are used. An invented one ("ZOO": three architectures, classes
cat and dog, its own tag, run suffix, partition and rig subset), written to a
temporary folder with predictions whose macro-F1 the test computes on its own;
and the Visual Wake Words run that is in the repository, whose figures are the
ones the README reports. The tables of the vehicle benchmark, which the script
builds without options, are covered by tests/test_published_tables.py.

Needs numpy and scipy (the packages of the script); skipped without them.

    python -m unittest discover -s tests -v
"""
import csv
import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = str(REPO / "src" / "build_ladder_tables.py")
READY = all(importlib.util.find_spec(name) for name in ("numpy", "scipy"))

TAG, SUFFIX = "ZOO", "+4+3+0.01+8+6"
ARCHS, DEPLOYABLE = ["Tiny", "Small", "Huge"], ["Tiny", "Small"]
CLASSES = ["cat", "dog"]
N_PER_CLASS, RIG_PER_CLASS = 20, 6
IMAGES = [("%s_%03d.png" % (CLASSES[label], i), label) for label in (0, 1) for i in range(N_PER_CLASS)]
RIG = [name for name, _ in IMAGES if int(name[4:7]) < RIG_PER_CLASS]
TRUE = dict(IMAGES)
# share of wrong answers of each level, so that the levels differ
ERROR_RATE = {"MIL": 0.10, "SIL": 0.15, "PIL": 0.20, "HIL": 0.30, "MILc": 0.20, "SILc": 0.25}
OPTIONS = ["--tag", TAG, "--run_suffix", SUFFIX, "--splits_dir", "data/splits_zoo",
           "--archs", ",".join(ARCHS), "--deployable", ",".join(DEPLOYABLE), "--bootstrap", "200"]


def run_of(arch):
    return "%s_%s%s" % (arch, TAG, SUFFIX)


def invented_predictions():
    """{(level, architecture): {image: predicted label}}, the same on every call."""
    rng = random.Random(20261115)
    out = {}
    for arch in ARCHS:
        for level in ("MIL", "SIL", "PIL", "HIL", "MILc", "SILc"):
            names = RIG if level in ("HIL", "MILc", "SILc") else [name for name, _ in IMAGES]
            out[(level, arch)] = {name: (1 - TRUE[name]) if rng.random() < ERROR_RATE[level] else TRUE[name]
                                  for name in names}
    return out


def macro_f1(true, predicted):
    """Macro-F1 of two lists of 0/1 labels, written out from the definition."""
    total = 0.0
    for c in (0, 1):
        tp = sum(1 for t, p in zip(true, predicted) if t == c and p == c)
        fp = sum(1 for t, p in zip(true, predicted) if t != c and p == c)
        fn = sum(1 for t, p in zip(true, predicted) if t == c and p != c)
        total += 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return total / 2


def write_predictions(path, predicted):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter=";")
        writer.writerow(["Image", "RelPath", "TrueLabel", "TrueClass", "PredLabel", "PredClass", "Correct"])
        for name, label in IMAGES:
            p = predicted[name]
            writer.writerow([name, "%s/%s" % (CLASSES[label], name), label, CLASSES[label], p, CLASSES[p], int(p == label)])


def latency(*fields):
    return {"latency": {"per_field": {f: {"mean": 1000.0 * (i + 1), "std": 10.0, "n": 5} for i, f in enumerate(fields)}}}


def write_hil(folder, stem, pred, announces, seed=42):
    """latency_metrics, metrics and conditions of one HIL run; pred = (HIL, SIL on the frame, MIL on the frame)."""
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / ("latency_metrics_%s.csv" % stem), "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["idx", "image", "true_label", "true_class", "pred_label", "sil_label", "mil_label"])
        # shuffled on purpose: the script must line the rig images up by name
        for i, name in enumerate(sorted(RIG, key=lambda n: n[::-1])):
            writer.writerow([i + 1, name, TRUE[name], CLASSES[TRUE[name]], pred[0][name], pred[1][name], pred[2][name]])
    metrics = dict(latency("US_CAPTURE", "US_INF", "US_TOTAL", "CYC_INF"), run={"board": announces})
    (folder / ("metrics_%s.json" % stem)).write_text(json.dumps(metrics), encoding="utf-8")
    conditions = {"seed": seed, "frame_dump": True, "firmware_model": {"arena_used_bytes": 123456}}
    (folder / ("conditions_%s.json" % stem)).write_text(json.dumps(conditions), encoding="utf-8")


def build_benchmark(root, board_folder="results", announces="PORTENTA_H7"):
    """Writes the invented benchmark under root; returns its predictions."""
    root = Path(root)
    pred = invented_predictions()
    splits = root / "data" / "splits_zoo"
    splits.mkdir(parents=True)
    with open(splits / "test_manifest.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["split", "class_name", "label", "filename", "src_path", "dst_path", "hil_subset"])
        for name, label in IMAGES:
            writer.writerow(["test", CLASSES[label], label, name, "", "data/splits_zoo/test/%s/%s" % (CLASSES[label], name),
                             int(name in RIG)])
    board_root = root.joinpath(*board_folder.split("/"))
    for arch in ARCHS:
        run = run_of(arch)
        write_predictions(root / "results" / "mil" / ("predictions_%s.csv" % run), pred[("MIL", arch)])
        write_predictions(root / "results" / "sil" / ("predictions_%s_int8.csv" % run), pred[("SIL", arch)])
        size = 3 * 1024 * 1024 if arch == "Huge" else 300 * 1024
        (root / "results" / "sil" / ("metrics_%s_int8.json" % run)).write_text(
            json.dumps({"model": {"bytes": size, "sha256": "0" * 64, "kernels": "reference"}}), encoding="utf-8")
        # An alternative SIL run with every answer wrong: it must never be taken for the SIL level.
        write_predictions(root / "results" / "sil_kernels" / "default" / ("predictions_%s_int8.csv" % run),
                          {name: 1 - label for name, label in IMAGES})
        if arch in DEPLOYABLE:
            write_predictions(board_root / "pil" / ("predictions_%s_int8.csv" % run), pred[("PIL", arch)])
            metrics = dict(latency("US_PRE", "US_INF", "US_POST", "US_TOTAL", "CYC_INF"), run={"board": announces})
            (board_root / "pil" / ("metrics_%s_int8.json" % run)).write_text(json.dumps(metrics), encoding="utf-8")
            write_hil(board_root / "hil", run + "_int8",
                      (pred[("HIL", arch)], pred[("SILc", arch)], pred[("MILc", arch)]), announces)
    # One repeatability run of this benchmark's reference architecture, and one of another benchmark.
    reference = pred[("HIL", DEPLOYABLE[0])]
    write_hil(board_root / "hil" / "rep1", "%s_%s_rep1" % (DEPLOYABLE[0], TAG), (reference, reference, reference), announces)
    write_hil(board_root / "hil" / "rep7", "%s_rep7" % DEPLOYABLE[0], (reference, reference, reference), announces)
    return pred


def run_script(root, *options):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    return subprocess.run([sys.executable, SCRIPT, "--root", str(root), *options], cwd=REPO, env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


@unittest.skipUnless(READY, "build_ladder_tables.py needs numpy and scipy")
class SecondBenchmarkTest(unittest.TestCase):
    """The invented benchmark, built once; every test reads the same tables."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.pred = build_benchmark(cls.root)
        cls.result = run_script(cls.root, *OPTIONS)
        cls.out = cls.root / "results" / "tables" / "zoo"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_tables_go_to_a_folder_of_their_own(self):
        self.assertEqual(self.result.returncode, 0, self.result.stdout[-3000:])
        names = sorted(p.name for p in self.out.iterdir())
        self.assertEqual(len([n for n in names if n.startswith("table") and n.endswith(".csv")]), 8, names)
        for name in ("ladder_tables.md", "ladder_tables.tex", "ladder_summary.json"):
            self.assertIn(name, names)
        # the folder of the default benchmark is left alone
        self.assertEqual([p.name for p in (self.root / "results" / "tables").iterdir()], ["zoo"])

    def test_every_level_is_the_one_of_its_own_folder(self):
        true = [label for _, label in IMAGES]
        rig_true = [TRUE[name] for name in RIG]
        rows = {row["arch"]: row for row in read_csv(self.out / "table1_ladder.csv")}
        self.assertEqual(list(rows), ARCHS)
        for arch in ARCHS:
            for level in ("MIL", "SIL") + (("PIL",) if arch in DEPLOYABLE else ()):
                expected = macro_f1(true, [self.pred[(level, arch)][name] for name, _ in IMAGES])
                self.assertAlmostEqual(float(rows[arch]["%s_f1" % level]), expected, places=12, msg="%s %s" % (level, arch))
            if arch in DEPLOYABLE:
                expected = macro_f1(rig_true, [self.pred[("HIL", arch)][name] for name in RIG])
                self.assertAlmostEqual(float(rows[arch]["HIL_f1"]), expected, places=12)
                expected = macro_f1(rig_true, [self.pred[("PIL", arch)][name] for name in RIG])
                self.assertAlmostEqual(float(rows[arch]["PIL200_f1"]), expected, places=12)
        self.assertEqual([rows[a]["exceeds_flash"] for a in ARCHS], ["False", "False", "True"])
        self.assertEqual(rows["Huge"]["PIL_f1"], "")
        # camera-frame columns of the decomposition come from the HIL file of the same run
        frames = {row["arch"]: row for row in read_csv(self.out / "table2_decomposition.csv")}
        expected = macro_f1(rig_true, [self.pred[("SILc", "Small")][name] for name in RIG])
        self.assertAlmostEqual(float(frames["Small"]["f1_SILc"]), expected, places=12)

    def test_alternative_sil_runs_are_not_read(self):
        with open(self.out / "ladder_summary.json", encoding="utf-8") as fh:
            inputs = json.load(fh)["inputs_sha256"]
        self.assertTrue(any(path.startswith("results/sil/") for path in inputs))
        self.assertFalse([path for path in inputs if "sil_kernels" in path])

    def test_texts_follow_the_benchmark(self):
        text = (self.out / "ladder_tables.md").read_text(encoding="utf-8")
        self.assertIn("# PHLAME - fidelity-ladder tables (ZOO)", text)
        self.assertIn("Runs named `<architecture>_ZOO+4+3+0.01+8+6`", text)
        self.assertIn("| PIL (12) | HIL (12) |", text)
        self.assertIn("Recall cat | Recall dog", text)
        self.assertIn("n = 40 test images and 12 rig images (6/6 per class)", text)
        self.assertNotIn("(200)", text)
        self.assertNotIn("Class1", text)
        tex = (self.out / "ladder_tables.tex").read_text(encoding="utf-8")
        self.assertIn("\\label{tab:ladder-zoo-1}", tex)

    def test_summary_records_what_defines_the_benchmark(self):
        with open(self.out / "ladder_summary.json", encoding="utf-8") as fh:
            summary = json.load(fh)
        self.assertEqual((summary["n_test"], summary["n_rig"], summary["rig_matches_manifest"]), (40, 12, True))
        benchmark = summary["parameters"]["benchmark"]
        self.assertEqual((benchmark["tag"], benchmark["run_suffix"], benchmark["board"]), (TAG, SUFFIX, "PORTENTA_H7"))
        self.assertEqual(benchmark["manifest"], "data/splits_zoo/test_manifest.csv")
        self.assertEqual(benchmark["class_names"], CLASSES)

    def test_repeatability_runs_of_other_benchmarks_are_left_out(self):
        rows = read_csv(self.out / "table4_repeatability.csv")
        self.assertEqual([row["run"] for row in rows], ["original", "rep1"])
        self.assertIn("left out: rep7", self.result.stdout)
        # the sentence about the first run of the vehicle benchmark is not said of another benchmark
        self.assertEqual([row["role"] for row in rows],
                         ["run of the ladder (not in the mean of the repeatability runs)", "strict replicate"])


@unittest.skipUnless(READY, "build_ladder_tables.py needs numpy and scipy")
class RefusedBenchmarkTest(unittest.TestCase):
    """What the script says when the benchmark it is asked for cannot be tabulated."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        build_benchmark(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def refused(self, *options):
        result = run_script(self.root, *options)
        self.assertEqual(result.returncode, 1, result.stdout[-3000:])
        self.assertNotIn("Traceback", result.stdout)
        self.assertFalse((self.root / "results" / "tables").exists(), "nothing is written by a refused run")
        return result.stdout

    def test_a_level_that_was_not_measured_is_named(self):
        for path in (self.root / "results" / "hil").glob("*%s*" % run_of("Small")):
            path.unlink()
        text = self.refused(*OPTIONS)
        self.assertIn("need the four levels", text)
        self.assertIn("HIL: Small", text)
        self.assertNotIn("PIL:", text)
        self.assertIn("build_benchmark_table.py", text)

    def test_a_wrong_run_name_lists_the_runs_there_are(self):
        options = [o for o in OPTIONS if o not in ("--run_suffix", SUFFIX)]
        text = self.refused(*options)
        self.assertIn("No MIL run is named Tiny_ZOO+32+20+0.0001+160+120", text)
        self.assertIn("results/mil has: Tiny_ZOO+4+3+0.01+8+6", text)

    def test_the_partition_of_another_benchmark_is_refused(self):
        manifest = self.root / "data" / "splits_zoo" / "test_manifest.csv"
        original = manifest.read_text(encoding="utf-8")
        manifest.write_text(original.replace("cat", "apple").replace("dog", "pear"), encoding="utf-8")
        self.assertIn("does not list the 40 test images", self.refused(*OPTIONS))
        manifest.write_text(original.replace(",cat,", ",apple,").replace(",dog,", ",pear,"), encoding="utf-8")
        self.assertIn("Class names differ", self.refused(*OPTIONS))
        manifest.unlink()
        self.assertIn("Test manifest not found: data/splits_zoo/test_manifest.csv", self.refused(*OPTIONS))

    def test_runs_of_another_board_are_refused(self):
        (self.root / "results" / "nicla").mkdir()
        shutil.move(str(self.root / "results" / "pil"), str(self.root / "results" / "nicla" / "pil"))
        shutil.move(str(self.root / "results" / "hil"), str(self.root / "results" / "nicla" / "hil"))
        text = self.refused(*OPTIONS, "--board", "nicla")
        self.assertIn("was measured on PORTENTA_H7, not on the board given with --board (NICLA_VISION)", text)


@unittest.skipUnless(READY, "build_ladder_tables.py needs numpy and scipy")
class OptionsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_board_chooses_the_folders(self):
        build_benchmark(self.root, board_folder="results/nicla", announces="NICLA_VISION")
        result = run_script(self.root, *OPTIONS, "--board", "nicla")
        self.assertEqual(result.returncode, 0, result.stdout[-3000:])
        out = self.root / "results" / "nicla" / "tables" / "zoo"
        with open(out / "ladder_summary.json", encoding="utf-8") as fh:
            summary = json.load(fh)
        self.assertEqual(summary["parameters"]["benchmark"]["board"], "NICLA_VISION")
        self.assertTrue(any(path.startswith("results/nicla/hil/") for path in summary["inputs_sha256"]))
        text = (out / "ladder_tables.md").read_text(encoding="utf-8")
        self.assertIn("fidelity-ladder tables (ZOO, Nicla Vision)", text)
        self.assertIn("captured by the GC2145", text)
        self.assertIn("\\label{tab:ladder-zoo-niclavision-1}", (out / "ladder_tables.tex").read_text(encoding="utf-8"))

    def test_earlier_option_name_and_output_folder(self):
        build_benchmark(self.root)
        options = ["--tag_suffix" if o == "--run_suffix" else o for o in OPTIONS]
        result = run_script(self.root, *options, "--output_dir", "elsewhere")
        self.assertEqual(result.returncode, 0, result.stdout[-3000:])
        self.assertTrue((self.root / "elsewhere" / "table1_ladder.csv").is_file())
        self.assertFalse((self.root / "results" / "tables").exists())

    def test_class_names_without_a_manifest_that_has_them(self):
        build_benchmark(self.root)
        manifest = self.root / "data" / "splits_zoo" / "test_manifest.csv"
        rows = read_csv(manifest)
        with open(manifest, "w", encoding="utf-8", newline="") as fh:   # an older manifest: no class columns
            writer = csv.writer(fh)
            writer.writerow(["filename", "hil_subset"])
            writer.writerows([row["filename"], row["hil_subset"]] for row in rows)
        result = run_script(self.root, *OPTIONS)
        self.assertEqual(result.returncode, 0, result.stdout[-3000:])
        self.assertIn("Recall cat | Recall dog", (self.root / "results" / "tables" / "zoo" / "ladder_tables.md")
                      .read_text(encoding="utf-8"))
        # a file given with --class_names is used, and must agree with what the runs recorded
        (self.root / "names.txt").write_text("cat\ndog\n", encoding="utf-8")
        self.assertEqual(run_script(self.root, *OPTIONS, "--class_names", "names.txt").returncode, 0)
        (self.root / "names.txt").write_text("dog\ncat\n", encoding="utf-8")
        swapped = run_script(self.root, *OPTIONS, "--class_names", "names.txt")
        self.assertEqual(swapped.returncode, 1)
        self.assertIn("Class names differ", swapped.stdout)


@unittest.skipUnless(READY, "build_ladder_tables.py needs numpy and scipy")
class VisualWakeWordsTest(unittest.TestCase):
    """A real second benchmark: the four-level Visual Wake Words run published in the repository."""

    def test_one_command_gives_the_figures_of_the_readme(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_script(REPO, "--tag", "VWW", "--archs", "MobileNetV2", "--deployable", "MobileNetV2",
                                "--splits_dir", "data/splits_vww", "--bootstrap", "200", "--output_dir", tmp)
            self.assertEqual(result.returncode, 0, result.stdout[-3000:])
            row = read_csv(Path(tmp) / "table1_ladder.csv")[0]
            frames = read_csv(Path(tmp) / "table2_decomposition.csv")[0]
            with open(Path(tmp) / "ladder_summary.json", encoding="utf-8") as fh:
                summary = json.load(fh)
        # README, "A Four-Level Run on Public Images: Visual Wake Words"
        published = {"MIL_f1": "0.6892", "SIL_f1": "0.6765", "PIL_f1": "0.6865", "PIL200_f1": "0.7196", "HIL_f1": "0.6795"}
        self.assertEqual({key: "%.4f" % float(row[key]) for key in published}, published)
        self.assertEqual(("%.4f" % float(frames["f1_MILc"]), "%.4f" % float(frames["f1_SILc"])), ("0.6748", "0.6892"))
        self.assertEqual((summary["n_test"], summary["n_rig"]), (300, 200))
        self.assertEqual(summary["parameters"]["benchmark"]["class_names"], ["non_person", "person"])
        # the repeatability runs in results/hil/rep*/ belong to the vehicle benchmark
        self.assertEqual(summary["parameters"]["benchmark"]["repeatability_runs"], [])
        self.assertIsNone(summary["frontend_gap_reference"])


if __name__ == "__main__":
    unittest.main()
