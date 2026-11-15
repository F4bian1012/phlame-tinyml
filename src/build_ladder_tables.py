#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_ladder_tables.py - Reproduce the statistical tables of the PHLAME fidelity
ladder (MIL -> SIL -> PIL -> HIL) from the per-image result files in results/.

Run from the repository root:

    python src/build_ladder_tables.py
    python src/build_ladder_tables.py --benchmark vehicle
    python src/build_ladder_tables.py --output_dir results/smoke_test/tables/coco --bootstrap 10000

Without options it builds the tables of the COCO benchmark (person / non_person
on MS-COCO), the benchmark of the repository, in results/tables/coco.
--benchmark vehicle builds those of the vehicle application case
(docs/vehicle_benchmark.md) in results/tables, where they are published. Another
benchmark that has the four levels is one command, here the Visual Wake Words run:

    python src/build_ladder_tables.py --tag VWW --run_suffix +32+20+0.0001+160+120 --splits_dir data/splits_vww \
        --archs MobileNetV2 --deployable MobileNetV2

  --benchmark    coco (default) or vehicle: the tag, the run suffix and the partition of
                 one of the two published benchmarks. Each of the next three options
                 replaces its part, so the three together name any other benchmark
  --tag          tag of the runs, as given to the training scripts: the runs are named
                 <Architecture>_<tag><run_suffix>; with an empty tag (--tag ""),
                 <Architecture><run_suffix>
  --run_suffix   the rest of the run name: +batch+epochs+learning rate+width+height
  --splits_dir   partition of the benchmark; its test_manifest.csv gives the rig subset
                 (hil_subset) and the class names
  --archs, --deployable   architectures at MIL and SIL, and those that also run on the board
  --board        board of the PIL and HIL runs: results/{pil,hil} for the Portenta H7,
                 results/nicla/{pil,hil} for the Nicla Vision
  --output_dir   default results/tables/<tag in lower case> (results/tables/coco), and
                 results/tables for runs without a tag (the vehicle benchmark), so that
                 two benchmarks do not replace each other's tables

Inputs (<run> = <Architecture>[_<tag>]<run_suffix>, by default
<Architecture>_COCO+32+50+0.0003+160+120; n is given for the COCO benchmark):
  results/mil/predictions_<run>.csv           MIL: float model on the host        (n = 5,000)
  results/sil/predictions_<run>_int8.csv      SIL: INT8 model on the host         (n = 5,000)
  results/sil/metrics_<run>_int8.json         exact .tflite size and SHA-256
  results/pil/predictions_<run>_int8.csv      PIL: INT8 on the MCU, images sent over serial (n = 5,000)
  results/pil/metrics_<run>_int8.json         PIL per-phase latency (DWT cycle counter)
  results/hil/latency_metrics_<run>_int8.csv  HIL: INT8 on the MCU, camera frames (n = 200), plus the
                                              host float/INT8 re-classification of the same frame
  results/hil/metrics_<run>_int8.json         HIL per-phase latency
  results/hil/conditions_<run>_int8.json      firmware fingerprint and tensor-arena usage
  results/hil/rep*/                           HIL repeatability runs of the reference architecture,
                                              named <Architecture>[_<tag>]_<rep folder>; a rep folder
                                              without a run of this benchmark is left out
  <splits_dir>/test_manifest.csv              hil_subset flag (the rig images) and class names

Only these folders are read: the SIL of record is the one in results/sil/, and the
alternative SIL runs of results/sil_kernels/ are never taken for a level. The script
stops with the list of what is missing when a level has not been measured; a
benchmark measured up to PIL is summarised by build_benchmark_table.py instead.

Outputs (in --output_dir):
  ladder_tables.md          the eight tables, formatted
  ladder_tables.tex         the same tables in LaTeX (booktabs)
  table<k>_<name>.csv       unrounded values of every table
  ladder_summary.json       headline numbers and provenance (input SHA-256, seeds,
                            library versions, git commit; and the options that define
                            the benchmark, except in the summary of the vehicle
                            benchmark, which keeps the keys of the published file)

Methods
  * Metric: binary macro-averaged F1. Differences are computed from unrounded values.
  * 95% CIs: paired percentile bootstrap with B image resamples. The same resample
    indices are applied to the two conditions being compared, and to the two
    architectures in between-architecture contrasts, because they share images.
  * Tests: exact McNemar (two-sided binomial test on the discordant images);
    Fisher exact test for per-class sensitivity; Holm correction within each family.
  * Determinism: with the default seeds the bootstrap index matrices are drawn in a
    fixed order (full test set first, then the rig subset), so repeated runs give
    identical numbers.

Note: the keys of the metrics_*.json files read here ('model', 'latency',
'per_field', 'mean', 'std') are documented in "Results Format"
in the README.

Dependencies: numpy and scipy (scipy is installed with scikit-learn). pandas is not needed.
"""

import argparse
import csv
import hashlib
import itertools
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import scipy
from scipy.stats import binomtest, fisher_exact
from scipy.stats import t as t_dist
import boards
import cli

# What names the runs and the partition of the two published benchmarks. COCO is the
# benchmark of the repository and the one built without options; the vehicle benchmark is
# the application case (docs/vehicle_benchmark.md), whose runs carry no tag.
BENCHMARKS = {
    "coco": {"tag": "COCO", "run_suffix": "+32+50+0.0003+160+120", "splits_dir": "data/splits_coco"},
    "vehicle": {"tag": None, "run_suffix": "+32+20+0.0001+160+120", "splits_dir": "data/splits_plates"},
}
DEFAULT_BENCHMARK = "coco"
VEHICLE = BENCHMARKS["vehicle"]
# The same architectures in both benchmarks.
DEFAULT_ARCHS = "MobileNetV2,MobileNetV3Large,ResNet18,MobileNet,MobileNetV3Small,SqueezeNet,ResNet8"
DEFAULT_DEPLOYABLE = "MobileNetV2,MobileNet,MobileNetV3Small,SqueezeNet,ResNet8"
FLASH_BYTES = 2 * 1024 * 1024  # STM32H747XI internal flash: 2 MiB


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------
def parse_args(argv=None):
    of_default = BENCHMARKS[DEFAULT_BENCHMARK]
    p = cli.ArgumentParser(
        description="Rebuild the fidelity-ladder tables (paired bootstrap CIs, exact McNemar, "
                    "Holm) from the per-image result files in results/. Without options: the "
                    "COCO benchmark, the benchmark of the repository; --benchmark vehicle: the "
                    "vehicle application case; --tag, --run_suffix and --splits_dir name another "
                    "benchmark that has the four levels.")
    p.add_argument("--root", default=".",
                   help="Repository root; all other paths are relative to it (default: .)")
    p.add_argument("--benchmark", choices=sorted(BENCHMARKS), default=DEFAULT_BENCHMARK,
                   help="Published benchmark to build: its tag, run suffix and partition (default: %s). "
                        "vehicle: runs without a tag, suffix %s, partition %s, tables in results/tables. "
                        "--tag, --run_suffix and --splits_dir each replace their part"
                        % (DEFAULT_BENCHMARK, VEHICLE["run_suffix"], VEHICLE["splits_dir"]))
    p.add_argument("--output_dir", default=None,
                   help="Output folder (default: results/tables/<tag in lower case>, which is "
                        "results/tables/%s without options; results/tables for runs without a tag, as "
                        "with --benchmark vehicle; under results/nicla/ with --board nicla)"
                        % of_default["tag"].lower())
    p.add_argument("--tag", default=None,
                   help="Tag of the runs, as given to the training scripts with --tag: the runs are "
                        "named {architecture}_{tag}{run_suffix} (default: %s, or that of --benchmark). "
                        "An empty tag (--tag \"\") is for runs trained without one, which are named "
                        "{architecture}{run_suffix}" % of_default["tag"])
    p.add_argument("--run_suffix", default=None,
                   help="Rest of the run name after the architecture and the tag: "
                        "+batch+epochs+learning rate+width+height (default: %s, or that of --benchmark)"
                        % of_default["run_suffix"])
    # Name of --run_suffix until November 2026; still accepted.
    p.add_argument("--tag_suffix", dest="run_suffix", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    p.add_argument("--archs", default=DEFAULT_ARCHS,
                   help="Comma-separated architectures evaluated at MIL and SIL. The first one "
                        "defines the image order of the test set.")
    p.add_argument("--deployable", default=DEFAULT_DEPLOYABLE,
                   help="Comma-separated architectures that fit on the MCU (have PIL and HIL runs). "
                        "The first one is the reference for the rig subset and the repeatability runs.")
    p.add_argument("--splits_dir", default=None,
                   help="Partition folder of the benchmark, as written by split_dataset.py; its "
                        "test_manifest.csv gives the rig subset (hil_subset) and the class names "
                        "(default: %s, or that of --benchmark)" % of_default["splits_dir"])
    p.add_argument("--manifest", default=None,
                   help="Test manifest with the hil_subset column, when it is not "
                        "<splits_dir>/test_manifest.csv (checked if present)")
    p.add_argument("--class_names", default=None,
                   help="File with the two class names, one per line, in label order. Default: the "
                        "names in the test manifest, else those recorded in the result files")
    p.add_argument("--board", choices=sorted(boards.BOARDS), default=boards.DEFAULT_BOARD,
                   help="Board of the PIL and HIL runs (default: %s). It sets the folders they are read "
                        "from: results/{pil,hil}, or results/nicla/{pil,hil} for the Nicla Vision"
                        % boards.DEFAULT_BOARD)
    p.add_argument("--bootstrap", type=int, default=10000,
                   help="Number of paired bootstrap resamples (default: 10000)")
    p.add_argument("--seed", type=int, default=20261016,
                   help="Seed of the bootstrap resamples (default: 20261016)")
    p.add_argument("--subsample_draws", type=int, default=4000,
                   help="Stratified subsamples used to judge the representativeness of the rig "
                        "subset (default: 4000)")
    p.add_argument("--subsample_seed", type=int, default=7,
                   help="Seed of those subsamples (default: 7)")
    args = p.parse_args(argv)
    # What --tag, --run_suffix and --splits_dir leave unsaid comes from the benchmark.
    for key, value in BENCHMARKS[args.benchmark].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    args.tag = args.tag or None      # --tag "": runs named without a tag
    return args


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
def f1_macro(y, p):
    """Binary macro-F1, vectorised over leading axes. y, p: integer arrays (..., n) of 0/1."""
    out = 0.0
    for c in (0, 1):
        tp = ((p == c) & (y == c)).sum(-1)
        fp = ((p == c) & (y != c)).sum(-1)
        fn = ((p != c) & (y == c)).sum(-1)
        d = 2 * tp + fp + fn
        out = out + np.where(d > 0, 2 * tp / np.maximum(d, 1), 0)
    return out / 2


def paired(y, pa, pb, idx):
    """Condition a -> condition b on the same images: delta F1, paired bootstrap 95% CI,
    discordant counts (b = only a correct, c = only b correct) and exact McNemar p."""
    d = float(f1_macro(y, pb) - f1_macro(y, pa))
    bd = f1_macro(y[idx], pb[idx]) - f1_macro(y[idx], pa[idx])
    ok_a, ok_b = pa == y, pb == y
    b = int((ok_a & ~ok_b).sum())
    c = int((~ok_a & ok_b).sum())
    p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    return {"d_f1": d, "ci_lo": float(np.percentile(bd, 2.5)), "ci_hi": float(np.percentile(bd, 97.5)),
            "only_a": b, "only_b": c, "p": float(p), "agreement_pct": 100.0 * float((pa == pb).mean())}


def holm(ps):
    """Holm step-down adjusted p-values."""
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, (m - k) * ps[i])
        adj[i] = min(1.0, running)
    return adj


def t_halfwidth(v):
    """Mean, sample sd and 95% t half-width of a small set of values."""
    v = np.asarray(v, float)
    sd = float(v.std(ddof=1))
    return float(v.mean()), sd, float(t_dist.ppf(0.975, len(v) - 1) * sd / np.sqrt(len(v)))


# ----------------------------------------------------------------------------
# Input
# ----------------------------------------------------------------------------
class Inputs:
    """Resolves paths under --root and records the SHA-256 of every file read."""

    def __init__(self, root):
        self.root = root
        self.read = {}

    def path(self, rel):
        full = os.path.join(self.root, rel)
        if not os.path.isfile(full):
            sys.exit(f"[ERROR] Missing input file: {rel}")
        with open(full, "rb") as f:
            # Hash with LF line endings: git on Windows (core.autocrlf) checks text files
            # out with CRLF, and the digests must not depend on the platform.
            self.read[rel.replace(os.sep, "/")] = hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()
        return full

    def json(self, rel):
        with open(self.path(rel), encoding="utf-8") as f:
            return json.load(f)

    def rows(self, rel, delimiter):
        with open(self.path(rel), newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f, delimiter=delimiter))


def as_int(rows, key, rel):
    out = []
    for r in rows:
        v = r.get(key)
        if v in (None, ""):
            sys.exit(f"[ERROR] Column '{key}' is empty for image {r.get('image') or r.get('Image')} in {rel}")
        out.append(int(float(v)))
    return np.array(out, dtype=np.int64)


def load_predictions(io, rel, order=None):
    """Semicolon-separated predictions_*.csv -> (image order, true labels, predictions)."""
    rows = io.rows(rel, ";")
    names = [r["Image"] for r in rows]
    if len(set(names)) != len(names):
        sys.exit(f"[ERROR] Duplicate image names in {rel}")
    by = {r["Image"]: r for r in rows}
    if order is not None:
        if set(names) != set(order):
            sys.exit(f"[ERROR] {rel} does not contain the same images as the reference MIL file")
        rows = [by[n] for n in order]
        names = list(order)
    return names, as_int(rows, "TrueLabel", rel), as_int(rows, "PredLabel", rel)


def load_hil(io, rel, order):
    rows = io.rows(rel, ",")
    by = {r["image"]: r for r in rows}
    if set(by) != set(order):
        sys.exit(f"[ERROR] {rel} does not contain the same {len(order)} rig images")
    return [by[n] for n in order]


def run_name(arch, tag, suffix):
    """Name of a run: {architecture}[_{tag}]{suffix}, as the training scripts write it."""
    return f"{arch}_{tag}{suffix}" if tag else f"{arch}{suffix}"


def level_dir(board, level):
    """Folder of a board level relative to the root, with '/' on every operating system."""
    return boards.results_dir(board, level).replace(os.sep, "/")


def required_inputs(archs, dep, name, pil_dir, hil_dir):
    """(level, architecture, path) of every result file the tables cannot be built without."""
    out = []
    for a in archs:
        out += [("MIL", a, f"results/mil/predictions_{name(a)}.csv"),
                ("SIL", a, f"results/sil/predictions_{name(a)}_int8.csv"),
                ("SIL", a, f"results/sil/metrics_{name(a)}_int8.json")]
    for a in dep:
        out += [("PIL", a, f"{pil_dir}/predictions_{name(a)}_int8.csv"),
                ("PIL", a, f"{pil_dir}/metrics_{name(a)}_int8.json"),
                ("HIL", a, f"{hil_dir}/latency_metrics_{name(a)}_int8.csv"),
                ("HIL", a, f"{hil_dir}/metrics_{name(a)}_int8.json"),
                ("HIL", a, f"{hil_dir}/conditions_{name(a)}_int8.json")]
    return out


def stop_if_levels_are_missing(root, required, archs, tag, suffix):
    """Ends the run with the list of what is missing, instead of failing on the first file.

    The tables compare the four levels image by image, so a benchmark whose HIL
    (or any other level) has not been measured cannot be tabulated here."""
    missing = [(lv, a, rel) for lv, a, rel in required if not os.path.isfile(os.path.join(root, rel))]
    if not missing:
        return
    lines = ["[ERROR] These tables need the four levels (MIL, SIL, PIL, HIL) of every architecture, and "
             f"{len(missing)} result file(s) are missing:"]
    for lv in ("MIL", "SIL", "PIL", "HIL"):
        of_level = [(a, rel) for level, a, rel in missing if level == lv]
        if of_level:
            who = sorted({a for a, _ in of_level}, key=[a for a in archs].index)
            lines.append(f"  {lv}: {', '.join(who)}  (e.g. {of_level[0][1]})")
    # The most common cause is a run name that does not match: say which runs there are.
    mil_dir = os.path.join(root, "results", "mil")
    prefix = f"predictions_{archs[0]}_{tag}+" if tag else f"predictions_{archs[0]}+"
    found = sorted(f[len("predictions_"):-len(".csv")] for f in (os.listdir(mil_dir) if os.path.isdir(mil_dir) else [])
                   if f.startswith(prefix) and f.endswith(".csv"))
    wanted = run_name(archs[0], tag, suffix)
    if found and wanted not in found:
        lines.append(f"  No MIL run is named {wanted}; results/mil has: {', '.join(found)}. "
                     "Check --tag and --run_suffix.")
    lines.append("  A benchmark measured up to PIL is summarised by src/build_benchmark_table.py instead.")
    sys.exit("\n".join(lines))


def read_class_names(io, args, manifest_rows, mil_rel):
    """The two class names in label order, and where they were taken from.

    --class_names wins; otherwise the test manifest (columns label and
    class_name), otherwise the names the reference MIL run recorded for each
    label. The names of the result files must agree with the ones chosen: a
    partition of another benchmark would label the tables wrongly."""
    def from_pairs(pairs):
        by_label = {}
        for label, name in pairs:
            if name not in (None, "") and str(label).strip() != "":
                by_label.setdefault(int(float(label)), set()).add(name.strip())
        if sorted(by_label) == [0, 1] and all(len(v) == 1 for v in by_label.values()):
            return [next(iter(by_label[0])), next(iter(by_label[1]))]
        return None

    recorded = from_pairs((r.get("TrueLabel"), r.get("TrueClass")) for r in io.rows(mil_rel, ";"))
    if args.class_names:
        path = os.path.join(args.root, args.class_names)
        if not os.path.isfile(path):
            sys.exit(f"[ERROR] --class_names: {args.class_names} not found")
        with open(path, encoding="utf-8") as f:
            names, source = f.read().split(), args.class_names
        if len(names) != 2:
            sys.exit(f"[ERROR] This script handles binary tasks; {args.class_names} lists {len(names)} classes")
    else:
        names = None
        if manifest_rows and "label" in manifest_rows[0] and "class_name" in manifest_rows[0]:
            names, source = from_pairs((r["label"], r["class_name"]) for r in manifest_rows), "the test manifest"
            if names is None:
                sys.exit("[ERROR] This script handles binary tasks with labels 0 and 1; the test manifest "
                         "does not describe one")
        if names is None:
            names, source = recorded, mil_rel
        if names is None:
            names, source = ["class0", "class1"], "no source (generic names)"
    if recorded is not None and recorded != names:
        sys.exit(f"[ERROR] Class names differ: {source} says {names} and {mil_rel} recorded {recorded}. "
                 "Check --splits_dir (or --class_names): is it the partition of this benchmark?")
    return names, source


# ----------------------------------------------------------------------------
# Formatting
# ----------------------------------------------------------------------------
def num(v, d=4):
    v = float(v)
    if abs(v) < 0.5 * 10 ** -d:
        v = 0.0
    return f"{v:.{d}f}".replace("-", "\u2212")


def pval(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def ci(lo, hi):
    return f"[{num(lo)}, {num(hi)}]"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


TEX_MAP = [("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("_", "\\_"), ("#", "\\#"),
           ("\u0394", "$\\Delta$"), ("\u2192", "$\\rightarrow$"), ("\u2212", "$-$"),
           ("\u00b7", "$\\cdot$"), ("<", "$<$"), ("\u00d7", "$\\times$"), ("\u2014", "---"), ("\u2013", "--")]


def tex_escape(s):
    s = str(s)
    for a, b in TEX_MAP:
        s = s.replace(a, b)
    return s


def tex_table(caption, label, header, rows):
    cols = "l" + "r" * (len(header) - 1)
    body = "\n".join(" & ".join(tex_escape(c) for c in r) + " \\\\" for r in rows)
    return ("\\begin{table}[htbp]\n\\centering\n\\scriptsize\n"
            f"\\caption{{{tex_escape(caption)}}}\n\\label{{{label}}}\n"
            f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n"
            + " & ".join(tex_escape(h) for h in header) + " \\\\\n\\midrule\n"
            + body + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    args = parse_args()
    archs = [a.strip() for a in args.archs.split(",") if a.strip()]
    dep = [a.strip() for a in args.deployable.split(",") if a.strip()]
    if not set(dep) <= set(archs):
        sys.exit("[ERROR] Every --deployable architecture must also be in --archs")
    if not dep:
        sys.exit("[ERROR] --deployable is empty: the tables need at least one architecture with PIL and HIL runs")
    ref = dep[0]
    io = Inputs(args.root)
    tag = lambda a: run_name(a, args.tag, args.run_suffix)   # the full name of the run of an architecture

    # ---- what defines the benchmark: run names, partition, board -------------
    pil_dir, hil_dir = level_dir(args.board, "pil"), level_dir(args.board, "hil")
    board = boards.BOARDS[args.board]
    manifest_given = args.manifest is not None
    manifest = (args.manifest if manifest_given else os.path.join(args.splits_dir, "test_manifest.csv")).replace(os.sep, "/")
    # The vehicle benchmark as it was published (--benchmark vehicle): its tables keep the
    # wording of the published files, and its summary their keys.
    vehicle_benchmark = (not args.tag and args.run_suffix == VEHICLE["run_suffix"]
                         and args.board == boards.DEFAULT_BOARD
                         and manifest == f"{VEHICLE['splits_dir']}/test_manifest.csv")
    if args.output_dir is None:
        args.output_dir = os.path.join(boards.results_dir(args.board, "tables"), *([args.tag.lower()] if args.tag else []))
    stop_if_levels_are_missing(args.root, required_inputs(archs, dep, tag, pil_dir, hil_dir), archs,
                               args.tag, args.run_suffix)

    # ---- full test set: MIL, SIL, PIL ---------------------------------------
    order, y_full, _ = load_predictions(io, f"results/mil/predictions_{tag(archs[0])}.csv")
    pred = {}
    for a in archs:
        _, y_m, pred[("MIL", a)] = load_predictions(io, f"results/mil/predictions_{tag(a)}.csv", order)
        _, y_s, pred[("SIL", a)] = load_predictions(io, f"results/sil/predictions_{tag(a)}_int8.csv", order)
        if not ((y_m == y_full).all() and (y_s == y_full).all()):
            sys.exit(f"[ERROR] Ground truth differs between levels for {a}")
        if a in dep:
            _, y_p, pred[("PIL", a)] = load_predictions(io, f"{pil_dir}/predictions_{tag(a)}_int8.csv", order)
            if not (y_p == y_full).all():
                sys.exit(f"[ERROR] Ground truth differs at PIL for {a}")
    n_full = len(order)
    pos = {name: i for i, name in enumerate(order)}

    # ---- rig subset: HIL and host re-classification of the same frames ------
    rel_ref = f"{hil_dir}/latency_metrics_{tag(ref)}_int8.csv"
    rig = [r["image"] for r in io.rows(rel_ref, ",")]
    if not set(rig) <= set(order):
        sys.exit("[ERROR] Some rig images are not in the test set")
    man_path = os.path.join(args.root, manifest)
    subset_check = None
    man = []
    if os.path.isfile(man_path):
        man = io.rows(manifest, ",")
        key = "filename" if man and "filename" in man[0] else (list(man[0])[0] if man else None)
        if man and {os.path.basename(r[key]) for r in man} != set(order):
            sys.exit(f"[ERROR] {manifest} does not list the {n_full} test images of the result files. "
                     "Check --splits_dir: is it the partition of this benchmark?")
        if man and "hil_subset" in man[0]:
            flagged = {os.path.basename(r[key]) for r in man if str(r["hil_subset"]).strip() in ("1", "1.0", "True")}
            subset_check = flagged == {os.path.basename(n) for n in rig}
            if not subset_check:
                sys.exit("[ERROR] The rig images do not match hil_subset == 1 in the manifest")
    elif manifest_given or args.splits_dir.replace(os.sep, "/") != VEHICLE["splits_dir"]:
        # Only the partition of the vehicle benchmark may come without its manifest, as it always could.
        sys.exit(f"[ERROR] Test manifest not found: {manifest}")
    class_names, class_names_source = read_class_names(io, args, man, f"results/mil/predictions_{tag(archs[0])}.csv")
    rig_pos = np.array([pos[n] for n in rig])
    y_rig = y_full[rig_pos]
    n_rig = len(rig)
    for a in dep:
        rel = f"{hil_dir}/latency_metrics_{tag(a)}_int8.csv"
        rows = load_hil(io, rel, rig)
        if not (as_int(rows, "true_label", rel) == y_rig).all():
            sys.exit(f"[ERROR] Ground truth of the rig subset differs for {a}")
        pred[("HIL", a)] = as_int(rows, "pred_label", rel)
        pred[("MILc", a)] = as_int(rows, "mil_label", rel)   # same camera frame, float on host
        pred[("SILc", a)] = as_int(rows, "sil_label", rel)   # same camera frame, INT8 on host

    def on_rig(level, a):
        return pred[(level, a)] if level in ("HIL", "MILc", "SILc") else pred[(level, a)][rig_pos]

    # ---- bootstrap index matrices (fixed draw order: full set, then rig) ----
    rng = np.random.default_rng(args.seed)
    idx_full = rng.integers(0, n_full, (args.bootstrap, n_full), dtype=np.int32)
    idx_rig = rng.integers(0, n_rig, (args.bootstrap, n_rig), dtype=np.int32)

    tfl = {}
    for a in archs:
        mo = io.json(f"results/sil/metrics_{tag(a)}_int8.json")["model"]
        tfl[a] = (int(mo["bytes"]), mo.get("sha256"))

    # ======================================================================
    # Table 1 - ladder
    # ======================================================================
    t1 = {}
    for a in archs:
        r = {"tflite_bytes": tfl[a][0], "deployable": a in dep}
        for lv in ("MIL", "SIL") + (("PIL",) if a in dep else ()):
            r[f"{lv}_f1"] = float(f1_macro(y_full, pred[(lv, a)]))
            r[f"{lv}_errors"] = int((pred[(lv, a)] != y_full).sum())
        r["quant"] = paired(y_full, pred[("MIL", a)], pred[("SIL", a)], idx_full)
        if a in dep:
            r["exec"] = paired(y_full, pred[("SIL", a)], pred[("PIL", a)], idx_full)
            r["PIL200_f1"] = float(f1_macro(y_rig, on_rig("PIL", a)))
            r["HIL_f1"] = float(f1_macro(y_rig, pred[("HIL", a)]))
            r["HIL_errors"] = int((pred[("HIL", a)] != y_rig).sum())
            r["frontend"] = paired(y_rig, on_rig("PIL", a), pred[("HIL", a)], idx_rig)
        t1[a] = r
    for fam, names in (("quant", archs), ("exec", dep), ("frontend", dep)):
        for a, q in zip(names, holm([t1[a][fam]["p"] for a in names])):
            t1[a][fam]["p_holm"] = float(q)

    # ======================================================================
    # Table 2 - paired decomposition input x engine on the rig subset
    # ======================================================================
    effects = {"quant_dataset": ("MIL", "SIL"), "quant_camera": ("MILc", "SILc"),
               "optical_float": ("MIL", "MILc"), "optical_int8": ("SIL", "SILc"),
               "optical_mcu": ("PIL", "HIL"), "exec_dataset": ("SIL", "PIL"), "exec_camera": ("SILc", "HIL")}
    t2 = {a: {"f1": {lv: float(f1_macro(y_rig, on_rig(lv, a))) for lv in ("MIL", "SIL", "PIL", "MILc", "SILc", "HIL")}}
          for a in dep}
    for a in dep:
        for k, (x1, x2) in effects.items():
            t2[a][k] = paired(y_rig, on_rig(x1, a), on_rig(x2, a), idx_rig)
    for k in effects:
        for a, q in zip(dep, holm([t2[a][k]["p"] for a in dep])):
            t2[a][k]["p_holm"] = float(q)
    for a in dep:   # quantisation x input interaction (difference of differences)
        o = lambda lv: (on_rig(lv, a) == y_rig).astype(int)
        dd = (o("SILc") - o("MILc")) - (o("SIL") - o("MIL"))
        nz = dd[dd != 0]
        bdd = ((f1_macro(y_rig[idx_rig], on_rig("SILc", a)[idx_rig]) - f1_macro(y_rig[idx_rig], on_rig("MILc", a)[idx_rig]))
               - (f1_macro(y_rig[idx_rig], on_rig("SIL", a)[idx_rig]) - f1_macro(y_rig[idx_rig], on_rig("MIL", a)[idx_rig])))
        t2[a]["interaction"] = {"dd": t2[a]["quant_camera"]["d_f1"] - t2[a]["quant_dataset"]["d_f1"],
                                "ci_lo": float(np.percentile(bdd, 2.5)), "ci_hi": float(np.percentile(bdd, 97.5)),
                                "p": float(binomtest(int((nz < 0).sum()), len(nz), 0.5).pvalue) if len(nz) else 1.0}
    for a, q in zip(dep, holm([t2[a]["interaction"]["p"] for a in dep])):
        t2[a]["interaction"]["p_holm"] = float(q)

    # ======================================================================
    # Table 3 - per-phase latency and memory
    # ======================================================================
    t3 = {}
    for a in dep:
        pil_rel, hil_rel = f"{pil_dir}/metrics_{tag(a)}_int8.json", f"{hil_dir}/metrics_{tag(a)}_int8.json"
        pil_doc, hil_doc = io.json(pil_rel), io.json(hil_rel)
        for rel, doc in ((pil_rel, pil_doc), (hil_rel, hil_doc)):
            # A run filed under the folder of one board but measured on the other would mix two boards.
            measured_on = (doc.get("run") or {}).get("board")
            if measured_on not in (None, board["announces"]):
                sys.exit(f"[ERROR] {rel} was measured on {measured_on}, not on the board given with --board "
                         f"({board['announces']})")
        P, H = pil_doc["latency"]["per_field"], hil_doc["latency"]["per_field"]
        fw = io.json(f"{hil_dir}/conditions_{tag(a)}_int8.json")["firmware_model"]
        ms = lambda d, k: d[k]["mean"] / 1000.0
        t3[a] = {"tflite_KiB": tfl[a][0] / 1024.0, "arena_KiB": fw["arena_used_bytes"] / 1024.0,
                 "PIL_pre_ms": ms(P, "US_PRE"), "PIL_inf_ms": ms(P, "US_INF"), "PIL_post_ms": ms(P, "US_POST"),
                 "PIL_total_ms": ms(P, "US_TOTAL"),
                 "PIL_cv_pct": 100 * P["US_TOTAL"]["std"] / P["US_TOTAL"]["mean"],
                 "PIL_cycles_inf": P["CYC_INF"]["mean"],
                 "PIL_cycles_per_us": P["CYC_INF"]["mean"] / P["US_INF"]["mean"],
                 "HIL_capture_ms": ms(H, "US_CAPTURE"), "HIL_inf_ms": ms(H, "US_INF"), "HIL_total_ms": ms(H, "US_TOTAL"),
                 "HIL_cv_pct": 100 * H["US_TOTAL"]["std"] / H["US_TOTAL"]["mean"],
                 "HIL_cycles_inf": H["CYC_INF"]["mean"],
                 "d_cycles_inf_HIL_vs_PIL_pct": 100 * (H["CYC_INF"]["mean"] - P["CYC_INF"]["mean"]) / P["CYC_INF"]["mean"],
                 "inf_share_PIL_pct": 100 * P["US_INF"]["mean"] / P["US_TOTAL"]["mean"],
                 "n_PIL": P["US_TOTAL"].get("n"), "n_HIL": H["US_TOTAL"].get("n")}
    t3_order = sorted(dep, key=lambda a: t3[a]["PIL_total_ms"])

    # ======================================================================
    # Table 4 - HIL repeatability of the reference architecture
    # ======================================================================
    hil_root = os.path.join(args.root, *hil_dir.split("/"))
    rep_stem = lambda d: f"{ref}_{args.tag}_{d}" if args.tag else f"{ref}_{d}"
    rep_dirs, rep_dirs_of_other_runs = [], []
    for d in sorted(os.listdir(hil_root)):
        if d.startswith("rep") and os.path.isdir(os.path.join(hil_root, d)):
            # The rep folders are shared by the benchmarks: one that holds no run of this
            # benchmark's reference architecture is left out; one that holds half of it is an error below.
            has_run = any(os.path.isfile(os.path.join(hil_root, d, name % rep_stem(d)))
                          for name in ("conditions_%s.json", "latency_metrics_%s.csv"))
            (rep_dirs if has_run else rep_dirs_of_other_runs).append(d)
    ref_cond = io.json(f"{hil_dir}/conditions_{tag(ref)}_int8.json")
    pil_rig = on_rig("PIL", ref)
    pil_rig_f1 = float(f1_macro(y_rig, pil_rig))
    runs = [{"run": "original", "seed": ref_cond.get("seed"), "frame_dump": bool(ref_cond.get("frame_dump")),
             "pred": pred[("HIL", ref)]}]
    # "original" is the HIL run of the ladder (the one in the HIL folder itself). In the vehicle
    # benchmark it was measured before the protocol of the repeatability runs was fixed, which is
    # what its role says; in another benchmark it is simply the run the other tables use.
    original_role = ("original (different protocol, excluded from the mean)" if vehicle_benchmark
                     else "run of the ladder (not in the mean of the repeatability runs)" if rep_dirs
                     else "run of the ladder")
    for d in rep_dirs:
        cond = io.json(f"{hil_dir}/{d}/conditions_{rep_stem(d)}.json")
        rel = f"{hil_dir}/{d}/latency_metrics_{rep_stem(d)}.csv"
        runs.append({"run": d, "seed": cond.get("seed"), "frame_dump": bool(cond.get("frame_dump")),
                     "pred": as_int(load_hil(io, rel, rig), "pred_label", rel)})
    for r in runs:
        r["f1"] = float(f1_macro(y_rig, r["pred"]))
        r["errors"] = int((r["pred"] != y_rig).sum())
        r["gap"] = r["f1"] - pil_rig_f1
        r["agreement_with_original_pct"] = 100.0 * float((r["pred"] == runs[0]["pred"]).mean())
        r["role"] = (original_role if r["run"] == "original"
                     else "strict replicate" if r["seed"] == runs[0]["seed"] else "presentation-order control")
    reps = [r for r in runs if r["run"] != "original"]
    rep_summary = None
    if len(reps) >= 2:
        strict = [r["gap"] for r in reps if r["role"] == "strict replicate"]
        gaps = [r["gap"] for r in reps]
        _, sd_all, hw_all = t_halfwidth(gaps)
        gm = float(np.mean(gaps))
        gm_b = np.mean([f1_macro(y_rig[idx_rig], r["pred"][idx_rig]) for r in reps], axis=0) \
            - f1_macro(y_rig[idx_rig], pil_rig[idx_rig])
        var_total = gm_b.var() + sd_all ** 2 / len(reps)
        rep_summary = {
            "n_runs": len(reps), "mean_gap": gm,
            "ci95_combined": [gm - 1.96 * np.sqrt(var_total), gm + 1.96 * np.sqrt(var_total)],
            "ci95_image_sampling_only": [float(np.percentile(gm_b, 2.5)), float(np.percentile(gm_b, 97.5))],
            "image_sampling_share_of_variance": float(gm_b.var() / var_total),
            "repeatability_sd_all_runs": sd_all, "repeatability_t_halfwidth_all_runs": hw_all,
            "repeatability_sd_strict": t_halfwidth(strict)[1] if len(strict) >= 2 else None,
            "n_strict": len(strict),
            "identical_prediction_in_all_runs": int(np.all([r["pred"] == reps[0]["pred"] for r in reps], axis=0).sum()),
        }

    # ======================================================================
    # Table 5 - per-class sensitivity
    # ======================================================================
    t5 = []
    for a in archs:
        for lv, sub in (("MIL", False), ("SIL", False), ("PIL", False), ("PIL", True), ("HIL", True)):
            if (lv, a) not in pred:
                continue
            yy = y_rig if sub else y_full
            pp = on_rig(lv, a) if sub else pred[(lv, a)]
            e1, e2 = int(((yy == 0) & (pp != 0)).sum()), int(((yy == 1) & (pp != 1)).sum())
            n1, n2 = int((yy == 0).sum()), int((yy == 1).sum())
            t5.append({"arch": a, "level": lv + (f" ({n_rig})" if sub and lv != "HIL" else ""), "n": len(yy),
                       "recall_0": 1 - e1 / n1, "recall_1": 1 - e2 / n2, "errors_0": e1, "errors_1": e2,
                       "p": float(fisher_exact([[e1, n1 - e1], [e2, n2 - e2]])[1])})
    for r, q in zip(t5, holm([r["p"] for r in t5])):
        r["p_holm"] = float(q)

    # ======================================================================
    # Table 6 - between-architecture contrasts of the PIL -> HIL gap
    # ======================================================================
    gap_b = {a: f1_macro(y_rig[idx_rig], pred[("HIL", a)][idx_rig]) - f1_macro(y_rig[idx_rig], on_rig("PIL", a)[idx_rig])
             for a in dep}
    t6 = []
    for a, b in itertools.combinations(dep, 2):
        dd = gap_b[a] - gap_b[b]
        p = max(2 * min((dd <= 0).mean(), (dd >= 0).mean()), 1 / args.bootstrap)
        t6.append({"a": a, "b": b, "dd": t1[a]["frontend"]["d_f1"] - t1[b]["frontend"]["d_f1"],
                   "ci_lo": float(np.percentile(dd, 2.5)), "ci_hi": float(np.percentile(dd, 97.5)), "p": float(p)})
    for r, q in zip(t6, holm([r["p"] for r in t6])):
        r["p_holm"] = float(q)

    # ======================================================================
    # Table 7 - PIL attribution check (agreement of PIL(a) with SIL(b))
    # ======================================================================
    t7 = {a: {b: 100.0 * float((pred[("PIL", a)] == pred[("SIL", b)]).mean()) for b in archs} for a in dep}
    attribution_ok = all(max(t7[a], key=t7[a].get) == a for a in dep)

    # ======================================================================
    # Table 8 - representativeness of the rig subset
    # ======================================================================
    rng2 = np.random.default_rng(args.subsample_seed)
    pos0, pos1 = np.where(y_full == 0)[0], np.where(y_full == 1)[0]
    k0, k1 = int((y_rig == 0).sum()), int((y_rig == 1).sum())
    sims = {a: [] for a in dep}
    for _ in range(args.subsample_draws):
        s = np.concatenate([rng2.choice(pos0, k0, replace=False), rng2.choice(pos1, k1, replace=False)])
        for a in dep:
            sims[a].append(f1_macro(y_full[s], pred[("PIL", a)][s]) - f1_macro(y_full, pred[("PIL", a)]))
    t8 = {}
    for a in dep:
        dev = {lv: float(f1_macro(y_rig, on_rig(lv, a)) - f1_macro(y_full, pred[(lv, a)])) for lv in ("MIL", "SIL", "PIL")}
        sv = np.asarray(sims[a])
        sd = float(sv.std())
        t8[a] = {**dev, "sd_expected_PIL": sd, "z_PIL": dev["PIL"] / sd,
                 "p_two_sided": float(2 * min((sv >= dev["PIL"]).mean(), (sv <= dev["PIL"]).mean()))}
    mean_dev = float(np.mean([t8[a]["PIL"] for a in dep]))
    p_mean_dev = float((np.mean([sims[a] for a in dep], axis=0) >= mean_dev).mean())

    # ======================================================================
    # Output
    # ======================================================================
    out_dir = os.path.join(args.root, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    c0, c1 = class_names
    md, tex = [], []
    # What names the benchmark in the title and the LaTeX labels: its tag and, when it is not the
    # Portenta H7, its board (tab:ladder-coco-1 for the COCO benchmark; tab:ladder1 for the
    # vehicle benchmark, whose runs carry no tag).
    marks = ([args.tag] if args.tag else []) + ([board["label"]] if args.board != boards.DEFAULT_BOARD else [])
    label_stem = "ladder" + "".join("-" + m.lower().replace(" ", "") for m in marks) + ("-" if marks else "")

    def emit(k, title, header, rows, note, csv_name, csv_header, csv_rows):
        md.append(f"## Table {k} - {title}\n\n{md_table(header, rows)}\n\n*{note}*\n")
        tex.append(tex_table(f"{title}. {note}", f"tab:{label_stem}{k}", header, rows))
        write_csv(os.path.join(out_dir, f"table{k}_{csv_name}.csv"), csv_header, csv_rows)

    # Table 1
    rows, crows = [], []
    for a in archs:
        r = t1[a]; q = r["quant"]
        row = [a, f"{r['tflite_bytes'] / 1024:,.1f}", num(r["MIL_f1"]), num(r["SIL_f1"]),
               f"{num(q['d_f1'])} {ci(q['ci_lo'], q['ci_hi'])}", pval(q["p_holm"])]
        if a in dep:
            e, h = r["exec"], r["frontend"]
            row += [num(r["PIL_f1"]), f"{e['agreement_pct']:.1f} %", num(r["PIL200_f1"]), num(r["HIL_f1"]),
                    f"{num(h['d_f1'])} {ci(h['ci_lo'], h['ci_hi'])}", pval(h["p_holm"])]
        else:
            row += ["does not fit"] + ["\u2014"] * 5
        rows.append(row)
        g = lambda fam, k: r[fam][k] if fam in r else ""
        crows.append([a, r["tflite_bytes"], r["tflite_bytes"] > FLASH_BYTES, r["MIL_f1"], r["MIL_errors"], r["SIL_f1"], r["SIL_errors"],
                      q["d_f1"], q["ci_lo"], q["ci_hi"], q["only_a"], q["only_b"], q["p"], q["p_holm"],
                      r.get("PIL_f1", ""), r.get("PIL_errors", ""), g("exec", "d_f1"), g("exec", "ci_lo"), g("exec", "ci_hi"),
                      g("exec", "only_a"), g("exec", "only_b"), g("exec", "p"), g("exec", "p_holm"), g("exec", "agreement_pct"),
                      r.get("PIL200_f1", ""), r.get("HIL_f1", ""), r.get("HIL_errors", ""), g("frontend", "d_f1"),
                      g("frontend", "ci_lo"), g("frontend", "ci_hi"), g("frontend", "only_a"), g("frontend", "only_b"),
                      g("frontend", "p"), g("frontend", "p_holm"), g("frontend", "agreement_pct")])
    big = ", ".join(f"{a} {tfl[a][0]:,} B" for a in archs if a not in dep)
    does_not_fit = f"'does not fit': the .tflite alone exceeds the 2 MiB internal flash ({big}). " if big else ""
    emit(1, "Fidelity ladder MIL \u2192 SIL \u2192 PIL \u2192 HIL (macro-F1)",
         ["Architecture", ".tflite (KiB)", "MIL", "SIL", "\u0394 MIL\u2192SIL [95% CI]", "p Holm", "PIL",
          "SIL\u2013PIL agreement", f"PIL ({n_rig})", f"HIL ({n_rig})", "\u0394 PIL\u2192HIL [95% CI]", "p Holm"],
         rows,
         f"MIL, SIL and PIL on the full test set (n = {n_full}); PIL ({n_rig}) and HIL on the rig subset (n = {n_rig}), "
         f"same images. {does_not_fit}"
         "SIL\u2013PIL agreement: share of images with the same prediction in the INT8 simulator and on the MCU. "
         f"95% CI: paired bootstrap ({args.bootstrap:,} resamples); p: exact McNemar, Holm within each column.",
         "ladder",
         ["arch", "tflite_bytes", "exceeds_flash", "MIL_f1", "MIL_errors", "SIL_f1", "SIL_errors",
          "quant_d_f1", "quant_ci_lo", "quant_ci_hi", "quant_only_MIL", "quant_only_SIL", "quant_p", "quant_p_holm",
          "PIL_f1", "PIL_errors", "exec_d_f1", "exec_ci_lo", "exec_ci_hi", "exec_only_SIL", "exec_only_PIL",
          "exec_p", "exec_p_holm", "exec_agreement_pct", "PIL200_f1", "HIL_f1", "HIL_errors",
          "frontend_d_f1", "frontend_ci_lo", "frontend_ci_hi", "frontend_only_PIL", "frontend_only_HIL",
          "frontend_p", "frontend_p_holm", "frontend_agreement_pct"], crows)

    # Table 2
    eff_cell = lambda e: f"{num(e['d_f1'])} ({e['only_a']}/{e['only_b']}; p {pval(e['p_holm'])})"
    rows = [[a] + [num(t2[a]["f1"][lv]) for lv in ("MIL", "SIL", "PIL", "MILc", "SILc", "HIL")]
            + [eff_cell(t2[a][k]) for k in ("quant_dataset", "quant_camera", "optical_mcu", "exec_camera")] for a in dep]
    ch = ["arch"] + [f"f1_{lv}" for lv in ("MIL", "SIL", "PIL", "MILc", "SILc", "HIL")]
    for k in effects:
        ch += [f"{k}_{x}" for x in ("d_f1", "ci_lo", "ci_hi", "only_a", "only_b", "p", "p_holm")]
    ch += ["interaction_dd", "interaction_ci_lo", "interaction_ci_hi", "interaction_p", "interaction_p_holm"]
    crows = [[a] + [t2[a]["f1"][lv] for lv in ("MIL", "SIL", "PIL", "MILc", "SILc", "HIL")]
             + [t2[a][k][x] for k in effects for x in ("d_f1", "ci_lo", "ci_hi", "only_a", "only_b", "p", "p_holm")]
             + [t2[a]["interaction"][x] for x in ("dd", "ci_lo", "ci_hi", "p", "p_holm")] for a in dep]
    emit(2, f"Paired decomposition input \u00d7 engine (n = {n_rig})",
         ["Architecture", "Dataset \u00b7 host float", "Dataset \u00b7 host INT8", "Dataset \u00b7 MCU",
          "Camera \u00b7 host float", "Camera \u00b7 host INT8", "Camera \u00b7 MCU",
          "Quantization \u00b7 dataset", "Quantization \u00b7 camera", "Physical front end \u00b7 MCU", "MCU vs host INT8 \u00b7 camera"],
         rows,
         f"The six F1 columns use the same rig images. 'Camera' = the frame captured by the {board['camera']} from the on-screen "
         "stimulus, re-classified on the host (float and INT8) and on the MCU. In brackets: b/c (images correct only in "
         "the first / only in the second condition) and exact McNemar p, Holm within each effect. The CSV also holds "
         "the optical effects at every engine, the dataset-input execution effect and the quantization \u00d7 input interaction.",
         "decomposition", ch, crows)

    # Table 3
    rows = [[a, f"{t3[a]['tflite_KiB']:,.1f}", f"{t3[a]['arena_KiB']:,.1f}", f"{t3[a]['PIL_pre_ms']:.2f}",
             f"{t3[a]['PIL_inf_ms']:.1f}", f"{t3[a]['PIL_post_ms']:.3f}", f"{t3[a]['PIL_total_ms']:.1f}",
             f"{t3[a]['PIL_cv_pct']:.3f}", f"{t3[a]['HIL_capture_ms']:.2f}", f"{t3[a]['HIL_total_ms']:.1f}",
             num(t3[a]["d_cycles_inf_HIL_vs_PIL_pct"], 2)] for a in t3_order]
    keys3 = list(t3[t3_order[0]])
    emit(3, "Per-phase latency and memory on the Cortex-M7",
         ["Architecture", ".tflite (KiB)", "Arena (KiB)", "PRE (ms)", "INF (ms)", "POST (ms)", "PIL total (ms)",
          "PIL CV (%)", "CAPTURE (ms)", "HIL total (ms)", "\u0394 INF cycles HIL\u2212PIL (%)"],
         rows,
         "Means over the PIL and HIL runs (DWT cycle counter). PRE, INF and POST are PIL phases; CAPTURE exists only "
         "in HIL. Arena: tensor-arena bytes used, reported by the firmware.",
         "latency", ["arch"] + keys3, [[a] + [t3[a][k] for k in keys3] for a in t3_order])

    # Table 4
    rows = [[r["run"], r["seed"], r["role"], num(r["f1"]), r["errors"], num(r["gap"]),
             f"{r['agreement_with_original_pct']:.1f}"] for r in runs]
    note4 = f"Same {n_rig} rig images in every run; gap = HIL F1 \u2212 PIL F1 on those images."
    if rep_summary:
        lo, hi = rep_summary["ci95_combined"]
        note4 += (f" Mean front-end gap over the {rep_summary['n_runs']} protocol runs: {num(rep_summary['mean_gap'])} "
                  f"(95% CI {num(lo)} to {num(hi)}), combining image sampling (paired bootstrap, "
                  f"{100 * rep_summary['image_sampling_share_of_variance']:.0f} % of the variance) and run-to-run "
                  f"repeatability. Bench repeatability on fixed images: sd = {num(rep_summary['repeatability_sd_strict'])} "
                  f"({rep_summary['n_strict']} strict replicates), sd = {num(rep_summary['repeatability_sd_all_runs'])} "
                  f"(all {rep_summary['n_runs']} runs; t half-width {num(rep_summary['repeatability_t_halfwidth_all_runs'])}); "
                  "this measures instrument stability, not the uncertainty of the gap.")
    emit(4, f"HIL repeatability ({ref}, same {n_rig} images)",
         ["Run", "Seed", "Role", "Macro-F1", "Errors", f"Gap vs PIL ({n_rig})", "Agreement with original (%)"],
         rows, note4, "repeatability",
         ["run", "seed", "frame_dump", "role", "f1", "errors", "gap", "agreement_with_original_pct"],
         [[r["run"], r["seed"], r["frame_dump"], r["role"], r["f1"], r["errors"], r["gap"],
           r["agreement_with_original_pct"]] for r in runs])

    # Table 5
    rows = [[r["arch"], r["level"], r["n"], num(r["recall_0"]), num(r["recall_1"]), r["errors_0"], r["errors_1"],
             pval(r["p_holm"])] for r in t5]
    worse = sum(r["recall_1"] < r["recall_0"] for r in t5)
    emit(5, "Per-class sensitivity",
         ["Architecture", "Level", "n", f"Recall {c0}", f"Recall {c1}", f"Errors {c0}", f"Errors {c1}", "p Holm (Fisher)"],
         rows, f"{c1} has the lower recall in {worse} of {len(t5)} rows. Fisher exact test per row, Holm over all rows.",
         "per_class", ["arch", "level", "n", "recall_0", "recall_1", "errors_0", "errors_1", "p_fisher", "p_holm"],
         [[r[k] for k in ("arch", "level", "n", "recall_0", "recall_1", "errors_0", "errors_1", "p", "p_holm")] for r in t5])

    # Table 6
    rows = [[r["a"], r["b"], num(r["dd"]), num(r["ci_lo"]), num(r["ci_hi"]), pval(r["p"]), pval(r["p_holm"])] for r in t6]
    emit(6, "Between-architecture contrasts of the PIL \u2192 HIL gap",
         ["Architecture A", "Architecture B", "\u0394\u0394 gap (A\u2212B)", "95% CI low", "95% CI high", "p bootstrap", "p Holm"],
         rows, "Paired bootstrap with the same image resamples for both architectures; p is the two-sided bootstrap "
               f"p-value (floor 1/B), Holm over the {len(t6)} pairs.",
         "gap_contrasts", ["a", "b", "dd", "ci_lo", "ci_hi", "p_bootstrap", "p_holm"],
         [[r[k] for k in ("a", "b", "dd", "ci_lo", "ci_hi", "p", "p_holm")] for r in t6])

    # Table 7
    rows = [[a] + [f"{t7[a][b]:.1f}" for b in archs] for a in dep]
    emit(7, f"PIL attribution check (% per-image agreement, n = {n_full})", ["PIL / SIL"] + archs, rows,
         "Row: PIL run labelled as that architecture; column: SIL of each architecture. If the diagonal is the row "
         f"maximum, the MCU ran the labelled model. Diagonal is the maximum in every row: {attribution_ok}.",
         "attribution", ["PIL_arch"] + [f"SIL_{b}" for b in archs], [[a] + [t7[a][b] for b in archs] for a in dep])

    # Table 8
    rows = [[a, num(t8[a]["MIL"]), num(t8[a]["SIL"]), num(t8[a]["PIL"]), num(t8[a]["sd_expected_PIL"]),
             num(t8[a]["z_PIL"], 2), num(t8[a]["p_two_sided"], 3)] for a in sorted(dep)]
    emit(8, f"Representativeness of the rig subset (F1 on {n_rig} \u2212 F1 on {n_full})",
         ["Architecture", "MIL", "SIL", "PIL", "Expected sd from sampling (PIL)", "z", "p"], rows,
         f"Expected sd: standard deviation of the same difference over {args.subsample_draws:,} stratified "
         f"{k0}/{k1} subsamples without replacement of the test set (seed {args.subsample_seed}). Mean PIL deviation "
         f"of the {len(dep)} architectures: {num(mean_dev)} (one-sided p = {p_mean_dev:.2f}).",
         "representativeness", ["arch", "dev_MIL", "dev_SIL", "dev_PIL", "sd_expected_PIL", "z_PIL", "p_two_sided"],
         [[a] + [t8[a][k] for k in ("MIL", "SIL", "PIL", "sd_expected_PIL", "z_PIL", "p_two_sided")] for a in sorted(dep)])

    # ---- documents ----------------------------------------------------------
    try:
        commit = subprocess.run(["git", "-C", args.root, "rev-parse", "HEAD"], capture_output=True,
                                text=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        commit = None
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    which = f" ({', '.join(marks)})" if marks else ""
    head = (f"# PHLAME - fidelity-ladder tables{which}\n\nGenerated by `src/build_ladder_tables.py` on {stamp}"
            f"{f' (commit {commit[:10]})' if commit else ''}. Macro-F1 throughout; n = {n_full} test images "
            f"and {n_rig} rig images ({k0}/{k1} per class). Paired bootstrap: "
            f"{args.bootstrap:,} resamples, seed {args.seed}.\n")
    if marks:
        head += (f"\nRuns named `{tag('<architecture>')}`; partition `{manifest}`; classes `{c0}` (0) and "
                 f"`{c1}` (1); PIL and HIL measured on the {board['label']} (`{pil_dir}`, `{hil_dir}`).\n")
    with open(os.path.join(out_dir, "ladder_tables.md"), "w", encoding="utf-8") as f:
        f.write(head + "\n" + "\n".join(md))
    with open(os.path.join(out_dir, "ladder_tables.tex"), "w", encoding="utf-8") as f:
        f.write("% Generated by src/build_ladder_tables.py - requires \\usepackage{booktabs}\n\n" + "\n".join(tex))
    summary = {
        "generated_utc": stamp, "git_commit": commit,
        "parameters": {"bootstrap": args.bootstrap, "seed": args.seed, "subsample_draws": args.subsample_draws,
                       "subsample_seed": args.subsample_seed, "archs": archs, "deployable": dep, "reference": ref},
        "n_test": n_full, "n_rig": n_rig, "rig_matches_manifest": subset_check,
        "frontend_gap_reference": rep_summary,
        "frontend_gap_per_arch": {a: {k: t1[a]["frontend"][k] for k in ("d_f1", "ci_lo", "ci_hi", "p_holm")} for a in dep},
        "quantization_per_arch": {a: {k: t1[a]["quant"][k] for k in ("d_f1", "ci_lo", "ci_hi", "p_holm")} for a in archs},
        "pil_attribution_diagonal_is_max": attribution_ok,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                        "platform": platform.platform()},
        "inputs_sha256": dict(sorted(io.read.items())),
    }
    # The options that define the benchmark are recorded, except for the vehicle benchmark,
    # whose summary keeps the keys of the published file.
    if not vehicle_benchmark or args.class_names:
        summary["parameters"]["benchmark"] = {
            "tag": args.tag, "run_suffix": args.run_suffix, "board": board["announces"], "manifest": manifest,
            "class_names": [c0, c1], "class_names_from": class_names_source,
            "results_dirs": {"MIL": "results/mil", "SIL": "results/sil", "PIL": pil_dir, "HIL": hil_dir},
            "repeatability_runs": rep_dirs,
        }
    with open(os.path.join(out_dir, "ladder_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)

    print(f"[INFO] Benchmark: runs {tag('<architecture>')} | classes {c0}, {c1} (from {class_names_source}) | "
          f"PIL and HIL on the {board['label']}")
    if rep_dirs_of_other_runs:
        print(f"[INFO] Repeatability folders without a run named {rep_stem('<rep>')}, left out: "
              f"{', '.join(rep_dirs_of_other_runs)}")
    print(f"[INFO] Test set: {n_full} images | rig subset: {n_rig} ({k0}/{k1}) | "
          f"architectures: {len(archs)} ({len(dep)} deployable)")
    if rep_summary:
        lo, hi = rep_summary["ci95_combined"]
        print(f"[INFO] Front-end gap ({ref}, {rep_summary['n_runs']} runs): {rep_summary['mean_gap']:+.4f} "
              f"[{lo:+.4f}, {hi:+.4f}] (95% CI)")
    print(f"[INFO] PIL attribution check passed: {attribution_ok}")
    print(f"[INFO] Wrote 8 tables to {out_dir} (ladder_tables.md, ladder_tables.tex, table*.csv, ladder_summary.json)")


if __name__ == "__main__":
    main()
