#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_ladder_tables.py - Reproduce the statistical tables of the PHLAME fidelity
ladder (MIL -> SIL -> PIL -> HIL) from the per-image result files in results/.

Run from the repository root:

    python src/build_ladder_tables.py
    python src/build_ladder_tables.py --output_dir results/tables --bootstrap 10000

Inputs (all tracked in git; <tag> = <Architecture>+32+20+0.0001+160+120):
  results/mil/predictions_<tag>.csv           MIL: float model on the host        (n = 940)
  results/sil/predictions_<tag>_int8.csv      SIL: INT8 model on the host         (n = 940)
  results/sil/metrics_<tag>_int8.json         exact .tflite size and SHA-256
  results/pil/predictions_<tag>_int8.csv      PIL: INT8 on the MCU, images sent over serial (n = 940)
  results/pil/metrics_<tag>_int8.json         PIL per-phase latency (DWT cycle counter)
  results/hil/hil_latencies_<tag>_int8.csv    HIL: INT8 on the MCU, camera frames (n = 200), plus the
                                              host float/INT8 re-classification of the same frame
  results/hil/metrics_<tag>_int8.json         HIL per-phase latency
  results/hil/hil_conditions_<tag>_int8.json  firmware fingerprint and tensor-arena usage
  results/hil/rep*/                           HIL repeatability runs of the reference architecture
  data/splits/test_manifest.csv               hil_subset flag (the 200 rig images)

Outputs (in --output_dir, default results/tables/):
  ladder_tables.md          the eight tables, formatted
  ladder_tables.tex         the same tables in LaTeX (booktabs)
  table<k>_<name>.csv       unrounded values of every table
  ladder_summary.json       headline numbers and provenance (input SHA-256, seeds,
                            library versions, git commit)

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

Note: the metrics_*.json schema uses Spanish keys (e.g. 'modelo', 'latencia',
'por_campo', 'media', 'desv_std'); they are read here as-is. See "Results Format"
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

DEFAULT_ARCHS = "MobileNetV2,MobileNetV3Large,ResNet18,MobileNet,MobileNetV3Small,SqueezeNet,ResNet8"
DEFAULT_DEPLOYABLE = "MobileNetV2,MobileNet,MobileNetV3Small,SqueezeNet,ResNet8"
FLASH_BYTES = 2 * 1024 * 1024  # STM32H747XI internal flash: 2 MiB


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Rebuild the fidelity-ladder tables (paired bootstrap CIs, exact McNemar, "
                    "Holm) from the per-image result files in results/.")
    p.add_argument("--root", default=".",
                   help="Repository root; all other paths are relative to it (default: .)")
    p.add_argument("--output_dir", default=os.path.join("results", "tables"),
                   help="Output folder (default: results/tables)")
    p.add_argument("--archs", default=DEFAULT_ARCHS,
                   help="Comma-separated architectures evaluated at MIL and SIL. The first one "
                        "defines the image order of the test set.")
    p.add_argument("--deployable", default=DEFAULT_DEPLOYABLE,
                   help="Comma-separated architectures that fit on the MCU (have PIL and HIL runs). "
                        "The first one is the reference for the rig subset and the repeatability runs.")
    p.add_argument("--tag_suffix", default="+32+20+0.0001+160+120",
                   help="Model-name suffix after the architecture (batch+epochs+lr+W+H)")
    p.add_argument("--manifest", default=os.path.join("data", "splits", "test_manifest.csv"),
                   help="Test manifest with the hil_subset column (checked if present)")
    p.add_argument("--class_names", default=os.path.join("models", "class_names.txt"),
                   help="Class names, one per line, in label order")
    p.add_argument("--bootstrap", type=int, default=10000,
                   help="Number of paired bootstrap resamples (default: 10000)")
    p.add_argument("--seed", type=int, default=20261016,
                   help="Seed of the bootstrap resamples (default: 20261016)")
    p.add_argument("--subsample_draws", type=int, default=4000,
                   help="Stratified subsamples used to judge the representativeness of the rig "
                        "subset (default: 4000)")
    p.add_argument("--subsample_seed", type=int, default=7,
                   help="Seed of those subsamples (default: 7)")
    return p.parse_args()


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
            self.read[rel.replace(os.sep, "/")] = hashlib.sha256(f.read()).hexdigest()
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
    ref = dep[0]
    io = Inputs(args.root)
    tag = lambda a: f"{a}{args.tag_suffix}"

    # ---- class names --------------------------------------------------------
    cn_path = os.path.join(args.root, args.class_names)
    class_names = (open(cn_path, encoding="utf-8").read().split()
                   if os.path.isfile(cn_path) else ["class0", "class1"])
    if len(class_names) != 2:
        sys.exit(f"[ERROR] This script handles binary tasks; {args.class_names} lists {len(class_names)} classes")

    # ---- full test set: MIL, SIL, PIL ---------------------------------------
    order, y_full, _ = load_predictions(io, f"results/mil/predictions_{tag(archs[0])}.csv")
    pred = {}
    for a in archs:
        _, y_m, pred[("MIL", a)] = load_predictions(io, f"results/mil/predictions_{tag(a)}.csv", order)
        _, y_s, pred[("SIL", a)] = load_predictions(io, f"results/sil/predictions_{tag(a)}_int8.csv", order)
        if not ((y_m == y_full).all() and (y_s == y_full).all()):
            sys.exit(f"[ERROR] Ground truth differs between levels for {a}")
        if a in dep:
            _, y_p, pred[("PIL", a)] = load_predictions(io, f"results/pil/predictions_{tag(a)}_int8.csv", order)
            if not (y_p == y_full).all():
                sys.exit(f"[ERROR] Ground truth differs at PIL for {a}")
    n_full = len(order)
    pos = {name: i for i, name in enumerate(order)}

    # ---- rig subset: HIL and host re-classification of the same frames ------
    rel_ref = f"results/hil/hil_latencies_{tag(ref)}_int8.csv"
    rig = [r["image"] for r in io.rows(rel_ref, ",")]
    if not set(rig) <= set(order):
        sys.exit("[ERROR] Some rig images are not in the test set")
    man_path = os.path.join(args.root, args.manifest)
    subset_check = None
    if os.path.isfile(man_path):
        man = io.rows(args.manifest, ",")
        if man and "hil_subset" in man[0]:
            key = "filename" if "filename" in man[0] else list(man[0])[0]
            flagged = {os.path.basename(r[key]) for r in man if str(r["hil_subset"]).strip() in ("1", "1.0", "True")}
            subset_check = flagged == {os.path.basename(n) for n in rig}
            if not subset_check:
                sys.exit("[ERROR] The rig images do not match hil_subset == 1 in the manifest")
    rig_pos = np.array([pos[n] for n in rig])
    y_rig = y_full[rig_pos]
    n_rig = len(rig)
    for a in dep:
        rel = f"results/hil/hil_latencies_{tag(a)}_int8.csv"
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
        mo = io.json(f"results/sil/metrics_{tag(a)}_int8.json")["modelo"]
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
        P = io.json(f"results/pil/metrics_{tag(a)}_int8.json")["latencia"]["por_campo"]
        H = io.json(f"results/hil/metrics_{tag(a)}_int8.json")["latencia"]["por_campo"]
        fw = io.json(f"results/hil/hil_conditions_{tag(a)}_int8.json")["firmware_model"]
        ms = lambda d, k: d[k]["media"] / 1000.0
        t3[a] = {"tflite_kB": tfl[a][0] / 1000.0, "arena_kB": fw["arena_used_bytes"] / 1000.0,
                 "PIL_pre_ms": ms(P, "US_PRE"), "PIL_inf_ms": ms(P, "US_INF"), "PIL_post_ms": ms(P, "US_POST"),
                 "PIL_total_ms": ms(P, "US_TOTAL"),
                 "PIL_cv_pct": 100 * P["US_TOTAL"]["desv_std"] / P["US_TOTAL"]["media"],
                 "PIL_cycles_inf": P["CYC_INF"]["media"],
                 "PIL_cycles_per_us": P["CYC_INF"]["media"] / P["US_INF"]["media"],
                 "HIL_capture_ms": ms(H, "US_CAPTURE"), "HIL_inf_ms": ms(H, "US_INF"), "HIL_total_ms": ms(H, "US_TOTAL"),
                 "HIL_cv_pct": 100 * H["US_TOTAL"]["desv_std"] / H["US_TOTAL"]["media"],
                 "HIL_cycles_inf": H["CYC_INF"]["media"],
                 "d_cycles_inf_HIL_vs_PIL_pct": 100 * (H["CYC_INF"]["media"] - P["CYC_INF"]["media"]) / P["CYC_INF"]["media"],
                 "inf_share_PIL_pct": 100 * P["US_INF"]["media"] / P["US_TOTAL"]["media"],
                 "n_PIL": P["US_TOTAL"].get("n"), "n_HIL": H["US_TOTAL"].get("n")}
    t3_order = sorted(dep, key=lambda a: t3[a]["PIL_total_ms"])

    # ======================================================================
    # Table 4 - HIL repeatability of the reference architecture
    # ======================================================================
    rep_dirs = sorted(d for d in os.listdir(os.path.join(args.root, "results", "hil"))
                      if d.startswith("rep") and os.path.isdir(os.path.join(args.root, "results", "hil", d)))
    ref_cond = io.json(f"results/hil/hil_conditions_{tag(ref)}_int8.json")
    pil_rig = on_rig("PIL", ref)
    pil_rig_f1 = float(f1_macro(y_rig, pil_rig))
    runs = [{"run": "original", "seed": ref_cond.get("seed"), "frame_dump": bool(ref_cond.get("frame_dump")),
             "pred": pred[("HIL", ref)]}]
    for d in rep_dirs:
        cond = io.json(f"results/hil/{d}/hil_conditions_{ref}_{d}.json")
        rel = f"results/hil/{d}/hil_latencies_{ref}_{d}.csv"
        runs.append({"run": d, "seed": cond.get("seed"), "frame_dump": bool(cond.get("frame_dump")),
                     "pred": as_int(load_hil(io, rel, rig), "pred_label", rel)})
    for r in runs:
        r["f1"] = float(f1_macro(y_rig, r["pred"]))
        r["errors"] = int((r["pred"] != y_rig).sum())
        r["gap"] = r["f1"] - pil_rig_f1
        r["agreement_with_original_pct"] = 100.0 * float((r["pred"] == runs[0]["pred"]).mean())
        r["role"] = ("original (different protocol, excluded from the mean)" if r["run"] == "original"
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
            t5.append({"arch": a, "level": lv + (" (200)" if sub and lv != "HIL" else ""), "n": len(yy),
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

    def emit(k, title, header, rows, note, csv_name, csv_header, csv_rows):
        md.append(f"## Table {k} - {title}\n\n{md_table(header, rows)}\n\n*{note}*\n")
        tex.append(tex_table(f"{title}. {note}", f"tab:ladder{k}", header, rows))
        write_csv(os.path.join(out_dir, f"table{k}_{csv_name}.csv"), csv_header, csv_rows)

    # Table 1
    rows, crows = [], []
    for a in archs:
        r = t1[a]; q = r["quant"]
        row = [a, f"{r['tflite_bytes'] / 1000:,.0f}", num(r["MIL_f1"]), num(r["SIL_f1"]),
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
    emit(1, "Fidelity ladder MIL \u2192 SIL \u2192 PIL \u2192 HIL (macro-F1)",
         ["Architecture", ".tflite (kB)", "MIL", "SIL", "\u0394 MIL\u2192SIL [95% CI]", "p Holm", "PIL",
          "SIL\u2013PIL agreement", "PIL (200)", "HIL (200)", "\u0394 PIL\u2192HIL [95% CI]", "p Holm"],
         rows,
         f"MIL, SIL and PIL on the full test set (n = {n_full}); PIL (200) and HIL on the rig subset (n = {n_rig}), "
         f"same images. 'does not fit': the .tflite alone exceeds the 2 MiB internal flash ({big}). "
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
         "The six F1 columns use the same rig images. 'Camera' = the frame captured by the HM01B0 from the on-screen "
         "stimulus, re-classified on the host (float and INT8) and on the MCU. In brackets: b/c (images correct only in "
         "the first / only in the second condition) and exact McNemar p, Holm within each effect. The CSV also holds "
         "the optical effects at every engine, the dataset-input execution effect and the quantization \u00d7 input interaction.",
         "decomposition", ch, crows)

    # Table 3
    rows = [[a, f"{t3[a]['tflite_kB']:.0f}", f"{t3[a]['arena_kB']:.0f}", f"{t3[a]['PIL_pre_ms']:.2f}",
             f"{t3[a]['PIL_inf_ms']:.1f}", f"{t3[a]['PIL_post_ms']:.3f}", f"{t3[a]['PIL_total_ms']:.1f}",
             f"{t3[a]['PIL_cv_pct']:.3f}", f"{t3[a]['HIL_capture_ms']:.2f}", f"{t3[a]['HIL_total_ms']:.1f}",
             num(t3[a]["d_cycles_inf_HIL_vs_PIL_pct"], 2)] for a in t3_order]
    keys3 = list(t3[t3_order[0]])
    emit(3, "Per-phase latency and memory on the Cortex-M7",
         ["Architecture", ".tflite (kB)", "Arena (kB)", "PRE (ms)", "INF (ms)", "POST (ms)", "PIL total (ms)",
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
    head = (f"# PHLAME - fidelity-ladder tables\n\nGenerated by `src/build_ladder_tables.py` on {stamp}"
            f"{f' (commit {commit[:10]})' if commit else ''}. Macro-F1 throughout; n = {n_full} test images "
            f"and {n_rig} rig images ({k0}/{k1} per class). Paired bootstrap: "
            f"{args.bootstrap:,} resamples, seed {args.seed}.\n")
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
    with open(os.path.join(out_dir, "ladder_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)

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
