#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
independence_report.py - Do the results depend on the test images that also
appear in train/val?

find_near_duplicates.py lists, for every test image, its closest train/val
image and a verdict. This script takes that list and recomputes accuracy and
macro-F1 of every per-image result file of the partition without the suspect
images, next to the published figure, with a paired bootstrap interval for
the difference.

Typical usage
-------------
    python src/find_near_duplicates.py --splits_dir data/splits
    python src/independence_report.py  --splits_dir data/splits

Two ways of cleaning the test are reported for every result file:

    confirmed   without the test images that have a near-duplicate in
                train/val. If <splits_dir>/near_duplicates_review.csv exists
                (pairs a person looked at, columns test_file, match_file,
                checked, note; checked is same_photo, same_scene or
                different), a pair checked as different is kept and a pair
                checked as the same is removed whatever the detector said.
    strict      without every test image the detector did not call distinct
                (near_duplicate and review). An upper bound: most of the
                review band are different photos.

Result files are found under --results_dir: <level>/predictions_*.csv (MIL,
SIL, PIL) and hil/latency_metrics_*.csv (HIL). A file is used when all its
images belong to the test split of --splits_dir, so the results of other
partitions kept in the same folders are left out without naming them.

Output: one CSV (default results/tables/independence_<splits folder>.csv)
with one row per result file and cleaning. delta = cleaned - full; its 95 %
interval comes from resampling the test images (--bootstrap times, seeded),
computing both figures on each resample.

Exit codes: 0 finished; 1 bad input, missing package or existing output
(--overwrite replaces it).
"""
import csv
import glob
import os
import sys

import artifacts
import cli

try:
    import numpy as np
except ImportError:      # reported by main(); importing the module stays possible
    np = None

SAME = ("same_photo", "same_scene")
CHECKED = SAME + ("different",)


def read_csv(path, sep=","):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter=sep))


def exclusions(table, review):
    """Test files removed by each cleaning: {'confirmed': set, 'strict': set}."""
    same = {r["test_file"] for r in review if r["checked"] in SAME}
    cleared = {(r["test_file"], r["match_file"]) for r in review if r["checked"] == "different"}
    confirmed = {r["test_file"] for r in table
                 if r["verdict"] == "near_duplicate" and (r["test_file"], r["match_file"]) not in cleared}
    confirmed |= same
    strict = {r["test_file"] for r in table if r["verdict"] != "distinct"} | confirmed
    return {"confirmed": confirmed, "strict": strict}


def read_predictions(path):
    """[(image, true label, predicted label)] of the rows that have a prediction."""
    name = os.path.basename(path)
    if name.startswith("predictions_"):
        rows, image, true, pred = read_csv(path, sep=";"), "Image", "TrueLabel", "PredLabel"
    else:
        rows, image, true, pred = read_csv(path), "image", "true_label", "pred_label"
    out = []
    for r in rows:
        if r.get(image) and r.get(true) not in (None, "") and r.get(pred) not in (None, ""):
            out.append((r[image], int(r[true]), int(r[pred])))
    return out


def scores(confusion):
    """Accuracy and macro-F1 from confusion matrices of shape (..., K, K), rows = true label.

    Macro-F1 averages over the labels present in the truth or in the
    predictions, like scikit-learn does in the benches."""
    confusion = np.asarray(confusion, dtype=float)
    tp = np.diagonal(confusion, axis1=-2, axis2=-1)
    truth = confusion.sum(axis=-1)
    predicted = confusion.sum(axis=-2)
    total = truth.sum(axis=-1)
    denom = truth + predicted
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(denom), where=denom > 0)
    present = denom > 0
    macro = f1.sum(axis=-1) / np.maximum(present.sum(axis=-1), 1)
    accuracy = np.divide(tp.sum(axis=-1), total, out=np.zeros_like(total), where=total > 0)
    return accuracy, macro


def compare(y_true, y_pred, keep, n_boot, rng):
    """Figures on all the images and on the kept ones, with the bootstrap interval of the differences."""
    labels = sorted(set(y_true) | set(y_pred))
    k = len(labels)
    index = {lab: i for i, lab in enumerate(labels)}
    code = np.array([index[t] * k + index[p] for t, p in zip(y_true, y_pred)])
    keep = np.asarray(keep, bool)
    code_kept = np.where(keep, code, k * k)                 # removed images fall in an extra bin

    def confusions(sample):
        full = np.bincount(code[sample], minlength=k * k).reshape(k, k)
        kept = np.bincount(code_kept[sample], minlength=k * k + 1)[:k * k].reshape(k, k)
        return full, kept

    everything = np.arange(len(code))
    full, kept = confusions(everything)
    (acc_full, f1_full), (acc_kept, f1_kept) = scores(full), scores(kept)
    boot = np.empty((n_boot, 2, k, k))
    for b in range(n_boot):
        boot[b] = confusions(rng.integers(0, len(code), len(code)))
    acc_b, f1_b = scores(boot)                               # shape (n_boot, 2)
    d_acc = np.percentile(acc_b[:, 1] - acc_b[:, 0], [2.5, 97.5])
    d_f1 = np.percentile(f1_b[:, 1] - f1_b[:, 0], [2.5, 97.5])
    return {"n_full": int(len(code)), "accuracy_full": float(acc_full), "f1_macro_full": float(f1_full),
            "n_removed": int((~keep).sum()), "n_kept": int(keep.sum()),
            "accuracy_kept": float(acc_kept), "f1_macro_kept": float(f1_kept),
            "delta_accuracy": float(acc_kept - acc_full), "delta_accuracy_ci95_low": float(d_acc[0]),
            "delta_accuracy_ci95_high": float(d_acc[1]),
            "delta_f1_macro": float(f1_kept - f1_full), "delta_f1_macro_ci95_low": float(d_f1[0]),
            "delta_f1_macro_ci95_high": float(d_f1[1])}


def result_files(results_dir):
    files = sorted(glob.glob(os.path.join(results_dir, "*", "predictions_*.csv")))
    files += sorted(glob.glob(os.path.join(results_dir, "hil", "latency_metrics_*.csv")))
    return files


def parse_args(argv=None):
    p = cli.ArgumentParser(description="Recompute the results without the test images that also appear in train/val.")
    p.add_argument("--splits_dir", default=os.path.join("data", "splits"),
                   help="Folder with near_duplicates.csv, written by find_near_duplicates.py (default: data/splits)")
    p.add_argument("--results_dir", default="results", help="Folder with the result files (default: results)")
    p.add_argument("--output", default=None,
                   help="CSV to write (default: results/tables/independence_<splits folder>.csv)")
    p.add_argument("--bootstrap", type=int, default=10000, help="Resamples of the test images (default: 10000)")
    p.add_argument("--seed", type=int, default=20261114, help="Seed of the resamples (default: 20261114)")
    p.add_argument("--overwrite", action="store_true", help="Replace the output file if it already exists")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if np is None:
        print("ERROR: numpy is needed. Run: pip install -r requirements.txt")
        sys.exit(1)
    table_path = os.path.join(args.splits_dir, "near_duplicates.csv")
    if not os.path.isfile(table_path):
        print("ERROR: %s not found. Run find_near_duplicates.py --splits_dir %s first." % (table_path, args.splits_dir))
        sys.exit(1)
    output = args.output or os.path.join("results", "tables", "independence_%s.csv"
                                         % os.path.basename(os.path.normpath(args.splits_dir)))
    artifacts.refuse_to_overwrite([output], args.overwrite, hint="write somewhere else with --output")

    table = read_csv(table_path)
    review_path = os.path.join(args.splits_dir, "near_duplicates_review.csv")
    review = read_csv(review_path) if os.path.isfile(review_path) else []
    unknown = sorted({r["checked"] for r in review} - set(CHECKED))
    if unknown:
        print("ERROR: %s has values of 'checked' other than %s: %s" % (review_path, ", ".join(CHECKED), unknown))
        sys.exit(1)
    removed = exclusions(table, review)
    test_files = {r["test_file"] for r in table}
    print("[INFO] Test images: %d | removed as confirmed near-duplicates: %d | removed by the strict cleaning: %d%s"
          % (len(test_files), len(removed["confirmed"]), len(removed["strict"]),
             " | pairs checked by a person: %d" % len(review) if review else ""))

    rng = np.random.default_rng(args.seed)
    rows, used, skipped = [], 0, 0
    for path in result_files(args.results_dir):
        try:
            preds = read_predictions(path)
        except (KeyError, ValueError):
            skipped += 1
            continue
        if not preds or any(image not in test_files for image, _, _ in preds):
            skipped += 1                                     # empty, or the results of another partition
            continue
        level = os.path.basename(os.path.dirname(path)).upper()
        stem = os.path.splitext(os.path.basename(path))[0]
        model = stem.split("_", 2)[2] if stem.startswith("latency_metrics_") else stem.split("_", 1)[1]
        y_true = [t for _, t, _ in preds]
        y_pred = [p for _, _, p in preds]
        used += 1
        for cleaning in ("confirmed", "strict"):
            keep = [image not in removed[cleaning] for image, _, _ in preds]
            if sum(keep) == 0:
                continue
            figures = compare(y_true, y_pred, keep, args.bootstrap, rng)
            rows.append({"level": level, "model": model, "cleaning": cleaning,
                         **{key: (round(v, 4) if isinstance(v, float) else v) for key, v in figures.items()}})
    if not rows:
        print("ERROR: no result file under %s has all its images in the test split of %s."
              % (args.results_dir, args.splits_dir))
        sys.exit(1)

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print("%-4s %-46s %-9s %5s %8s %8s %9s  %s" % ("lvl", "model", "cleaning", "n-", "F1 full", "F1 kept", "delta", "95 % interval"))
    for r in rows:
        print("%-4s %-46s %-9s %5d %8.4f %8.4f %+9.4f  [%+.4f, %+.4f]"
              % (r["level"], r["model"], r["cleaning"], r["n_removed"], r["f1_macro_full"], r["f1_macro_kept"],
                 r["delta_f1_macro"], r["delta_f1_macro_ci95_low"], r["delta_f1_macro_ci95_high"]))
    print("[INFO] %d result files used, %d left out (empty or from another partition). Wrote %s"
          % (used, skipped, output))


if __name__ == "__main__":
    main()
