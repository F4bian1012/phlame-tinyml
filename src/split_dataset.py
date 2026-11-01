#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
split_dataset.py - Reproducible dataset splitting for PHLAME.

Final step of data preparation, BEFORE any model training. It takes the
already-processed and resized directory (output of reshape_images.py) and
produces three disjoint splits on disk plus the CSV manifests guaranteeing
that all four levels of the fidelity ladder (MIL -> SIL -> PIL -> HIL)
evaluate EXACTLY the same held-out images.

Why this script exists
----------------------
Without an explicit on-disk split, train_*.py uses `validation_split` over the
full directory while the evaluators (test_model.py, test_tflite_model.py) read
THAT SAME full directory -> reported accuracy includes training images (data
leakage). Moreover, each benchmark selected images differently (manual folder
in PIL, random sampling in HIL), so gaps across ladder levels were not
attributable. This script solves both issues.

Input
-----
    <input_dir>/
        Class1/*.jpg
        Class2/*.jpg

Output
------
    <output_dir>/
        train/Class1/... , train/Class2/...
        val/Class1/...   , val/Class2/...
        test/Class1/...  , test/Class2/...
        train_manifest.csv
        val_manifest.csv
        test_manifest.csv        <- includes the hil_subset (1/0) column
        split_report.json        <- counts, class balance, integrity checks

Typical usage
-------------
    python src/split_dataset.py \
        --input_dir data/processed/160x120 \
        --output_dir data/splits \
        --train 0.70 --val 0.15 --test 0.15 \
        --seed 42 --hil-subset 200

How the four levels consume the splits
--------------------------------------
    train_*.py              --> data/splits/train      (val: data/splits/val)
    test_model.py           --> data/splits/test       (MIL, float, PC)
    test_tflite_model.py    --> data/splits/test       (SIL, INT8, PC)
    pil_benchmark.py        --> data/splits/test       (PIL, INT8, MCU over serial)
    hil_camera_benchmark.py --> test_manifest.csv      (HIL, rows with hil_subset == 1)

Reproducibility notes
---------------------
- The file list is sorted BEFORE shuffling because os.listdir order depends on
  the filesystem. With the same seed, the split is identical on any machine.
- Label order is read from models/class_names.txt, matching the argmax index
  returned by the firmware. It is inferred from the filesystem only if that
  file does not exist.
- Splitting is stratified: it is applied per class so each split preserves the
  natural class ratio of the input dataset (imbalance is not corrected here;
  it is documented in split_report.json).
"""

import argparse
import csv
import json
import os
import random
import shutil
import sys
from datetime import datetime, timezone

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")
SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------- #
# Class and image discovery
# --------------------------------------------------------------------------- #
def load_class_names(class_names_path, input_dir):
    """Label order: class_names.txt if it exists, otherwise sorted subfolders."""
    if class_names_path and os.path.exists(class_names_path):
        with open(class_names_path, encoding="utf-8") as fh:
            names = [ln.strip() for ln in fh if ln.strip()]
        if names:
            return names, class_names_path.replace(os.sep, "/")

    names = sorted(
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d))
    )
    missing = os.path.basename(class_names_path) if class_names_path else "class_names.txt"
    return names, "subfolders of %s (%s not found)" % (input_dir, missing)


def collect_images(class_dir):
    """All images in a class folder, collected recursively in deterministic order."""
    found = []
    for root, _dirs, files in os.walk(class_dir):
        for fn in files:
            if fn.lower().endswith(IMG_EXT):
                found.append(os.path.join(root, fn))
    return sorted(found)


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def split_one_class(files, p_train, p_val, rng):
    """Stratified split of a single class. Returns (train, val, test)."""
    items = list(files)
    rng.shuffle(items)

    n = len(items)
    n_train = int(round(n * p_train))
    n_val = int(round(n * p_val))
    # test receives the remainder so that the three splits sum to n exactly
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)

    return (
        items[:n_train],
        items[n_train:n_train + n_val],
        items[n_train + n_val:],
    )


def pick_hil_subset(test_by_class, n_target, rng):
    """Fixed, stratified subset of the test split for the physical HIL rig."""
    total_test = sum(len(v) for v in test_by_class.values())
    if n_target <= 0 or total_test == 0:
        return {c: set() for c in test_by_class}
    if n_target >= total_test:
        return {c: set(v) for c, v in test_by_class.items()}

    chosen = {}
    assigned = 0
    classes = list(test_by_class.keys())
    for i, cls in enumerate(classes):
        pool = sorted(test_by_class[cls])
        if i == len(classes) - 1:
            k = min(n_target - assigned, len(pool))          # remainder goes to the last class
        else:
            k = int(round(n_target * len(pool) / total_test))
            k = max(0, min(k, len(pool)))
        chosen[cls] = set(rng.sample(pool, k)) if k else set()
        assigned += k
    return chosen


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
def safe_dst(dst_dir, filename, used):
    """Collision-free destination path (nested subfolders may repeat filenames)."""
    base, ext = os.path.splitext(filename)
    candidate = filename
    k = 1
    while candidate.lower() in used:
        candidate = "%s_%d%s" % (base, k, ext)
        k += 1
    used.add(candidate.lower())
    return os.path.join(dst_dir, candidate)


def write_split(output_dir, split, class_names, by_class, hil_chosen,
                label_of, dry_run):
    """Copies images into the target split directory and returns manifest rows."""
    rows = []
    for cls in class_names:
        files = by_class.get(cls, [])
        dst_dir = os.path.join(output_dir, split, cls)
        if not dry_run:
            os.makedirs(dst_dir, exist_ok=True)
        used = set()
        for src in files:
            dst = safe_dst(dst_dir, os.path.basename(src), used)
            if not dry_run:
                shutil.copy2(src, dst)
            rows.append({
                "split": split,
                "class_name": cls,
                "label": label_of[cls],
                "filename": os.path.basename(dst),
                "src_path": src.replace(os.sep, "/"),
                "dst_path": dst.replace(os.sep, "/"),
                "hil_subset": 1 if (split == "test" and
                                    src in hil_chosen.get(cls, ())) else 0,
            })
    return rows


def write_manifest(path, rows):
    cols = ["split", "class_name", "label", "filename",
            "src_path", "dst_path", "hil_subset"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Reproducible train/val/test dataset split for PHLAME.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--input_dir", default="data/processed/160x120",
                    help="Processed directory with one subfolder per class")
    ap.add_argument("--output_dir", default="data/splits",
                    help="Output directory for the generated splits")
    ap.add_argument("--train", type=float, default=0.70, help="Train fraction")
    ap.add_argument("--val", type=float, default=0.15, help="Validation fraction")
    ap.add_argument("--test", type=float, default=0.15, help="Test fraction")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed fixing a reproducible split")
    ap.add_argument("--hil-subset", dest="hil_subset", type=int, default=200,
                    help="Number of test images flagged for the HIL benchmark (0 = none)")
    ap.add_argument("--class_names_path", default="models/class_names.txt",
                    help="File defining the canonical label order (argmax)")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite output_dir if it already exists")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true",
                    help="Do not copy files; only report what would be done")
    args = ap.parse_args()

    # ---- input validation -------------------------------------------------
    total_p = args.train + args.val + args.test
    if abs(total_p - 1.0) > 1e-6:
        sys.exit("ERROR: split fractions must sum to 1.0 (got %.4f)" % total_p)
    if not os.path.isdir(args.input_dir):
        sys.exit("ERROR: input directory does not exist: %s" % args.input_dir)

    if os.path.isdir(args.output_dir) and os.listdir(args.output_dir):
        if not (args.force or args.dry_run):
            sys.exit("ERROR: %s already exists and is not empty. Use --force to "
                     "overwrite (or --dry-run to inspect only)."
                     % args.output_dir)
        if args.force and not args.dry_run:
            for sp in SPLITS:
                shutil.rmtree(os.path.join(args.output_dir, sp),
                              ignore_errors=True)

    class_names, class_src = load_class_names(args.class_names_path,
                                              args.input_dir)
    label_of = {c: i for i, c in enumerate(class_names)}

    print("Input       : %s" % args.input_dir)
    print("Output      : %s" % args.output_dir)
    print("Classes     : %s  (order from %s)" % (class_names, class_src))
    print("Proportions : train %.2f / val %.2f / test %.2f | seed %d"
          % (args.train, args.val, args.test, args.seed))
    if args.dry_run:
        print(">>> DRY RUN: no files will be copied <<<")
    print("")

    # ---- stratified per-class split ---------------------------------------
    rng = random.Random(args.seed)
    per_split = {sp: {} for sp in SPLITS}
    counts = {}

    for cls in class_names:
        class_dir = os.path.join(args.input_dir, cls)
        if not os.path.isdir(class_dir):
            sys.exit("ERROR: missing subfolder for class '%s' in %s"
                     % (cls, args.input_dir))
        files = collect_images(class_dir)
        if not files:
            sys.exit("ERROR: class '%s' contains no images" % cls)

        tr, va, te = split_one_class(files, args.train, args.val, rng)
        per_split["train"][cls] = tr
        per_split["val"][cls] = va
        per_split["test"][cls] = te
        counts[cls] = {"total": len(files), "train": len(tr),
                       "val": len(va), "test": len(te)}

    # ---- fixed HIL subset -------------------------------------------------
    hil_chosen = pick_hil_subset(per_split["test"], args.hil_subset,
                                 random.Random(args.seed + 1))

    # ---- leakage verification (the primary reason this script exists) -----
    seen = {}
    duplicated = []
    for sp in SPLITS:
        for cls, files in per_split[sp].items():
            for f in files:
                if f in seen:
                    duplicated.append((f, seen[f], sp))
                seen[f] = sp
    if duplicated:
        for f, a, b in duplicated[:5]:
            print("LEAK: %s appears in both '%s' and '%s'" % (f, a, b))
        sys.exit("ERROR: splits are not disjoint (%d collisions)."
                 % len(duplicated))

    n_total_src = sum(c["total"] for c in counts.values())
    if len(seen) != n_total_src:
        sys.exit("ERROR: %d images assigned across splits, expected %d from source."
                 % (len(seen), n_total_src))

    # ---- write splits and manifests ---------------------------------------
    if not args.dry_run:
        os.makedirs(args.output_dir, exist_ok=True)

    manifest_rows = {}
    for sp in SPLITS:
        rows = write_split(args.output_dir, sp, class_names, per_split[sp],
                           hil_chosen, label_of, args.dry_run)
        manifest_rows[sp] = rows
        if not args.dry_run:
            write_manifest(os.path.join(args.output_dir,
                                        "%s_manifest.csv" % sp), rows)

    # ---- summary report ---------------------------------------------------
    totals = {sp: sum(len(v) for v in per_split[sp].values()) for sp in SPLITS}
    balance_pct = {cls: round(100.0 * counts[cls]["total"] / n_total_src, 2)
                   for cls in class_names}
    hil_per_class = {cls: len(hil_chosen.get(cls, ())) for cls in class_names}

    print("%-14s %8s %8s %8s %8s" % ("Class", "total", "train", "val", "test"))
    for cls in class_names:
        c = counts[cls]
        print("%-14s %8d %8d %8d %8d"
              % (cls, c["total"], c["train"], c["val"], c["test"]))
    print("%-14s %8d %8d %8d %8d"
          % ("TOTAL", n_total_src, totals["train"], totals["val"],
             totals["test"]))
    print("")
    print("Class balance (%% of dataset): %s" % balance_pct)
    print("HIL subset     : %d images marked -> %s"
          % (sum(hil_per_class.values()), hil_per_class))
    print("Verification   : disjoint splits OK, %d images accounted for"
          % len(seen))

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "split_dataset.py",
        "input_dir": args.input_dir.replace(os.sep, "/"),
        "output_dir": args.output_dir.replace(os.sep, "/"),
        "seed": args.seed,
        "proportions": {"train": args.train, "val": args.val,
                        "test": args.test},
        "class_names": class_names,
        "class_names_source": class_src,
        "label_map": label_of,
        "counts_per_class": counts,
        "totals": dict(totals, dataset=n_total_src),
        "class_balance_pct": balance_pct,
        "hil_subset": {"requested": args.hil_subset,
                       "selected": sum(hil_per_class.values()),
                       "per_class": hil_per_class,
                       "seed": args.seed + 1},
        "integrity": {"splits_disjoint": True,
                      "images_accounted_for": len(seen) == n_total_src},
        "dry_run": args.dry_run,
    }

    if not args.dry_run:
        with open(os.path.join(args.output_dir, "split_report.json"),
                  "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        print("")
        print("Written: %s/{train,val,test}/ + *_manifest.csv + "
              "split_report.json" % args.output_dir)
    else:
        print("")
        print("(dry-run) nothing was written.")


if __name__ == "__main__":
    main()
