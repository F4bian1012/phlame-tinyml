#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reshape_images.py - Resizes the dataset and optionally balances it.

Intermediate step in PHLAME data preparation:

    process_images.py  ->  data/processed/grayscale      (grayscale conversion)
    reshape_images.py  ->  data/processed/{W}x{H}        (THIS script)
    split_dataset.py   ->  data/splits/{train,val,test}  (stratified split)

Why balancing is performed HERE
-------------------------------
split_dataset.py performs a *stratified* split: it intentionally preserves the
natural class ratio and documents class imbalance in split_report.json instead
of correcting it. If a balanced dataset is desired, that decision must be made
before splitting.

Of the two preceding steps, this is the appropriate place:

  - process_images.py produces the complete grayscale superset. Keeping it
    intact allows re-balancing with another seed or another per-class cap
    without repeating the full dataset conversion, which is the expensive step.
  - reshape_images.py is the last step before splitting and only resizes the
    images that are kept: no work is wasted on discarded files.

Balancing strategy
------------------
ONLY majority-class undersampling is implemented, never oversampling by file
duplication. Duplicating files BEFORE split_dataset.py would place identical
copies of the same image in both train and test: data leakage. Test accuracy
would rise without the model actually improving, and all four levels of the
fidelity ladder (MIL/SIL/PIL/HIL) would inherit the inflated metric.

If class imbalance must be compensated without discarding images, the proper
approach is class_weight during training, not duplicating files on disk.

Selection is reproducible: the file list is sorted before sampling, just as in
split_dataset.py, so the same seed produces the exact same subset on any
machine.

Usage
-----
    # Resize only (historical behavior: nothing is discarded)
    python src/reshape_images.py --width 160 --height 120

    # Resize and balance via undersampling
    python src/reshape_images.py --width 160 --height 120 --balance undersample

    # Balance and also cap the maximum number of images per class
    python src/reshape_images.py --width 160 --height 120 --balance undersample --max_per_class 2000

    # Inspect the per-class breakdown without writing anything
    python src/reshape_images.py --width 160 --height 120 --balance undersample --dry-run
"""

import argparse
import json
import os
import random
from datetime import datetime, timezone

import cv2

PROCESSED_BASE_DIR = "data/processed"
# Accepted extensions, compared in lowercase (including .JPG, .Png, etc.)
EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')
REPORT_NAME = "reshape_report.json"
# Key recorded in reshape_report.json for loose files in the root folder
# (mapped to English in the console table printed by imprimir_reparto)
SIN_CLASE = "(raiz, sin clase)"


def collect_by_class(input_dir):
    """Groups images by class.

    The class is the first path component relative to input_dir. Loose files in
    the root directory (without a class subfolder) are grouped under None and
    are never discarded: they do not belong to any balanceable class.
    """
    por_clase = {}
    for root, _, names in os.walk(input_dir):
        for name in names:
            if os.path.splitext(name)[1].lower() not in EXTENSIONS:
                continue
            ruta = os.path.join(root, name)
            rel = os.path.relpath(ruta, input_dir)
            partes = rel.replace(os.sep, '/').split('/')
            clase = partes[0] if len(partes) > 1 else None
            por_clase.setdefault(clase, []).append(ruta)

    # Sort BEFORE any sampling: the traversal order of os.walk depends on the
    # filesystem, and without sorting, the seed would not be reproducible across
    # machines. Same precaution as in split_dataset.py.
    for clase in por_clase:
        por_clase[clase].sort()
    return por_clase


def select_files(por_clase, balance, max_per_class, seed):
    """Applies class balancing. Returns (selected_files, per_class_detail, target)."""
    clases = sorted([c for c in por_clase if c is not None])
    sin_clase = por_clase.get(None, [])

    objetivo = None
    if balance == 'undersample' and clases:
        objetivo = min(len(por_clase[c]) for c in clases)
    if max_per_class is not None:
        objetivo = max_per_class if objetivo is None else min(objetivo, max_per_class)

    rng = random.Random(seed)
    seleccion = []
    detalle = {}

    # Fixed class order so that RNG consumption is deterministic.
    for clase in clases:
        disponibles = por_clase[clase]
        if objetivo is None or len(disponibles) <= objetivo:
            elegidas = list(disponibles)
        else:
            # Sort at the end only to keep the write traversal order stable;
            # the subset choice was already determined by the seed.
            elegidas = sorted(rng.sample(disponibles, objetivo))
        detalle[clase] = {
            "disponibles": len(disponibles),
            "seleccionadas": len(elegidas),
            "descartadas": len(disponibles) - len(elegidas),
        }
        seleccion.extend(elegidas)

    if sin_clase:
        seleccion.extend(sin_clase)
        detalle[SIN_CLASE] = {
            "disponibles": len(sin_clase),
            "seleccionadas": len(sin_clase),
            "descartadas": 0,
        }

    return seleccion, detalle, objetivo


def imprimir_reparto(detalle, total_encontradas, total_seleccionadas):
    print()
    print("  %-24s %11s %14s %12s"
          % ("Class", "available", "selected", "discarded"))
    print("  " + "-" * 64)
    for clase in sorted(detalle):
        d = detalle[clase]
        display_name = "(root, unclassified)" if clase == SIN_CLASE else clase
        print("  %-24s %11d %14d %12d"
              % (display_name, d["disponibles"], d["seleccionadas"], d["descartadas"]))
    print("  " + "-" * 64)
    print("  %-24s %11d %14d %12d"
          % ("TOTAL", total_encontradas, total_seleccionadas,
             total_encontradas - total_seleccionadas))
    print()


def reshape_images(width, height, input_dir, balance, max_per_class, seed, dry_run, output_dir=None):
    if not output_dir:
        output_dir = os.path.join(PROCESSED_BASE_DIR, "%dx%d" % (width, height))

    print("Searching for images recursively in %s..." % input_dir)
    por_clase = collect_by_class(input_dir)
    total_encontradas = sum(len(v) for v in por_clase.values())

    if not total_encontradas:
        print("No images found in %s or its subdirectories" % input_dir)
        return 1

    seleccion, detalle, objetivo = select_files(
        por_clase, balance, max_per_class, seed)

    print("Found %d images." % total_encontradas)
    imprimir_reparto(detalle, total_encontradas, len(seleccion))

    # Warn when the dataset is imbalanced and balancing was not requested:
    # without this warning, class imbalance propagates silently to final metrics.
    conteos = [d["seleccionadas"] for c, d in detalle.items() if c != SIN_CLASE]
    if balance == 'none' and len(conteos) > 1 and min(conteos) != max(conteos):
        print("WARNING: the dataset is imbalanced (%d vs %d) and --balance is 'none',"
              % (max(conteos), min(conteos)))
        print("         so class imbalance is preserved as is. split_dataset.py performs a")
        print("         STRATIFIED split: it preserves that ratio in train/val/test")
        print("         rather than correcting it. To balance:  --balance undersample")
        print()

    if dry_run:
        print("--dry-run: nothing was written to %s" % output_dir)
        return 0

    os.makedirs(output_dir, exist_ok=True)
    print("Resizing %d images to %dx%d..." % (len(seleccion), width, height))

    processed_count = 0
    fallidas = 0

    for file_path in seleccion:
        try:
            img = cv2.imread(file_path)
            if img is None:
                print("Failed to load: %s" % file_path)
                fallidas += 1
                continue

            resized = cv2.resize(img, (width, height))

            rel_path = os.path.relpath(file_path, input_dir)
            output_path = os.path.join(output_dir, rel_path)
            destino = os.path.dirname(output_path)
            if destino:
                os.makedirs(destino, exist_ok=True)

            cv2.imwrite(output_path, resized)
            processed_count += 1

        except Exception as e:
            print("Error processing %s: %s" % (file_path, e))
            fallidas += 1

    reporte = {
        "generado_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": input_dir,
        "output_dir": output_dir,
        "target_size": {"width": width, "height": height},
        "balance": {
            "modo": balance,
            "max_per_class": max_per_class,
            "objetivo_por_clase": objetivo,
            "seed": seed,
        },
        "clases": detalle,
        "totales": {
            "encontradas": total_encontradas,
            "seleccionadas": len(seleccion),
            "escritas": processed_count,
            "fallidas": fallidas,
        },
    }
    ruta_reporte = os.path.join(output_dir, REPORT_NAME)
    with open(ruta_reporte, "w", encoding="utf-8") as fh:
        json.dump(reporte, fh, indent=2, ensure_ascii=False)

    print()
    print("========================================")
    print("Reshape Complete!")
    print("Target Size: %dx%d" % (width, height))
    print("Output Folder: %s" % output_dir)
    print("Balance: %s (seed %d)" % (balance, seed))
    print("Processed: %d/%d" % (processed_count, len(seleccion)))
    if fallidas:
        print("Failed: %d" % fallidas)
    print("Report: %s" % ruta_reporte)
    print("========================================")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Resize (and optionally balance) the dataset for TinyML",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--width", type=int, default=28, help="Target width")
    parser.add_argument("--height", type=int, default=28, help="Target height")
    parser.add_argument("--input_dir", type=str, default="data/processed/grayscale",
                        help="Input directory")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory; overrides data/processed/{width}x{height}")
    parser.add_argument("--balance", choices=["none", "undersample"], default="none",
                        help="'undersample' trims every class down to the size of the "
                             "smallest class. Oversampling is intentionally omitted: "
                             "duplicating files before split_dataset.py causes data "
                             "leakage between train and test")
    parser.add_argument("--max_per_class", type=int, default=None,
                        help="Hard cap on images per class; combines with --balance")
    parser.add_argument("--seed", type=int, default=42,
                        help="Undersampling random seed (same convention as split_dataset.py)")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true",
                        help="Show the per-class breakdown without writing anything")

    args = parser.parse_args()

    if args.max_per_class is not None and args.max_per_class < 1:
        parser.error("--max_per_class must be >= 1")

    raise SystemExit(reshape_images(args.width, args.height, args.input_dir,
                                    args.balance, args.max_per_class,
                                    args.seed, args.dry_run, args.output_dir))
