#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
split_dataset.py - Particion reproducible del dataset para PHLAME.

Ultimo paso de la preparacion de datos, ANTES de cualquier entrenamiento.
Toma el directorio ya procesado y redimensionado (salida de reshape_images.py)
y produce tres particiones disjuntas en disco mas los manifiestos CSV que
garantizan que los cuatro niveles de la escalera de fidelidad
(MIL -> SIL -> PIL -> HIL) evaluen EXACTAMENTE las mismas imagenes.

Por que existe este script
--------------------------
Sin una particion explicita, train_*.py usa `validation_split` sobre el
directorio completo y los evaluadores (test_model.py, test_tflite_model.py)
leen ESE MISMO directorio completo -> la accuracy reportada incluye datos de
entrenamiento (fuga de datos). Ademas, cada banco seleccionaba sus imagenes de
forma distinta (carpeta manual en PIL, muestreo aleatorio en HIL), por lo que
los saltos de la escalera no eran atribuibles. Este script resuelve ambos.

Entrada
-------
    <input_dir>/
        Class1/*.jpg
        Class2/*.jpg

Salida
------
    <output_dir>/
        train/Class1/... , train/Class2/...
        val/Class1/...   , val/Class2/...
        test/Class1/...  , test/Class2/...
        train_manifest.csv
        val_manifest.csv
        test_manifest.csv        <- incluye la columna hil_subset (1/0)
        split_report.json        <- conteos, balance de clases, verificaciones

Uso tipico
----------
    python src/split_dataset.py \
        --input_dir data/processed/160x120 \
        --output_dir data/splits \
        --train 0.70 --val 0.15 --test 0.15 \
        --seed 42 --hil-subset 200

Como lo consumen los cuatro niveles
-----------------------------------
    train_*.py            --> data/splits/train      (val: data/splits/val)
    test_model.py         --> data/splits/test       (MIL, float, PC)
    test_tflite_model.py  --> data/splits/test       (SIL, INT8, PC)
    pil_benchmark.py      --> imagenes de test_manifest.csv
    hil_camera_benchmark  --> filas con hil_subset == 1

Notas de reproducibilidad
-------------------------
- La lista de archivos se ordena (sorted) ANTES de mezclar, porque el orden de
  os.listdir depende del sistema de archivos. Con la misma semilla, la particion
  es identica en cualquier maquina.
- El orden de las etiquetas se toma de models/class_names.txt, el mismo que
  define el argmax que devuelve el firmware. No se infiere del sistema de
  archivos salvo que el archivo no exista.
- La particion es estratificada: se aplica por clase, de modo que cada
  particion conserva la proporcion natural de clases del dataset original
  (el desbalance no se corrige aqui; se documenta en split_report.json).
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
# Descubrimiento de clases e imagenes
# --------------------------------------------------------------------------- #
def load_class_names(class_names_path, input_dir):
    """Orden de etiquetas: class_names.txt si existe, si no las subcarpetas."""
    if class_names_path and os.path.exists(class_names_path):
        with open(class_names_path, encoding="utf-8") as fh:
            names = [ln.strip() for ln in fh if ln.strip()]
        if names:
            return names, "models/class_names.txt"

    names = sorted(
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d))
    )
    return names, "subcarpetas de %s (class_names.txt no encontrado)" % input_dir


def collect_images(class_dir):
    """Todas las imagenes de una clase, recursivo y en orden determinista."""
    found = []
    for root, _dirs, files in os.walk(class_dir):
        for fn in files:
            if fn.lower().endswith(IMG_EXT):
                found.append(os.path.join(root, fn))
    return sorted(found)


# --------------------------------------------------------------------------- #
# Particion
# --------------------------------------------------------------------------- #
def split_one_class(files, p_train, p_val, rng):
    """Particion estratificada de una clase. Devuelve (train, val, test)."""
    items = list(files)
    rng.shuffle(items)

    n = len(items)
    n_train = int(round(n * p_train))
    n_val = int(round(n * p_val))
    # el test recibe el resto, para que las tres particiones sumen n exactamente
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)

    return (
        items[:n_train],
        items[n_train:n_train + n_val],
        items[n_train + n_val:],
    )


def pick_hil_subset(test_by_class, n_target, rng):
    """Subconjunto fijo y estratificado del test para el banco HIL."""
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
            k = min(n_target - assigned, len(pool))          # resto a la ultima
        else:
            k = int(round(n_target * len(pool) / total_test))
            k = max(0, min(k, len(pool)))
        chosen[cls] = set(rng.sample(pool, k)) if k else set()
        assigned += k
    return chosen


# --------------------------------------------------------------------------- #
# Escritura
# --------------------------------------------------------------------------- #
def safe_dst(dst_dir, filename, used):
    """Destino sin colisiones (subcarpetas anidadas pueden repetir nombre)."""
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
    """Copia las imagenes y devuelve las filas del manifiesto."""
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
        description="Particion reproducible train/val/test para PHLAME.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--input_dir", default="data/processed/160x120",
                    help="Directorio procesado con subcarpetas por clase")
    ap.add_argument("--output_dir", default="data/splits",
                    help="Directorio de salida de las particiones")
    ap.add_argument("--train", type=float, default=0.70, help="Fraccion train")
    ap.add_argument("--val", type=float, default=0.15, help="Fraccion val")
    ap.add_argument("--test", type=float, default=0.15, help="Fraccion test")
    ap.add_argument("--seed", type=int, default=42,
                    help="Semilla; fija la particion de forma reproducible")
    ap.add_argument("--hil-subset", dest="hil_subset", type=int, default=200,
                    help="Imagenes del test marcadas para el banco HIL (0 = ninguna)")
    ap.add_argument("--class_names_path", default="models/class_names.txt",
                    help="Archivo que fija el orden de etiquetas (argmax)")
    ap.add_argument("--force", action="store_true",
                    help="Sobrescribe output_dir si ya existe")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true",
                    help="No copia nada; solo informa lo que haria")
    args = ap.parse_args()

    # ---- validaciones de entrada -----------------------------------------
    total_p = args.train + args.val + args.test
    if abs(total_p - 1.0) > 1e-6:
        sys.exit("ERROR: las fracciones deben sumar 1.0 (suman %.4f)" % total_p)
    if not os.path.isdir(args.input_dir):
        sys.exit("ERROR: no existe el directorio de entrada: %s" % args.input_dir)

    if os.path.isdir(args.output_dir) and os.listdir(args.output_dir):
        if not (args.force or args.dry_run):
            sys.exit("ERROR: %s ya existe y no esta vacio. Usa --force para "
                     "sobrescribir (o --dry-run para solo inspeccionar)."
                     % args.output_dir)
        if args.force and not args.dry_run:
            for sp in SPLITS:
                shutil.rmtree(os.path.join(args.output_dir, sp),
                              ignore_errors=True)

    class_names, class_src = load_class_names(args.class_names_path,
                                              args.input_dir)
    label_of = {c: i for i, c in enumerate(class_names)}

    print("Entrada     : %s" % args.input_dir)
    print("Salida      : %s" % args.output_dir)
    print("Clases      : %s  (orden desde %s)" % (class_names, class_src))
    print("Proporciones: train %.2f / val %.2f / test %.2f | semilla %d"
          % (args.train, args.val, args.test, args.seed))
    if args.dry_run:
        print(">>> DRY RUN: no se copiara ningun archivo <<<")
    print("")

    # ---- particion estratificada por clase -------------------------------
    rng = random.Random(args.seed)
    per_split = {sp: {} for sp in SPLITS}
    counts = {}

    for cls in class_names:
        class_dir = os.path.join(args.input_dir, cls)
        if not os.path.isdir(class_dir):
            sys.exit("ERROR: falta la subcarpeta de la clase '%s' en %s"
                     % (cls, args.input_dir))
        files = collect_images(class_dir)
        if not files:
            sys.exit("ERROR: la clase '%s' no tiene imagenes" % cls)

        tr, va, te = split_one_class(files, args.train, args.val, rng)
        per_split["train"][cls] = tr
        per_split["val"][cls] = va
        per_split["test"][cls] = te
        counts[cls] = {"total": len(files), "train": len(tr),
                       "val": len(va), "test": len(te)}

    # ---- subconjunto HIL fijo --------------------------------------------
    hil_chosen = pick_hil_subset(per_split["test"], args.hil_subset,
                                 random.Random(args.seed + 1))

    # ---- verificacion de fuga (el motivo de este script) -----------------
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
            print("FUGA: %s aparece en '%s' y '%s'" % (f, a, b))
        sys.exit("ERROR: las particiones no son disjuntas (%d colisiones)."
                 % len(duplicated))

    n_total_src = sum(c["total"] for c in counts.values())
    if len(seen) != n_total_src:
        sys.exit("ERROR: %d imagenes repartidas pero %d en el origen."
                 % (len(seen), n_total_src))

    # ---- escritura --------------------------------------------------------
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

    # ---- informe ----------------------------------------------------------
    totals = {sp: sum(len(v) for v in per_split[sp].values()) for sp in SPLITS}
    balance_pct = {cls: round(100.0 * counts[cls]["total"] / n_total_src, 2)
                   for cls in class_names}
    hil_per_class = {cls: len(hil_chosen.get(cls, ())) for cls in class_names}

    print("%-14s %8s %8s %8s %8s" % ("Clase", "total", "train", "val", "test"))
    for cls in class_names:
        c = counts[cls]
        print("%-14s %8d %8d %8d %8d"
              % (cls, c["total"], c["train"], c["val"], c["test"]))
    print("%-14s %8d %8d %8d %8d"
          % ("TOTAL", n_total_src, totals["train"], totals["val"],
             totals["test"]))
    print("")
    print("Balance de clases (%% del dataset): %s" % balance_pct)
    print("Subconjunto HIL: %d imagenes marcadas -> %s"
          % (sum(hil_per_class.values()), hil_per_class))
    print("Verificacion   : particiones disjuntas OK, %d imagenes repartidas"
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
        print("Escrito: %s/{train,val,test}/ + *_manifest.csv + "
              "split_report.json" % args.output_dir)
    else:
        print("")
        print("(dry-run) no se escribio nada.")


if __name__ == "__main__":
    main()
