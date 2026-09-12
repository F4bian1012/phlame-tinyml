#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reshape_images.py - Redimensiona el dataset y, opcionalmente, lo balancea.

Paso intermedio de la preparacion de datos de PHLAME:

    process_images.py  ->  data/processed/grayscale      (conversion a gris)
    reshape_images.py  ->  data/processed/{W}x{H}        (ESTE script)
    split_dataset.py   ->  data/splits/{train,val,test}  (particion estratificada)

Por que el balanceo se hace AQUI
--------------------------------
split_dataset.py hace una particion *estratificada*: conserva a proposito la
proporcion natural de clases y documenta el desbalance en split_report.json en
lugar de corregirlo. Si se quiere un dataset balanceado, la decision tiene que
tomarse antes de particionar.

De los dos pasos previos, este es el adecuado:

  - process_images.py produce el superconjunto completo en escala de grises.
    Conservarlo intacto permite re-balancear con otra semilla o con otro tope
    sin repetir la conversion de todo el dataset, que es el paso caro.
  - reshape_images.py es el ultimo paso antes de la particion y, ademas, solo
    redimensiona las imagenes que se conservan: no gasta trabajo en las que se
    van a descartar.

Estrategia de balanceo
----------------------
Se implementa SOLO submuestreo de las clases mayoritarias, nunca sobremuestreo
por duplicacion. Duplicar archivos ANTES de split_dataset.py meteria copias
identicas de la misma imagen en train y en test: fuga de datos. La accuracy de
test subiria sin que el modelo haya mejorado, y los cuatro niveles de la
escalera (MIL/SIL/PIL/HIL) heredarian la medida inflada.

Si se necesita compensar el desbalance sin perder imagenes, el camino correcto
es class_weight en el entrenamiento, no duplicar archivos en disco.

La seleccion es reproducible: la lista se ordena (sorted) antes de mezclar,
igual que en split_dataset.py, de modo que la misma semilla produce el mismo
subconjunto en cualquier maquina.

Uso
---
    # Solo redimensionar (comportamiento historico: no descarta nada)
    python src/reshape_images.py --width 160 --height 120

    # Redimensionar y balancear por submuestreo
    python src/reshape_images.py --width 160 --height 120 --balance undersample

    # Balancear y ademas limitar el tamano por clase
    python src/reshape_images.py --width 160 --height 120 --balance undersample --max_per_class 2000

    # Ver el reparto por clase sin escribir nada
    python src/reshape_images.py --width 160 --height 120 --balance undersample --dry-run
"""

import argparse
import glob
import json
import os
import random
from datetime import datetime, timezone

import cv2

PROCESSED_BASE_DIR = "data/processed"
EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
REPORT_NAME = "reshape_report.json"
SIN_CLASE = "(raiz, sin clase)"


def collect_by_class(input_dir):
    """Agrupa las imagenes por clase.

    La clase es el primer componente de la ruta relativa a input_dir. Los
    archivos sueltos en la raiz (sin subcarpeta) se agrupan bajo None y nunca
    se descartan: no pertenecen a ninguna clase que se pueda balancear.
    """
    por_clase = {}
    for ext in EXTENSIONS:
        patron = os.path.join(input_dir, '**', ext)
        for ruta in glob.glob(patron, recursive=True):
            rel = os.path.relpath(ruta, input_dir)
            partes = rel.replace(os.sep, '/').split('/')
            clase = partes[0] if len(partes) > 1 else None
            por_clase.setdefault(clase, []).append(ruta)

    # sorted() ANTES de cualquier mezcla: el orden que devuelve glob depende del
    # sistema de archivos, y sin ordenar la semilla no seria reproducible entre
    # maquinas. Misma precaucion que en split_dataset.py.
    for clase in por_clase:
        por_clase[clase].sort()
    return por_clase


def select_files(por_clase, balance, max_per_class, seed):
    """Aplica el balanceo. Devuelve (seleccionadas, detalle_por_clase, objetivo)."""
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

    # Orden fijo de clases para que el consumo del generador sea determinista.
    for clase in clases:
        disponibles = por_clase[clase]
        if objetivo is None or len(disponibles) <= objetivo:
            elegidas = list(disponibles)
        else:
            # sorted() al final solo para que el recorrido de escritura sea
            # estable; la eleccion ya la fijo la semilla.
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
          % ("Clase", "disponibles", "seleccionadas", "descartadas"))
    print("  " + "-" * 64)
    for clase in sorted(detalle):
        d = detalle[clase]
        print("  %-24s %11d %14d %12d"
              % (clase, d["disponibles"], d["seleccionadas"], d["descartadas"]))
    print("  " + "-" * 64)
    print("  %-24s %11d %14d %12d"
          % ("TOTAL", total_encontradas, total_seleccionadas,
             total_encontradas - total_seleccionadas))
    print()


def reshape_images(width, height, input_dir, balance, max_per_class, seed, dry_run):
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

    # Aviso cuando el dataset esta desbalanceado y no se pidio corregirlo: sin
    # esto el desbalance se propaga en silencio hasta las metricas finales.
    conteos = [d["seleccionadas"] for c, d in detalle.items() if c != SIN_CLASE]
    if balance == 'none' and len(conteos) > 1 and min(conteos) != max(conteos):
        print("AVISO: el dataset esta desbalanceado (%d vs %d) y --balance es 'none',"
              % (max(conteos), min(conteos)))
        print("       asi que se copia tal cual. split_dataset.py particiona de forma")
        print("       ESTRATIFICADA: conserva esa proporcion en train/val/test, no la")
        print("       corrige. Para balancear:  --balance undersample")
        print()

    if dry_run:
        print("--dry-run: no se escribio nada en %s" % output_dir)
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
        print("Fallidas: %d" % fallidas)
    print("Reporte: %s" % ruta_reporte)
    print("========================================")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Redimensiona (y opcionalmente balancea) el dataset para TinyML",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--width", type=int, default=28, help="Target width")
    parser.add_argument("--height", type=int, default=28, help="Target height")
    parser.add_argument("--input_dir", type=str, default="data/processed/grayscale",
                        help="Input directory")
    parser.add_argument("--balance", choices=["none", "undersample"], default="none",
                        help="'undersample' recorta cada clase al tamano de la mas "
                             "pequena. No hay sobremuestreo a proposito: duplicar "
                             "archivos antes de split_dataset.py provoca fuga de "
                             "datos entre train y test")
    parser.add_argument("--max_per_class", type=int, default=None,
                        help="Tope duro de imagenes por clase; se combina con --balance")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla del submuestreo (misma convencion que split_dataset.py)")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true",
                        help="Muestra el reparto por clase y no escribe nada")

    args = parser.parse_args()

    if args.max_per_class is not None and args.max_per_class < 1:
        parser.error("--max_per_class debe ser >= 1")

    raise SystemExit(reshape_images(args.width, args.height, args.input_dir,
                                    args.balance, args.max_per_class,
                                    args.seed, args.dry_run))
