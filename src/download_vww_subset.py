#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_vww_subset.py - Descarga un subconjunto balanceado de Visual Wake Words
(VWW, Chowdhery et al., 2019 / MLPerf Tiny) directamente desde MS-COCO 2017 y lo
deja listo en data/raw_vww/{non_person,person}/ para ejecutar la escalera
MIL -> SIL -> PIL -> HIL de PHLAME (Actividad 48).

Regla oficial de etiquetado VWW (arXiv:1906.05721)
--------------------------------------------------
  - 'person'     : la imagen contiene al menos una anotacion de la categoria
                   'person' (category_id == 1, iscrowd == 0) cuya caja o area
                   segmentada ocupa >= 0.5% (0.005) del area total de la imagen.
  - 'non_person' : la imagen no contiene ninguna anotacion de 'person' (por
                   defecto se excluyen las imagenes ambiguas con personas
                   diminutas < 0.5% del area para evitar ruido de etiqueta;
                   pasa --include-small-person-as-negative para incluirlas
                   como 'non_person' al estilo estricto de VWW).

Eficiencia de descarga
----------------------
En lugar de bajar los ~20 GB de train2017.zip o los 241 MB completos de
annotations_trainval2017.zip, este script:
  1. Usa peticiones HTTP Range sobre el directorio central del ZIP remoto de
     COCO para descargar UNICAMENTE annotations/instances_val2017.json (~6 MB
     comprimidos).
  2. Selecciona de forma determinista (--seed 42) N imagenes por clase
     (--per_class, por defecto 1000 -> 2000 imagenes en total) y descarga solo
     esos JPEGs en paralelo (--workers 16) a data/raw_vww/{non_person,person}/.
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import struct
import sys
import time
import urllib.request
import zlib
from datetime import datetime, timezone

ANNOTATIONS_ZIP_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMAGE_BASE_URL = "http://images.cocodataset.org/{split}/{file_name}"
PERSON_CATEGORY_ID = 1


def _http_get_range(url: str, start: int, end: int, timeout: int = 60) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes={start}-{end}",
            "User-Agent": "PHLAME-VWW-Downloader/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_coco_instances_json(cache_path: str, target_member: str) -> dict:
    """Obtiene instances_{split}.json desde el ZIP remoto usando HTTP Range."""
    if os.path.isfile(cache_path) and os.path.getsize(cache_path) > 1024:
        print(f"[CACHE] Leyendo anotaciones desde {cache_path} ...")
        with open(cache_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    print(f"[NET] Consultando tamano de {ANNOTATIONS_ZIP_URL} ...")
    req = urllib.request.Request(
        ANNOTATIONS_ZIP_URL,
        method="HEAD",
        headers={"User-Agent": "PHLAME-VWW-Downloader/1.0"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        total_size = int(resp.headers["Content-Length"])

    tail_len = min(65536, total_size)
    tail = _http_get_range(ANNOTATIONS_ZIP_URL, total_size - tail_len, total_size - 1)
    eocd_pos = tail.rfind(b"PK\x05\x06")
    if eocd_pos < 0:
        raise RuntimeError("No se encontro el registro EOCD en el ZIP remoto de COCO.")

    cd_size, cd_offset = struct.unpack("<II", tail[eocd_pos + 12 : eocd_pos + 20])
    cd = _http_get_range(ANNOTATIONS_ZIP_URL, cd_offset, cd_offset + cd_size - 1)

    pos = 0
    while pos < len(cd):
        header = cd[pos : pos + 46]
        (
            _sig,
            _ver_made,
            _ver_need,
            _flags,
            comp_method,
            _mod_time,
            _mod_date,
            _crc32,
            comp_size,
            uncomp_size,
            name_len,
            extra_len,
            comm_len,
            _disk_start,
            _int_attr,
            _ext_attr,
            lfh_offset,
        ) = struct.unpack("<IHHHHHHIIIHHHHHII", header)
        name = cd[pos + 46 : pos + 46 + name_len].decode("utf-8", errors="replace")
        if name == target_member:
            print(
                f"[NET] Descargando {target_member} via HTTP Range "
                f"({comp_size / (1024 * 1024):.2f} MB comprimidos -> "
                f"{uncomp_size / (1024 * 1024):.2f} MB)..."
            )
            lfh = _http_get_range(
                ANNOTATIONS_ZIP_URL, lfh_offset, lfh_offset + 30 + name_len + 256
            )
            l_name_len, l_extra_len = struct.unpack("<HH", lfh[26:30])
            data_start = lfh_offset + 30 + l_name_len + l_extra_len
            comp_data = _http_get_range(
                ANNOTATIONS_ZIP_URL, data_start, data_start + comp_size - 1, timeout=180
            )
            if comp_method == 8:
                raw_bytes = zlib.decompress(comp_data, -zlib.MAX_WBITS)
            elif comp_method == 0:
                raw_bytes = comp_data
            else:
                raise RuntimeError(f"Metodo de compresion ZIP no soportado: {comp_method}")

            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as fh:
                fh.write(raw_bytes)
            return json.loads(raw_bytes.decode("utf-8"))

        pos += 46 + name_len + extra_len + comm_len

    raise RuntimeError(f"No se encontro {target_member} dentro del ZIP de COCO.")


def classify_vww_images(coco_doc: dict, min_area_fraction: float, include_small_as_neg: bool):
    """Clasifica las imagenes de COCO en person / non_person segun la regla VWW."""
    images_by_id = {}
    for img in coco_doc["images"]:
        img_id = int(img["id"])
        w = int(img.get("width", 0))
        h = int(img.get("height", 0))
        if w > 0 and h > 0:
            images_by_id[img_id] = {
                "id": img_id,
                "file_name": img["file_name"],
                "width": w,
                "height": h,
                "area": float(w * h),
                "max_person_frac": 0.0,
                "has_any_person": False,
                "has_crowd_person": False,
            }

    for ann in coco_doc["annotations"]:
        if int(ann.get("category_id", -1)) != PERSON_CATEGORY_ID:
            continue
        img_id = int(ann["image_id"])
        info = images_by_id.get(img_id)
        if info is None:
            continue
        info["has_any_person"] = True
        if int(ann.get("iscrowd", 0)) != 0:
            info["has_crowd_person"] = True
            continue
        # Regla VWW: area de la caja delimitadora (o segmentacion) respecto a la imagen
        bbox = ann.get("bbox", [0, 0, 0, 0])
        bbox_area = float(bbox[2]) * float(bbox[3]) if len(bbox) == 4 else 0.0
        seg_area = float(ann.get("area", 0.0))
        eff_area = max(bbox_area, seg_area)
        frac = eff_area / info["area"]
        if frac > info["max_person_frac"]:
            info["max_person_frac"] = frac

    positives = []
    negatives = []
    ambiguous = []

    for img_id in sorted(images_by_id.keys()):
        info = images_by_id[img_id]
        if info["max_person_frac"] >= min_area_fraction:
            positives.append(info)
        elif not info["has_any_person"]:
            negatives.append(info)
        else:
            if include_small_as_neg and not info["has_crowd_person"]:
                negatives.append(info)
            else:
                ambiguous.append(info)

    return positives, negatives, ambiguous


def _download_one(task):
    url, dst_path, timeout = task
    if os.path.isfile(dst_path) and os.path.getsize(dst_path) > 0:
        return ("cached", dst_path, None)
    tmp_path = dst_path + ".part"
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "PHLAME-VWW-Downloader/1.0"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            with open(tmp_path, "wb") as fh:
                fh.write(data)
            os.replace(tmp_path, dst_path)
            return ("downloaded", dst_path, None)
        except Exception as exc:
            if attempt == 2:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
                return ("failed", dst_path, str(exc))
            time.sleep(0.5 * (attempt + 1))


def main():
    parser = argparse.ArgumentParser(
        description="Descarga un subconjunto balanceado de Visual Wake Words (VWW) desde MS-COCO 2017",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw_vww",
        help="Directorio de salida donde se crearan las subcarpetas person/ y non_person/",
    )
    parser.add_argument(
        "--coco_split",
        choices=["val2017", "train2017"],
        default="val2017",
        help="Particion de COCO 2017 a usar como fuente (val2017 tiene 5000 imagenes y basta para hasta ~2100 por clase)",
    )
    parser.add_argument(
        "--per_class",
        type=int,
        default=1000,
        help="Numero de imagenes a descargar por cada clase (person y non_person)",
    )
    parser.add_argument(
        "--min_area_fraction",
        type=float,
        default=0.005,
        help="Umbral minimo de area relativa de 'person' segun VWW (0.005 = 0.5%% del area de la imagen)",
    )
    parser.add_argument(
        "--include-small-person-as-negative",
        dest="include_small_as_neg",
        action="store_true",
        help="Incluye imagenes con personas <0.5%% del area dentro de non_person (por defecto se descartan por limpieza)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para muestreo reproducible de imagenes",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Hilos paralelos de descarga HTTP",
    )
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Descarga/lee solo las anotaciones, muestra el conteo por clase y termina sin bajar imagenes",
    )
    args = parser.parse_args()

    cache_dir = os.path.join(args.output_dir, "_cache")
    cache_json = os.path.join(cache_dir, f"instances_{args.coco_split}.json")
    target_member = f"annotations/instances_{args.coco_split}.json"

    coco_doc = fetch_coco_instances_json(cache_json, target_member)
    positives, negatives, ambiguous = classify_vww_images(
        coco_doc, args.min_area_fraction, args.include_small_as_neg
    )

    print()
    print("============================================================")
    print(f"Resumen Visual Wake Words sobre COCO {args.coco_split}")
    print("============================================================")
    print(f"  person (>= {args.min_area_fraction*100:.2f}% area) : {len(positives):5d} disponibles")
    print(f"  non_person (0 personas)         : {len(negatives):5d} disponibles")
    print(f"  ambiguas (< {args.min_area_fraction*100:.2f}% o crowd) : {len(ambiguous):5d} excluidas")
    print("------------------------------------------------------------")

    max_posible = min(len(positives), len(negatives))
    if args.per_class > max_posible:
        print(
            f"[WARN] Pediste --per_class {args.per_class}, pero en {args.coco_split} "
            f"el maximo balanceado es {max_posible}. Ajustando a {max_posible} "
            f"(usa --coco_split train2017 si necesitas mas)."
        )
        objetivo = max_posible
    else:
        objetivo = args.per_class

    rng = random.Random(args.seed)
    sel_pos = sorted(rng.sample(positives, objetivo), key=lambda x: x["file_name"])
    sel_neg = sorted(rng.sample(negatives, objetivo), key=lambda x: x["file_name"])

    print(f"  Seleccionadas por clase (seed={args.seed}): {objetivo} person + {objetivo} non_person = {2 * objetivo} total")
    print("============================================================")

    if args.dry_run:
        print("\n[--dry-run] No se descargaron imagenes.")
        return 0

    person_dir = os.path.join(args.output_dir, "person")
    non_person_dir = os.path.join(args.output_dir, "non_person")
    os.makedirs(person_dir, exist_ok=True)
    os.makedirs(non_person_dir, exist_ok=True)

    tasks = []
    for item in sel_pos:
        url = IMAGE_BASE_URL.format(split=args.coco_split, file_name=item["file_name"])
        dst = os.path.join(person_dir, item["file_name"])
        tasks.append((url, dst, 30))
    for item in sel_neg:
        url = IMAGE_BASE_URL.format(split=args.coco_split, file_name=item["file_name"])
        dst = os.path.join(non_person_dir, item["file_name"])
        tasks.append((url, dst, 30))

    print(f"\nDescargando {len(tasks)} imagenes en {args.output_dir} con {args.workers} hilos...")
    t0 = time.time()
    counts = {"downloaded": 0, "cached": 0, "failed": 0}
    failures = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_download_one, t) for t in tasks]
        for idx, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            status, dst_path, err = fut.result()
            counts[status] += 1
            if err:
                failures.append((dst_path, err))
            if idx % 100 == 0 or idx == len(tasks):
                elapsed = time.time() - t0
                print(
                    f"  [{idx:4d}/{len(tasks):4d}] nuevas={counts['downloaded']} "
                    f"en_cache={counts['cached']} fallidas={counts['failed']} "
                    f"({elapsed:.1f}s)"
                )

    manifest_hash = hashlib.sha256(
        json.dumps(
            {
                "person": [x["file_name"] for x in sel_pos],
                "non_person": [x["file_name"] for x in sel_neg],
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    report = {
        "generado_utc": datetime.now(timezone.utc).isoformat(),
        "coco_split": args.coco_split,
        "min_area_fraction": args.min_area_fraction,
        "include_small_person_as_negative": args.include_small_as_neg,
        "seed": args.seed,
        "per_class_target": objetivo,
        "selection_sha256": manifest_hash,
        "available_in_split": {
            "person": len(positives),
            "non_person": len(negatives),
            "ambiguous_excluded": len(ambiguous),
        },
        "download_summary": counts,
    }
    report_path = os.path.join(args.output_dir, "vww_subset_report.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    print(f"\n[OK] Subconjunto VWW listo en: {args.output_dir}")
    print(f"     - {person_dir} ({objetivo} imagenes)")
    print(f"     - {non_person_dir} ({objetivo} imagenes)")
    print(f"     - Reporte: {report_path} (sha256 seleccion: {manifest_hash[:12]})")
    if failures:
        print(f"[WARN] {len(failures)} descargas fallaron. Vuelve a correr el script para reintentar solo las faltantes.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
