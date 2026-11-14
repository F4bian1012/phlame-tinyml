#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_results.py - Says whether a reproduced run gives the published result.

After repeating a published run into another folder, for example

    python src/mil_benchmark.py --model_path models/checkpoints/{model}.keras \
        --splits_dir data/splits_coco --output_dir results/smoke_test/mil
    python src/compare_results.py results/smoke_test results

every metrics_*.json found under the first folder is compared with the file of
the same level and name under the second one:

  - the evaluated model file (SHA-256 and size),
  - the class names and the number of images,
  - the global metrics, the per-class metrics and the confusion matrix,
  - the prediction of every image (predictions_*.csv, joined on RelPath).

What legitimately differs between two runs of the same thing is left out: the
timestamp, the recorded paths and their separators, and the latency measured
on a board. Exit code 0 when every compared run is identical, 1 otherwise, or
when there was nothing to compare.

Standard library only.
"""
import csv
import json
import os
import sys

import cli


def find_metrics(folder):
    """metrics_*.json under folder, as paths relative to it (with '/')."""
    found = []
    for root, _dirs, files in os.walk(folder):
        for name in files:
            if name.startswith("metrics_") and name.endswith(".json"):
                found.append(os.path.relpath(os.path.join(root, name), folder).replace(os.sep, "/"))
    return sorted(found)


def read_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def read_predictions(metrics_path):
    """{RelPath with '/': (true label, predicted label)} of the CSV next to a metrics file, or None."""
    folder, name = os.path.split(metrics_path)
    path = os.path.join(folder, "predictions_" + name[len("metrics_"):-len(".json")] + ".csv")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter=";"))
    return {r["RelPath"].replace("\\", "/"): (r["TrueLabel"], r["PredLabel"]) for r in rows}


def n_images(doc):
    dataset = doc.get("dataset") or {}
    for key in ("n_samples", "n_images", "n_stimuli"):
        if key in dataset:
            return dataset[key]
    return None


def compare_run(candidate_path, reference_path):
    """List of differences between two result files of the same run (empty = identical)."""
    a, b = read_json(candidate_path), read_json(reference_path)
    differences = []

    def check(label, x, y):
        if x != y:
            differences.append("%s: %s  vs  %s" % (label, json.dumps(x)[:90], json.dumps(y)[:90]))

    check("level", a.get("level"), b.get("level"))
    for key in ("sha256", "bytes"):
        check("model." + key, (a.get("model") or {}).get(key), (b.get("model") or {}).get(key))
    check("dataset.class_names", (a.get("dataset") or {}).get("class_names"),
          (b.get("dataset") or {}).get("class_names"))
    check("number of images", n_images(a), n_images(b))
    for key in ("n_answered", "n_unanswered"):
        check("dataset." + key, (a.get("dataset") or {}).get(key), (b.get("dataset") or {}).get(key))
    check("global_metrics", a.get("global_metrics"), b.get("global_metrics"))
    check("per_class_metrics", a.get("per_class_metrics"), b.get("per_class_metrics"))
    for key in ("labels", "matrix"):
        check("confusion_matrix." + key, (a.get("confusion_matrix") or {}).get(key),
              (b.get("confusion_matrix") or {}).get(key))

    pa, pb = read_predictions(candidate_path), read_predictions(reference_path)
    if pa is None or pb is None:
        if pa is not pb:
            differences.append("predictions_*.csv is missing on one side")
        compared = None
    else:
        if set(pa) != set(pb):
            differences.append("the two runs did not evaluate the same images (%d vs %d, %d in common)"
                               % (len(pa), len(pb), len(set(pa) & set(pb))))
        changed = sorted(k for k in set(pa) & set(pb) if pa[k] != pb[k])
        if changed:
            differences.append("%d images have a different prediction or label, e.g. %s"
                               % (len(changed), ", ".join(changed[:3])))
        compared = len(set(pa) & set(pb))
    return differences, compared, a


def main():
    parser = cli.ArgumentParser(
        description="Compare the result files of reproduced runs with the published ones.")
    parser.add_argument("candidate", help="Folder with the reproduced results, e.g. results/smoke_test "
                                          "(holding mil/, sil/, ... like results/)")
    parser.add_argument("reference", nargs="?", default="results",
                        help="Folder with the results to compare against (default: results)")
    args = parser.parse_args()

    runs = find_metrics(args.candidate)
    if not runs:
        print("ERROR: no metrics_*.json found under %s" % args.candidate)
        return 1

    n_bad = 0
    for rel in runs:
        reference_path = os.path.join(args.reference, rel)
        if not os.path.isfile(reference_path):
            print("NOT FOUND  %s  (no such file under %s)" % (rel, args.reference))
            n_bad += 1
            continue
        differences, compared, doc = compare_run(os.path.join(args.candidate, rel), reference_path)
        accuracy = (doc.get("global_metrics") or {}).get("accuracy")
        detail = "%s images compared one by one" % compared if compared is not None else "no per-image file"
        if accuracy is not None:
            detail += ", accuracy %.4f" % accuracy
        if differences:
            n_bad += 1
            print("DIFFERENT  %s" % rel)
            for line in differences:
                print("             %s" % line)
        else:
            print("identical  %s  (%s)" % (rel, detail))

    print("")
    print("%d of %d runs identical to %s" % (len(runs) - n_bad, len(runs), args.reference))
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
