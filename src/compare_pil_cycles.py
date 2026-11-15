#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_pil_cycles.py - The on-chip latency of two or more PIL runs of the same
model, on the images they have in common.

    python src/compare_pil_cycles.py results/pil results/nicla/pil --model_path models/tflite/{model_name}_int8.tflite --output_dir results/smoke_test/tables

A PIL run measures, for every image, the cycles and the microseconds of each
phase on the chip (pre-processing, inference, post-processing and their total).
This script puts runs of the same .tflite side by side: another board, or the
same board with its tensor arena somewhere else
(compile_upload_arduino.py --arena_memory internal). Runs need not cover the
same images: the means are taken over the images every run measured, so a run
of the 200 images of the HIL subset is compared with a run of the whole test
split on those 200.

Each run is given as a folder that holds the files pil_benchmark.py wrote
(latency_metrics_<run>.csv and metrics_<run>.json), or as one of those two
files. For a folder, --model_path says which run to take: the stem of the
.tflite is the run name. The first run given is the reference of the
differences.

What is read:
  latency_metrics_<run>.csv   per image: Image;TEMP_C;CYC_PRE;CYC_INF;CYC_POST;
                              CYC_TOTAL;US_PRE;US_INF;US_POST;US_TOTAL
                              (an image the board did not answer has empty
                              cells and is left out)
  metrics_<run>.json          run.board, model.arena_memory, model.sha256 and
                              model.name; the runs must name the same model
                              file. Without this file the run is compared all
                              the same, with its board and arena not known.

The arena memory of a run is the one its firmware announced
(model.arena_memory). A file that does not record it was measured with a
firmware that had no such line, in which the arena is where the default build
of the board has it; the table gives that place and marks it as not recorded.

Outputs, in --output_dir:
  pil_cycles_<run>.csv   one row per run, unrounded: means in cycles and in
                         microseconds, and the difference to the first run in
                         per cent
  pil_cycles_<run>.md    the same as tables

Exit codes: 0 the table was written; 1 a file is missing or cannot be used,
the runs are of different models or have no image in common; 2 wrong options.
Standard library only: no board, no images.
"""

import csv
import glob
import json
import os
import statistics
import sys

import boards
import cli

PHASES = ("PRE", "INF", "POST", "TOTAL")
UNITS = ("CYC", "US")
FIELDS = ["%s_%s" % (unit, phase) for unit in UNITS for phase in PHASES]
LATENCY_PREFIX = "latency_metrics_"
METRICS_PREFIX = "metrics_"
# pil_benchmark.py --image appends the single images it sends to this file: not a run
SINGLE_PREFIX = LATENCY_PREFIX + "single_"


class Problem(Exception):
    """Something the operator has to fix; the text is the message to show."""


def _shown(path):
    """A path as it is written in the outputs: as given, with '/'."""
    return os.path.normpath(path).replace(os.sep, "/")


def locate(run, stem=None):
    """(latency CSV, metrics JSON, run name) of a run given as a folder or as one of its two files.

    stem is the run name asked for with --model_path; a folder that holds one
    run does not need it. The metrics path is returned whether or not the file
    exists."""
    if os.path.isdir(run):
        if stem is None:
            found = [p for p in sorted(glob.glob(os.path.join(run, LATENCY_PREFIX + "*.csv")))
                     if not os.path.basename(p).startswith(SINGLE_PREFIX)]
            if len(found) != 1:
                raise Problem("%s holds %d PIL runs: say which one with --model_path" % (_shown(run), len(found)))
            stem = os.path.basename(found[0])[len(LATENCY_PREFIX):-len(".csv")]
        folder = run
    elif os.path.isfile(run):
        name = os.path.basename(run)
        if name.startswith(LATENCY_PREFIX) and name.endswith(".csv"):
            stem = name[len(LATENCY_PREFIX):-len(".csv")]
        elif name.startswith(METRICS_PREFIX) and name.endswith(".json"):
            stem = name[len(METRICS_PREFIX):-len(".json")]
        else:
            raise Problem("%s is neither a %s*.csv nor a %s*.json file of pil_benchmark.py"
                          % (_shown(run), LATENCY_PREFIX, METRICS_PREFIX))
        folder = os.path.dirname(run)
    else:
        raise Problem("%s does not exist" % _shown(run))
    latency = os.path.join(folder, "%s%s.csv" % (LATENCY_PREFIX, stem))
    if not os.path.isfile(latency):
        raise Problem("no %s in %s" % (os.path.basename(latency), _shown(folder) or "."))
    return latency, os.path.join(folder, "%s%s.json" % (METRICS_PREFIX, stem)), stem


def read_latency(path):
    """{image name: {field: value}} of a latency CSV, with the images the board answered.

    An image without an answer has empty cells and is left out. Stops when the
    file has not the columns pil_benchmark.py writes, or names an image twice
    (two classes with a file of the same name): the file identifies an image by
    its name alone, so those rows could not be told apart."""
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter=";")
        missing = [c for c in ["Image"] + FIELDS if c not in (reader.fieldnames or [])]
        if missing:
            raise Problem("%s is not a latency file of pil_benchmark.py: no column %s"
                          % (_shown(path), ", ".join(missing)))
        rows, seen = {}, set()
        for row in reader:
            name = row["Image"]
            if name in seen:
                raise Problem("%s has more than one row for %s: its images cannot be matched by name"
                              % (_shown(path), name))
            seen.add(name)
            try:
                rows[name] = {field: float(row[field]) for field in FIELDS}
            except (TypeError, ValueError):
                continue
    return rows


def read_metrics(path):
    """What the metrics JSON of a run says about it; every value is None when there is no file."""
    about = {"board": None, "arena_memory": None, "sha256": None, "name": None}
    if not os.path.isfile(path):
        return about
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
    except ValueError as exc:
        raise Problem("%s cannot be read as JSON (%s)" % (_shown(path), exc))
    model = doc.get("model") or {}
    about.update(board=(doc.get("run") or {}).get("board"), arena_memory=model.get("arena_memory"),
                 sha256=model.get("sha256"), name=model.get("name"))
    return about


def arena_memory(about):
    """(where the arena of the run was, whether its file records it).

    A file without model.arena_memory comes from a firmware that did not
    announce it: the arena was then where the default build of the board has
    it. None when the board is not known either."""
    if about["arena_memory"]:
        return about["arena_memory"], True
    key = boards.key_of(about["board"])
    return (boards.BOARDS[key]["arena_memory"] if key else None), False


def compare(runs, stem=None):
    """One row per run, in the order given, over the images all of them measured.

    Each row holds 'run', 'latency_csv', 'board', 'arena_memory',
    'arena_memory_recorded', 'n_images_run', 'n_images_common' and, for every
    phase and unit, '<UNIT>_<PHASE>_mean' and '<UNIT>_<PHASE>_vs_first_pct'
    (0 for the first run). Returns (rows, name of the model)."""
    loaded = []
    for run in runs:
        latency, metrics, name = locate(run, stem)
        loaded.append({"run": run, "latency": latency, "name": name, "rows": read_latency(latency),
                       "about": read_metrics(metrics)})
    digests = {item["about"]["sha256"] for item in loaded if item["about"]["sha256"]}
    if len(digests) > 1:
        raise Problem("the runs are not of the same model file (model.sha256): "
                      + "; ".join("%s %s" % (_shown(item["run"]), (item["about"]["sha256"] or "not recorded")[:12])
                                  for item in loaded))
    common = set(loaded[0]["rows"])
    for item in loaded[1:]:
        common &= set(item["rows"])
    if not common:
        raise Problem("the runs have no image in common: " + "; ".join(
            "%s %d images" % (_shown(item["run"]), len(item["rows"])) for item in loaded))
    common = sorted(common)

    rows = []
    for item in loaded:
        where, recorded = arena_memory(item["about"])
        row = {"run": _shown(item["run"]), "latency_csv": _shown(item["latency"]),
               "board": item["about"]["board"], "arena_memory": where, "arena_memory_recorded": recorded,
               "n_images_run": len(item["rows"]), "n_images_common": len(common)}
        for field in FIELDS:
            row[field + "_mean"] = statistics.fmean(item["rows"][name][field] for name in common)
        rows.append(row)
    for row in rows:
        for field in FIELDS:
            first = rows[0][field + "_mean"]
            row[field + "_vs_first_pct"] = (100.0 * (row[field + "_mean"] - first) / first) if first else None
    return rows, loaded[0]["about"]["name"] or loaded[0]["name"]


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def write_csv(path, rows):
    columns = list(rows[0])
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(["" if row[c] is None else row[c] for c in columns])


def _board(row):
    key = boards.key_of(row["board"])
    return boards.BOARDS[key]["label"] if key else (row["board"] or "not recorded")


def _arena(row):
    if row["arena_memory"] is None:
        return "not recorded"
    return row["arena_memory"] + ("" if row["arena_memory_recorded"] else " (board default)")


def markdown(rows, model):
    n = rows[0]["n_images_common"]
    lines = ["# PIL latency of `%s`, run by run" % model, "",
             "Built by `src/compare_pil_cycles.py` from the per-image latency files of %d PIL runs of the same "
             "model file. Every figure is a mean over the %d images that all the runs measured." % (len(rows), n),
             "", "## Core cycles per phase", ""]
    head = ["Run", "Board", "Arena memory", "Images of the run", "Images in common"]
    lines += ["| " + " | ".join(head + ["`%s`" % p for p in PHASES]) + " |", "|---|---|---|" + "---:|" * (2 + len(PHASES))]
    for row in rows:
        cells = ["`%s`" % row["run"], _board(row), _arena(row), str(row["n_images_run"]), str(n)]
        cells += ["{:,.0f}".format(row["CYC_%s_mean" % p]) for p in PHASES]
        lines.append("| " + " | ".join(cells) + " |")
    if any(row["arena_memory"] and not row["arena_memory_recorded"] for row in rows):
        lines += ["", "\"(board default)\": the files of the run do not record where the arena was. It was measured "
                      "with a firmware that did not announce it, in which the arena is where the default build of "
                      "the board has it."]

    lines += ["", "## Microseconds per phase", "",
              "| " + " | ".join(["Run"] + ["`%s`" % p for p in PHASES]) + " |", "|---|" + "---:|" * len(PHASES)]
    for row in rows:
        lines.append("| " + " | ".join(["`%s`" % row["run"]]
                                       + ["{:,.3f}".format(row["US_%s_mean" % p]) for p in PHASES]) + " |")

    lines += ["", "## Difference to the first run", "",
              "Cycles of each run relative to those of `%s`, in per cent; negative is fewer cycles." % rows[0]["run"], "",
              "| " + " | ".join(["Run"] + ["`%s`" % p for p in PHASES]) + " |", "|---|" + "---:|" * len(PHASES)]
    for row in rows[1:]:
        cells = ["—" if row["CYC_%s_vs_first_pct" % p] is None else "%+.2f%%" % row["CYC_%s_vs_first_pct" % p]
                 for p in PHASES]
        lines.append("| " + " | ".join(["`%s`" % row["run"]] + cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = cli.ArgumentParser(description="On-chip latency of two or more PIL runs of the same model, on the "
                                            "images they have in common: mean cycles and microseconds per "
                                            "phase, and the difference of each run to the first.")
    parser.add_argument("runs", nargs="+",
                        help="Two or more PIL runs: each a folder with the files pil_benchmark.py wrote, or one "
                             "of its latency_metrics_*.csv / metrics_*.json files. The first is the reference")
    parser.add_argument("--model_path", default=None,
                        help="The .tflite the runs measured, or the run name alone (the file is not opened: its "
                             "name without .tflite is the run name). It says which run to take from a folder "
                             "that holds several")
    parser.add_argument("--output_dir", required=True,
                        help="Where pil_cycles_<run>.csv and .md are written")
    args = parser.parse_args(argv)
    if len(args.runs) < 2:
        parser.error("give at least two runs to compare")

    # The run name is the file name of the model without its extension. Only that
    # extension is cut: the name itself has dots (the learning rate).
    stem = os.path.basename(args.model_path) if args.model_path else None
    if stem and stem.lower().endswith(".tflite"):
        stem = stem[:-len(".tflite")]
    try:
        rows, model = compare(args.runs, stem)
    except Problem as problem:
        print("ERROR: %s" % problem)
        return 1

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "pil_cycles_%s.csv" % model)
    md_path = os.path.join(args.output_dir, "pil_cycles_%s.md" % model)
    write_csv(csv_path, rows)
    text = markdown(rows, model)
    with open(md_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    try:
        print(text)
    except UnicodeEncodeError:      # a console that cannot show the dash of an empty cell
        print(text.encode("ascii", "replace").decode("ascii"))
    for item in rows:
        if not item["board"]:
            print("[WARN] %s: no metrics file next to the latency file, or no board in it" % item["run"])
    print("Written: %s" % csv_path)
    print("Written: %s" % md_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
