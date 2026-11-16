#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_benchmark_table.py - The summary table of a benchmark measured up to PIL,
rebuilt from the result files in results/.

    python src/build_benchmark_table.py --tag COCO
    python src/build_benchmark_table.py --tag COCO --output_dir results/smoke_test/tables

For every run named {architecture}_{tag}+... that has a MIL result it gathers:

  results/mil/                 float model on the PC
  results/sil/                 INT8 model on the PC (the SIL of record)
  results/pil/                 INT8 model on the Portenta H7
  results/nicla/pil/           INT8 model on the Nicla Vision
  results/sil_kernels/<kind>/  the same SIL with other TFLite kernels, when present

and counts, image by image (joined on RelPath), how many predictions two levels
have in common: MIL = SIL, SIL = PIL on each board, one board = the other, and
each alternative SIL = the SIL of record and = PIL.

It also says which model each PIL run agrees with. A PIL run names the .tflite
it measured, but the bench only records that name: a board still carrying the
model of an earlier run would be filed under the wrong one. Each PIL run is
therefore compared with the SIL of record of every model of the benchmark: the
model it names must be the one it agrees with, far ahead of the others. (Runs
measured with a firmware that announces the model fingerprint are also checked
directly, before measuring; for runs measured before that, this table is the
only check.)

Outputs, in --output_dir (default results/tables):
  benchmark_<tag>.csv              one row per run, unrounded
  benchmark_<tag>_attribution.csv  one row per PIL run and board: images in common with the SIL of each model
  benchmark_<tag>.md               the same as three tables: the levels, the effect of the SIL kernels,
                                   and the model each PIL run agrees with

build_ladder_tables.py builds the statistical tables of a benchmark that also
has HIL runs; this script covers the levels below it and needs no statistics.
Standard library only: no TensorFlow, no board, no images.
"""

import csv
import json
import os
import sys

import boards
import cli

KERNEL_KINDS = ("default", "builtin", "reference")


def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def predictions(path):
    """{RelPath with '/': predicted label} of a predictions CSV, or None when the file is missing.

    Images the board did not answer have an empty label and are left out."""
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8", newline="") as fh:
        return {row["RelPath"].replace("\\", "/"): row["PredLabel"]
                for row in csv.DictReader(fh, delimiter=";") if row["PredLabel"] != ""}


def same(a, b):
    """(images with the same prediction, images compared) for two prediction tables, or None."""
    if a is None or b is None:
        return None
    common = sorted(set(a) & set(b))
    return sum(1 for key in common if a[key] == b[key]), len(common)


def find_runs(results_dir, tag):
    """Run names {architecture}_{tag}+... that have a MIL metrics file, sorted."""
    folder = os.path.join(results_dir, "mil")
    runs = []
    if os.path.isdir(folder):
        for name in os.listdir(folder):
            if name.startswith("metrics_") and name.endswith(".json"):
                run = name[len("metrics_"):-len(".json")]
                if run.split("+", 1)[0].endswith("_" + tag):
                    runs.append(run)
    return sorted(runs)


def gather(results_dir, run, tag, board_keys):
    """Everything the table says about one run, unrounded."""
    row = {"run": run, "architecture": run.split("+", 1)[0][:-len("_" + tag)]}
    mil = load_json(os.path.join(results_dir, "mil", "metrics_%s.json" % run))
    row["n_test"] = mil["dataset"]["n_samples"]
    row["mil_accuracy"] = mil["global_metrics"]["accuracy"]
    row["mil_f1_macro"] = mil["global_metrics"]["f1_macro"]
    p_mil = predictions(os.path.join(results_dir, "mil", "predictions_%s.csv" % run))

    p_sil = None
    sil_path = os.path.join(results_dir, "sil", "metrics_%s_int8.json" % run)
    if os.path.isfile(sil_path):
        sil = load_json(sil_path)
        row["tflite_bytes"] = sil["model"].get("bytes")
        row["tflite_sha256"] = sil["model"].get("sha256")
        # Files written before the option existed were measured with the default interpreter.
        row["sil_kernels"] = sil["model"].get("kernels", "default")
        row["sil_accuracy"] = sil["global_metrics"]["accuracy"]
        row["sil_f1_macro"] = sil["global_metrics"]["f1_macro"]
        p_sil = predictions(os.path.join(results_dir, "sil", "predictions_%s_int8.csv" % run))
        row["mil_eq_sil"] = same(p_mil, p_sil)

    p_board = {}
    for key in board_keys:
        folder = os.path.join(results_dir, *boards.results_dir(key, "pil").replace("\\", "/").split("/")[1:])
        path = os.path.join(folder, "metrics_%s_int8.json" % run)
        if not os.path.isfile(path):
            continue
        pil = load_json(path)
        fields = pil["latency"]["per_field"]
        p_board[key] = predictions(os.path.join(folder, "predictions_%s_int8.csv" % run))
        row["pil_%s_accuracy" % key] = pil["global_metrics"]["accuracy"]
        row["pil_%s_f1_macro" % key] = pil["global_metrics"]["f1_macro"]
        row["pil_%s_n_images" % key] = pil["dataset"]["n_images"]
        row["pil_%s_n_answered" % key] = pil["dataset"]["n_answered"]
        row["pil_%s_board" % key] = pil["run"].get("board")
        row["pil_%s_n_reconnects" % key] = pil["run"].get("n_reconnects")
        row["pil_%s_same_file_as_sil" % key] = (pil["model"].get("sha256") == row.get("tflite_sha256"))
        row["pil_%s_inf_ms" % key] = fields["US_INF"]["mean"] / 1000.0
        row["pil_%s_total_ms" % key] = fields["US_TOTAL"]["mean"] / 1000.0
        row["pil_%s_inf_cycles" % key] = fields["CYC_INF"]["mean"]
        row["sil_eq_pil_%s" % key] = same(p_sil, p_board[key])
    if len(p_board) >= 2:
        first, second = [key for key in board_keys if key in p_board][:2]
        row["%s_eq_%s" % (first, second)] = same(p_board[first], p_board[second])

    for kind in KERNEL_KINDS:
        folder = os.path.join(results_dir, "sil_kernels", kind)
        path = os.path.join(folder, "metrics_%s_int8.json" % run)
        if not os.path.isfile(path):
            continue
        other = load_json(path)
        p_other = predictions(os.path.join(folder, "predictions_%s_int8.csv" % run))
        row["sil_%s_accuracy" % kind] = other["global_metrics"]["accuracy"]
        row["sil_%s_f1_macro" % kind] = other["global_metrics"]["f1_macro"]
        row["sil_%s_eq_sil" % kind] = same(p_other, p_sil)
        for key in p_board:
            row["sil_%s_eq_pil_%s" % (kind, key)] = same(p_other, p_board[key])
    return row


def attribution(results_dir, rows, board_keys):
    """For every PIL run on each board, the test images on which it gives the prediction of the SIL
    of record of each model of the benchmark.

    Returns a list of {"run", "architecture", "board", "n", "same": {architecture: count}, "own",
    "best_other", "best_other_architecture", "attributed"}; attributed is True when the model the run
    names is the one it agrees with most."""
    sil = {}
    for row in rows:
        table = predictions(os.path.join(results_dir, "sil", "predictions_%s_int8.csv" % row["run"]))
        if table is not None:
            sil[row["architecture"]] = table
    out = []
    for row in rows:
        for key in board_keys:
            folder = os.path.join(results_dir, *boards.results_dir(key, "pil").replace("\\", "/").split("/")[1:])
            pil = predictions(os.path.join(folder, "predictions_%s_int8.csv" % row["run"]))
            if pil is None or row["architecture"] not in sil:
                continue
            counts = {arch: same(table, pil)[0] for arch, table in sil.items()}
            own = counts[row["architecture"]]
            others = {arch: c for arch, c in counts.items() if arch != row["architecture"]}
            best = max(others, key=others.get) if others else None
            out.append({"run": row["run"], "architecture": row["architecture"], "board": key, "n": len(pil),
                        "same": counts, "own": own, "best_other": others[best] if best else None,
                        "best_other_architecture": best,
                        "attributed": bool(best is None or own > others[best])})
    return out


def write_attribution_csv(path, table, architectures):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["run", "architecture", "board", "n_answered"] + ["same_as_sil_%s" % a for a in architectures]
                        + ["own", "best_other", "best_other_architecture", "attributed"])
        for item in table:
            writer.writerow([item["run"], item["architecture"], item["board"], item["n"]]
                            + [item["same"].get(a, "") for a in architectures]
                            + [item["own"], _cell(item["best_other"]), _cell(item["best_other_architecture"]),
                               item["attributed"]])


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def _cell(value):
    if isinstance(value, tuple):
        return "%d/%d" % value
    return "" if value is None else value


def write_csv(path, rows):
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([_cell(row.get(c)) for c in columns])


def _acc(row, prefix):
    if row.get(prefix + "_accuracy") is None:
        return "—"
    return "%.4f / %.4f" % (row[prefix + "_accuracy"], row[prefix + "_f1_macro"])


def _same(value):
    return "—" if value is None else "%d / %d" % value


def markdown(rows, tag, board_keys, attributed=()):
    labels = {key: boards.BOARDS[key]["label"] for key in board_keys}
    lines = ["# Benchmark `%s`: MIL, SIL and PIL" % tag, "",
             "Built by `src/build_benchmark_table.py --tag %s` from the result files. Accuracy / macro-F1; "
             "`=` columns count the test images with the same prediction at both levels." % tag, ""]
    head = ["Architecture", "`.tflite` (INT8)", "n", "MIL", "SIL", "MIL = SIL"]
    for key in board_keys:
        head += ["PIL %s" % labels[key], "SIL = PIL", "`INF` (ms)", "`TOTAL` (ms)"]
    if len(board_keys) >= 2:
        head.append("%s = %s" % (labels[board_keys[0]], labels[board_keys[1]]))
    lines += ["| " + " | ".join(head) + " |", "|---|" + "---:|" * (len(head) - 1)]
    for row in rows:
        cells = ["`%s`" % row["architecture"],
                 "%.1f KiB" % (row["tflite_bytes"] / 1024.0) if row.get("tflite_bytes") else "—",
                 str(row["n_test"]), _acc(row, "mil"), _acc(row, "sil"), _same(row.get("mil_eq_sil"))]
        for key in board_keys:
            if row.get("pil_%s_accuracy" % key) is None:
                cells += ["—", "—", "—", "—"]
                continue
            answered = ""
            if row["pil_%s_n_answered" % key] != row["pil_%s_n_images" % key]:
                answered = " (%d answered)" % row["pil_%s_n_answered" % key]
            cells += [_acc(row, "pil_%s" % key) + answered, _same(row.get("sil_eq_pil_%s" % key)),
                      "%.2f" % row["pil_%s_inf_ms" % key], "%.2f" % row["pil_%s_total_ms" % key]]
        if len(board_keys) >= 2:
            cells.append(_same(row.get("%s_eq_%s" % (board_keys[0], board_keys[1]))))
        lines.append("| " + " | ".join(cells) + " |")

    kinds = [k for k in KERNEL_KINDS if any(("sil_%s_accuracy" % k) in row for row in rows)]
    if kinds:
        lines += ["", "## The same INT8 file with other TFLite kernels on the PC", "",
                  "The SIL column above uses the kernels named under \"SIL of record\". Each other kind is "
                  "compared with it and with the board (%s)." % " / ".join(labels[key] for key in board_keys), ""]
        head = ["Architecture", "SIL of record"]
        for kind in kinds:
            head += ["SIL `%s`" % kind, "= SIL of record", "= PIL"]
        lines += ["| " + " | ".join(head) + " |", "|---|---|" + "---:|" * (len(head) - 2)]
        for row in rows:
            cells = ["`%s`" % row["architecture"], "`%s`" % row.get("sil_kernels", "—")]
            for kind in kinds:
                if row.get("sil_%s_accuracy" % kind) is None:
                    cells += ["—", "—", "—"]
                    continue
                on_boards = [_same(row.get("sil_%s_eq_pil_%s" % (kind, key))) for key in board_keys]
                cells += [_acc(row, "sil_%s" % kind), _same(row.get("sil_%s_eq_sil" % kind)), " · ".join(on_boards)]
            lines.append("| " + " | ".join(cells) + " |")

    if attributed:
        architectures = [row["architecture"] for row in rows if any(row["architecture"] in a["same"] for a in attributed)]
        lines += ["", "## Which model each PIL run agrees with", "",
                  "Test images on which a PIL run gives the prediction of the SIL of record of each model. The bench "
                  "records the model a run names, not the one the board carries, so this is the check that each run "
                  "was filed under the model it measured: its own model (in bold) must be the best match.", ""]
        head = ["PIL run", "Board", "Images"] + ["SIL `%s`" % a for a in architectures] + ["Own model is the best match"]
        lines += ["| " + " | ".join(head) + " |", "|---|---|" + "---:|" * (len(head) - 2)]
        for item in attributed:
            cells = ["`%s`" % item["architecture"], labels[item["board"]], str(item["n"])]
            for a in architectures:
                count = item["same"].get(a)
                text = "—" if count is None else str(count)
                cells.append("**%s**" % text if a == item["architecture"] else text)
            cells.append("yes" if item["attributed"] else "**NO**")
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = cli.ArgumentParser(description="Summary table of a benchmark measured up to PIL, rebuilt from "
                                                 "the result files (MIL, SIL, PIL on each board).")
    parser.add_argument("--tag", required=True,
                        help="Tag of the runs to gather, as given to the training scripts (e.g. COCO for the runs "
                             "named {architecture}_COCO+...)")
    parser.add_argument("--results_dir", default="results", help="Folder with the result files (default: results)")
    parser.add_argument("--output_dir", default=os.path.join("results", "tables"),
                        help="Where benchmark_<tag>.csv and .md are written (default: results/tables)")
    parser.add_argument("--boards", default="portenta,nicla",
                        help="Boards whose PIL runs are gathered, in column order (default: portenta,nicla)")
    args = parser.parse_args()

    board_keys = [key for key in args.boards.split(",") if key]
    for key in board_keys:
        if key not in boards.BOARDS:
            print("ERROR: unknown board '%s' (known: %s)" % (key, ", ".join(sorted(boards.BOARDS))))
            sys.exit(2)
    runs = find_runs(args.results_dir, args.tag)
    if not runs:
        print("ERROR: no MIL result named {architecture}_%s+... in %s" % (args.tag, os.path.join(args.results_dir, "mil")))
        sys.exit(1)
    rows = [gather(args.results_dir, run, args.tag, board_keys) for run in runs]
    # Smallest model first; runs without an INT8 file last.
    rows.sort(key=lambda row: (row.get("tflite_bytes") is None, row.get("tflite_bytes") or 0, row["run"]))

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "benchmark_%s.csv" % args.tag)
    md_path = os.path.join(args.output_dir, "benchmark_%s.md" % args.tag)
    write_csv(csv_path, rows)
    attributed = attribution(args.results_dir, rows, board_keys)
    attribution_path = os.path.join(args.output_dir, "benchmark_%s_attribution.csv" % args.tag)
    if attributed:
        write_attribution_csv(attribution_path, attributed, [row["architecture"] for row in rows
                                                             if any(row["architecture"] in a["same"] for a in attributed)])
    text = markdown(rows, args.tag, board_keys, attributed)
    with open(md_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print(text)
    for row in rows:
        for key in board_keys:
            if row.get("pil_%s_same_file_as_sil" % key) is False:
                print("[WARN] %s: the PIL run on %s names another .tflite than the SIL run" % (row["run"], key))
    wrong = [item for item in attributed if not item["attributed"]]
    for item in wrong:
        print("[WARN] %s on %s agrees more with the SIL of %s (%d images) than with its own (%d): was the board "
              "carrying another model?" % (item["run"], item["board"], item["best_other_architecture"],
                                           item["best_other"], item["own"]))
    print("Wrote %s%s and %s (%d runs)" % (csv_path, ", %s" % attribution_path if attributed else "", md_path, len(rows)))
    if wrong:
        sys.exit(3)


if __name__ == "__main__":
    main()
