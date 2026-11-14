#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_models.py - Checks the model files against the published checksums.

The trained models (.keras) and their INT8 versions (.tflite) are too large
for the git tree. models/checksums.sha256 lists the SHA-256 of every model
file behind a published result, so a file obtained from a release, or from
anywhere else, can be checked before it is evaluated or flashed.

    # Check the model files present under models/
    python src/verify_models.py

    # Also fail when a listed file is missing
    python src/verify_models.py --require_all

    # Check that every metrics_*.json names a listed model with the listed
    # digest (reads the result files only; no model file is needed)
    python src/verify_models.py --results

    # Fetch the model archives attached to a release, unpack them and check them
    python src/verify_models.py --download v1.0.0 --set coco

    # Maintainer: build the archives to attach to a release
    python src/verify_models.py --bundle release_assets --tag v1.0.0

The models are grouped in sets by the run tag in their name: "coco"
({arch}_COCO+...), "vww" ({arch}_VWW+...) and "plates" (no tag). A release
carries one archive per set, phlame-models-<set>-<tag>.zip, holding the files
at their paths under models/.

Exit codes: 0 everything checked agrees with the list; 1 a file or a result
disagrees with it, a required file is missing or a download failed; 2 the
list or the command line cannot be used.

Standard library only.
"""
import glob
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile

import cli

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join("models", "checksums.sha256")
RELEASES_URL = "https://github.com/F4bian1012/phlame-tinyml/releases/download"
SETS = ("coco", "vww", "plates")
ENTRY = re.compile(r"^([0-9a-f]{64}) [ *](models/(?:checkpoints/[^/]+\.keras|tflite/[^/]+\.tflite))$")
# Fixed date of the archive members, so the same files give the same archive.
ZIP_DATE = (2026, 1, 1, 0, 0, 0)


def fail(code, message):
    print("ERROR: " + message)
    sys.exit(code)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def set_of(rel_path):
    """Set a model belongs to, from the run tag after the architecture."""
    stem = rel_path.rsplit("/", 1)[-1].split("+", 1)[0]
    if stem.endswith("_COCO"):
        return "coco"
    if stem.endswith("_VWW"):
        return "vww"
    return "plates"


def read_manifest(path):
    """{relative path: sha256} in file order. Stops with exit code 2 when the
    list is missing or has a line that is neither a comment nor an entry."""
    if not os.path.isfile(path):
        fail(2, "checksum list not found: %s" % path)
    entries = {}
    with open(path, encoding="utf-8") as fh:
        for number, line in enumerate(fh, start=1):
            line = line.rstrip("\r\n")
            if not line.strip() or line.startswith("#"):
                continue
            match = ENTRY.match(line)
            if not match:
                fail(2, "%s, line %d is not '<sha256>  models/...': %r" % (path, number, line[:80]))
            digest, rel = match.groups()
            if rel in entries:
                fail(2, "%s lists %s twice" % (path, rel))
            entries[rel] = digest
    if not entries:
        fail(2, "%s lists no model" % path)
    return entries


def select(entries, which):
    return {rel: d for rel, d in entries.items() if which == "all" or set_of(rel) == which}


def verify_files(entries, root, require_all):
    """Compares the files under root with the list. Returns the number of problems."""
    ok = missing = bad = 0
    for rel, digest in entries.items():
        path = os.path.join(root, rel)
        if not os.path.isfile(path):
            missing += 1
            if require_all:
                print("MISSING   %s" % rel)
            continue
        if sha256_file(path) == digest:
            ok += 1
        else:
            bad += 1
            print("MISMATCH  %s  (this is not the file the published results were obtained with)" % rel)
    print("Model files: %d listed, %d present and correct, %d different, %d not present."
          % (len(entries), ok, bad, missing))
    return bad + (missing if require_all else 0)


def manifest_path_of(model_path):
    """models/<folder>/<name> entry that a model path of a result file refers to."""
    name = model_path.replace("\\", "/").rsplit("/", 1)[-1]
    return "models/%s/%s" % ("checkpoints" if name.endswith(".keras") else "tflite", name)


def result_references(root):
    """(result file, field, model path, sha256) for every published result."""
    refs = []
    results = os.path.join(root, "results")
    for path in sorted(glob.glob(os.path.join(results, "**", "metrics_*.json"), recursive=True)):
        with open(path, encoding="utf-8") as fh:
            model = json.load(fh).get("model") or {}
        refs.append((os.path.relpath(path, root).replace("\\", "/"), "model.sha256",
                     model.get("path"), model.get("sha256")))
    for path in sorted(glob.glob(os.path.join(results, "**", "conditions_*.json"), recursive=True)):
        with open(path, encoding="utf-8") as fh:
            expected = json.load(fh).get("expected_model") or {}
        if expected:
            refs.append((os.path.relpath(path, root).replace("\\", "/"), "expected_model.sha256",
                         expected.get("path"), expected.get("sha256")))
    return refs


def verify_results(entries, root):
    """Compares the digests recorded in the result files with the list.
    Returns the number of problems."""
    ok = skipped = bad = 0
    for rel_json, field, model_path, digest in result_references(root):
        if not model_path or not digest:
            skipped += 1          # a run that did not record its model file
            continue
        rel = manifest_path_of(model_path)
        if rel not in entries:
            bad += 1
            print("NOT LISTED  %s  (named by %s)" % (rel, rel_json))
        elif entries[rel] != digest:
            bad += 1
            print("MISMATCH    %s  (%s of %s records another digest)" % (rel, field, rel_json))
        else:
            ok += 1
    print("Result files: %d references agree with the list, %d disagree, %d record no model file."
          % (ok, bad, skipped))
    return bad


def archive_name(which, tag):
    return "phlame-models-%s-%s.zip" % (which, tag)


def build_bundle(entries, root, out_dir, tag, manifest_file):
    """Writes one archive per set plus SHA256SUMS.txt. Every listed file of the
    set must be present and correct: an archive with a wrong or missing model
    would publish a file no result was obtained with."""
    os.makedirs(out_dir, exist_ok=True)
    sums = []
    for which in SETS:
        chosen = select(entries, which)
        if not chosen:
            continue
        if verify_files(chosen, root, require_all=True):
            fail(1, "set '%s' is incomplete or differs from the list; no archive was written for it." % which)
        target = os.path.join(out_dir, archive_name(which, tag))
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            members = [(MANIFEST.replace(os.sep, "/"), manifest_file)]
            members += [(rel, os.path.join(root, rel)) for rel in sorted(chosen)]
            for rel, path in members:
                info = zipfile.ZipInfo(rel, date_time=ZIP_DATE)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                with open(path, "rb") as src, archive.open(info, "w") as dst:
                    for block in iter(lambda: src.read(1 << 20), b""):
                        dst.write(block)
        sums.append((sha256_file(target), os.path.basename(target)))
        print("Wrote %s (%d models, %.1f MiB)" % (target, len(chosen), os.path.getsize(target) / 2 ** 20))
    with io.open(os.path.join(out_dir, "SHA256SUMS.txt"), "w", encoding="utf-8", newline="\n") as fh:
        for digest, name in sums:
            fh.write("%s  %s\n" % (digest, name))
    print("Wrote %s" % os.path.join(out_dir, "SHA256SUMS.txt"))


def download(entries, root, tag, which, base_url, overwrite):
    """Fetches the archive of each requested set and unpacks the listed models.
    Only members named in the list are written, at their listed path, and a
    model that is already there is kept unless it differs and --overwrite is
    given. Returns the number of problems."""
    problems = 0
    base = (base_url or "%s/%s" % (RELEASES_URL, tag)).rstrip("/")
    for name in (SETS if which == "all" else (which,)):
        chosen = select(entries, name)
        if not chosen:
            continue
        url = "%s/%s" % (base, archive_name(name, tag))
        print("Downloading %s" % url)
        with tempfile.TemporaryDirectory() as tmp:
            local = os.path.join(tmp, archive_name(name, tag))
            try:
                with urllib.request.urlopen(url) as response, open(local, "wb") as fh:
                    shutil.copyfileobj(response, fh)
                archive = zipfile.ZipFile(local)
            except (urllib.error.URLError, OSError) as exc:
                print("FAILED    %s (%s). Is release %s published and does it carry this archive?"
                      % (url, exc, tag))
                problems += 1
                continue
            except zipfile.BadZipFile:
                print("FAILED    %s is not a zip archive" % url)
                problems += 1
                continue
            with archive:
                for member in archive.namelist():
                    if member not in chosen:
                        continue      # the checksum list inside the archive, or anything unexpected
                    problems += unpack_member(archive, member, chosen[member], root, overwrite)
        problems += verify_files(chosen, root, require_all=True)
    return problems


def unpack_member(archive, member, digest, root, overwrite):
    """Writes one model of the archive at its listed path, unless the file is
    already there. The content is checked while it is copied, and a file that
    does not match the list is never left in place. Returns 1 on a problem."""
    target = os.path.join(root, member)
    if os.path.isfile(target):
        if sha256_file(target) == digest:
            return 0
        if not overwrite:
            print("KEPT      %s exists and differs from the list; pass --overwrite to replace it" % member)
            return 1
    os.makedirs(os.path.dirname(target), exist_ok=True)
    part = target + ".part"
    h = hashlib.sha256()
    with archive.open(member) as src, open(part, "wb") as dst:
        for block in iter(lambda: src.read(1 << 20), b""):
            h.update(block)
            dst.write(block)
    if h.hexdigest() != digest:
        os.remove(part)
        print("MISMATCH  %s inside the archive differs from the list; not written" % member)
        return 1
    os.replace(part, target)
    return 0


def parse_args():
    parser = cli.ArgumentParser(
        description="Check the model files against models/checksums.sha256, "
                    "fetch them from a release, or build the archives of a release.")
    parser.add_argument("--set", dest="which", choices=("all",) + SETS, default="all",
                        help="Models to act on: coco, vww, plates or all (default: all)")
    parser.add_argument("--require_all", action="store_true",
                        help="Fail when a listed model file is missing (default: only the files present are checked)")
    parser.add_argument("--results", action="store_true",
                        help="Check the digests recorded in results/**/metrics_*.json and conditions_*.json "
                             "against the list instead of checking model files")
    parser.add_argument("--download", metavar="TAG", default=None,
                        help="Fetch the model archives attached to this release (e.g. v1.0.0), unpack and check them")
    parser.add_argument("--base_url", default=None,
                        help="Where the archives are, for --download (default: the release page of TAG on GitHub)")
    parser.add_argument("--overwrite", action="store_true",
                        help="With --download: replace a model file that exists and differs from the list")
    parser.add_argument("--bundle", metavar="DIR", default=None,
                        help="Maintainer: write phlame-models-<set>-<tag>.zip and SHA256SUMS.txt into DIR")
    parser.add_argument("--tag", default=None,
                        help="Release tag used in the archive names, for --bundle (e.g. v1.0.0)")
    parser.add_argument("--root", default=REPO_ROOT,
                        help="Folder that holds models/ and results/ (default: the repository)")
    parser.add_argument("--manifest", default=None,
                        help="Checksum list (default: <root>/models/checksums.sha256)")
    args = parser.parse_args()
    if args.bundle and not args.tag:
        parser.error("--bundle needs --tag (it names the archives)")
    if sum(bool(x) for x in (args.results, args.download, args.bundle)) > 1:
        parser.error("--results, --download and --bundle are separate actions; give one")
    return args


def main():
    args = parse_args()
    manifest_file = args.manifest or os.path.join(args.root, MANIFEST)
    entries = read_manifest(manifest_file)
    chosen = select(entries, args.which)
    if not chosen:
        fail(2, "the list has no model in set '%s'" % args.which)

    if args.results:
        problems = verify_results(entries, args.root)
    elif args.download:
        problems = download(entries, args.root, args.download, args.which, args.base_url, args.overwrite)
    elif args.bundle:
        build_bundle(chosen, args.root, args.bundle, args.tag, manifest_file)
        problems = 0
    else:
        problems = verify_files(chosen, args.root, args.require_all)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
