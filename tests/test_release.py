#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Checks what a release rests on: the citation metadata (CITATION.cff,
.zenodo.json and the README must tell the same story, and a DOI appears in
all of them or in none), the checksum list of the model files
(models/checksums.sha256) against the published results, and
src/verify_models.py on made-up model files, including a download from a
local stand-in for the release page.

Standard library only (no network, no model file needed).

    python -m unittest discover -s tests -v
"""
import contextlib
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import verify_models  # noqa: E402

SCRIPT = str(REPO / "src" / "verify_models.py")
DOI = re.compile(r"10\.\d{4,9}/[^\s\"'<>)}\]]+")


def cff_scalar(text, key):
    match = re.search(r'^%s:\s*"?(.*?)"?\s*$' % re.escape(key), text, re.M)
    return match.group(1) if match else None


def cff_list(text, key):
    block = re.search(r"^%s:\n((?:  - .*\n)+)" % re.escape(key), text, re.M)
    return [line[4:].strip().strip('"') for line in block.group(1).splitlines()] if block else []


class CitationMetadataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cff = (REPO / "CITATION.cff").read_text(encoding="utf-8")
        cls.zenodo = json.loads((REPO / ".zenodo.json").read_text(encoding="utf-8"))
        cls.readme = (REPO / "README.md").read_text(encoding="utf-8")

    def test_zenodo_metadata_matches_the_citation_file(self):
        # Zenodo reads .zenodo.json and ignores CITATION.cff, so the two must agree.
        self.assertEqual(self.zenodo["title"], cff_scalar(self.cff, "title"))
        self.assertEqual(self.zenodo["license"].lower(), cff_scalar(self.cff, "license").lower())
        self.assertEqual(self.zenodo["keywords"], cff_list(self.cff, "keywords"))
        self.assertEqual(self.zenodo["upload_type"], "software")
        names = re.findall(r'family-names:\s*"(.*?)"\s*\n\s*given-names:\s*"(.*?)"', self.cff)
        self.assertEqual([c["name"] for c in self.zenodo["creators"]], ["%s, %s" % pair for pair in names])
        orcids = re.findall(r'orcid:\s*"https://orcid\.org/(.*?)"', self.cff)
        self.assertEqual([c["orcid"] for c in self.zenodo["creators"]], orcids)
        affiliations = re.findall(r'affiliation:\s*"(.*?)"', self.cff)
        self.assertEqual([c["affiliation"] for c in self.zenodo["creators"]], affiliations)
        repository = cff_scalar(self.cff, "repository-code")
        self.assertIn(repository, [r["identifier"] for r in self.zenodo["related_identifiers"]])

    def test_a_doi_is_stated_everywhere_or_nowhere(self):
        declared = cff_scalar(self.cff, "doi")
        in_readme = set(DOI.findall(self.readme))
        if declared is None:
            self.assertEqual(in_readme, set(), "the README gives a DOI that CITATION.cff does not declare")
            self.assertNotIn("zenodo.org/badge", self.readme)
        else:
            self.assertIn(declared, in_readme, "CITATION.cff declares a DOI the README does not give")

    def test_version_is_the_same_wherever_it_is_declared(self):
        version = cff_scalar(self.cff, "version")
        self.assertRegex(version, r"^\d+\.\d+\.\d+$")
        for declared in re.findall(r"version\s*=\s*\{(.*?)\}", self.readme):
            self.assertEqual(declared, version)
        if "version" in self.zenodo:
            self.assertEqual(self.zenodo["version"], version)

    def test_a_release_date_needs_a_release(self):
        # date-released is written on the day the tag is created, never before.
        released = cff_scalar(self.cff, "date-released")
        if released is not None:
            self.assertRegex(released, r"^\d{4}-\d{2}-\d{2}$")
            self.assertNotIn("has not been published yet", self.readme)


class ChecksumListTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.entries = verify_models.read_manifest(str(REPO / "models" / "checksums.sha256"))

    def test_list_is_sorted_and_names_only_model_files(self):
        paths = list(self.entries)
        self.assertEqual(paths, sorted(paths))
        for rel in paths:
            self.assertIn(verify_models.set_of(rel), verify_models.SETS)

    def test_every_published_result_used_a_listed_model(self):
        with contextlib.redirect_stdout(io.StringIO()) as log:
            problems = verify_models.verify_results(self.entries, str(REPO))
        self.assertEqual(problems, 0, log.getvalue())
        self.assertRegex(log.getvalue(), r"Result files: [1-9]\d* references agree")

    def test_model_files_are_kept_out_of_git(self):
        tracked = subprocess.run(["git", "ls-files", "*.keras", "*.tflite"], cwd=REPO,
                                 stdout=subprocess.PIPE, text=True).stdout.split()
        self.assertEqual(tracked, [])


def run(*args):
    return subprocess.run([sys.executable, SCRIPT, *args], stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)


class VerifyModelsScriptTest(unittest.TestCase):
    """Runs the script on a made-up repository: three small 'models' in the two published sets."""

    FILES = {
        "models/checkpoints/Tiny_COCO+1+1+0.1+8+8.keras": b"coco float model",
        "models/tflite/Tiny_COCO+1+1+0.1+8+8_int8.tflite": b"coco int8 model",
        "models/tflite/Tiny_VWW+1+1+0.1+8+8_int8.tflite": b"vww int8 model",
    }

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = os.path.join(self.tmp.name, "repo")
        self.make_root(self.root, with_models=True)

    def make_root(self, root, with_models):
        os.makedirs(os.path.join(root, "models"))
        lines = ["# made-up list"]
        for rel in sorted(self.FILES):
            lines.append("%s  %s" % (hashlib.sha256(self.FILES[rel]).hexdigest(), rel))
            if with_models:
                os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
                Path(root, rel).write_bytes(self.FILES[rel])
        Path(root, "models", "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_correct_files_pass_and_a_changed_file_fails(self):
        self.assertEqual(run("--root", self.root).returncode, 0)
        Path(self.root, "models/tflite/Tiny_VWW+1+1+0.1+8+8_int8.tflite").write_bytes(b"another model")
        changed = run("--root", self.root)
        self.assertEqual(changed.returncode, 1)
        self.assertIn("MISMATCH  models/tflite/Tiny_VWW+1+1+0.1+8+8_int8.tflite", changed.stdout)
        self.assertEqual(run("--root", self.root, "--set", "coco").returncode, 0)

    def test_a_missing_file_fails_only_when_all_are_required(self):
        os.remove(os.path.join(self.root, "models/tflite/Tiny_VWW+1+1+0.1+8+8_int8.tflite"))
        self.assertEqual(run("--root", self.root).returncode, 0)
        required = run("--root", self.root, "--require_all")
        self.assertEqual(required.returncode, 1)
        self.assertIn("MISSING   models/tflite/Tiny_VWW+1+1+0.1+8+8_int8.tflite", required.stdout)
        # the other set is complete
        self.assertEqual(run("--root", self.root, "--set", "coco", "--require_all").returncode, 0)

    def test_a_list_that_cannot_be_read_is_exit_code_2(self):
        Path(self.root, "models", "checksums.sha256").write_text("not a checksum line\n", encoding="utf-8")
        self.assertEqual(run("--root", self.root).returncode, 2)
        self.assertEqual(run("--root", os.path.join(self.tmp.name, "nowhere")).returncode, 2)
        self.assertEqual(run("--root", self.root, "--bundle", "x").returncode, 2)      # --tag is missing

    def test_results_are_checked_against_the_list(self):
        folder = os.path.join(self.root, "results", "sil")
        os.makedirs(folder)
        rel = "models/tflite/Tiny_COCO+1+1+0.1+8+8_int8.tflite"
        metrics = {"model": {"path": rel.replace("/", "\\"), "sha256": hashlib.sha256(self.FILES[rel]).hexdigest()}}
        Path(folder, "metrics_a.json").write_text(json.dumps(metrics), encoding="utf-8")
        self.assertEqual(run("--root", self.root, "--results").returncode, 0)
        metrics["model"]["sha256"] = "0" * 64
        Path(folder, "metrics_b.json").write_text(json.dumps(metrics), encoding="utf-8")
        metrics["model"]["path"] = "models/tflite/Unknown+1+1+0.1+8+8_int8.tflite"
        Path(folder, "metrics_c.json").write_text(json.dumps(metrics), encoding="utf-8")
        wrong = run("--root", self.root, "--results")
        self.assertEqual(wrong.returncode, 1)
        self.assertIn("MISMATCH", wrong.stdout)
        self.assertIn("NOT LISTED  models/tflite/Unknown+1+1+0.1+8+8_int8.tflite", wrong.stdout)

    def test_bundle_then_download_round_trip(self):
        # the records the scripts write next to a model, as a training machine leaves them
        record = {"script": "train_x.py", "splits_dir": "/home/someone/data_full", "epochs_run": 1,
                  "settings": {"weights": "imagenet"}, "software": {"gpus": ["/physical_device:GPU:0"]},
                  "history": {"loss": [0.5]}}
        Path(self.root, "models/checkpoints/Tiny_COCO+1+1+0.1+8+8.training.json").write_text(
            json.dumps(record), encoding="utf-8")
        Path(self.root, "models/checkpoints/Tiny_COCO+1+1+0.1+8+8.classes.txt").write_bytes(b"cat\ndog\n")
        Path(self.root, "models/tflite/Tiny_COCO+1+1+0.1+8+8_int8.classes.txt").write_bytes(b"cat\ndog\n")
        Path(self.root, "models/checkpoints/Other_COCO+1+1+0.1+8+8.training.json").write_text("{}", encoding="utf-8")

        assets = os.path.join(self.tmp.name, "assets")
        built = run("--root", self.root, "--bundle", assets, "--tag", "v9.9.9")
        self.assertEqual(built.returncode, 0, built.stdout)
        self.assertEqual(sorted(os.listdir(assets)),
                         ["SHA256SUMS.txt", "phlame-models-coco-v9.9.9.zip", "phlame-models-vww-v9.9.9.zip"])
        with zipfile.ZipFile(os.path.join(assets, "phlame-models-coco-v9.9.9.zip")) as archive:
            # the listed models, the list, and the records of those models only
            self.assertEqual(sorted(archive.namelist()),
                             ["models/checkpoints/Tiny_COCO+1+1+0.1+8+8.classes.txt",
                              "models/checkpoints/Tiny_COCO+1+1+0.1+8+8.keras",
                              "models/checkpoints/Tiny_COCO+1+1+0.1+8+8.training.json", "models/checksums.sha256",
                              "models/tflite/Tiny_COCO+1+1+0.1+8+8_int8.classes.txt",
                              "models/tflite/Tiny_COCO+1+1+0.1+8+8_int8.tflite"])
            published = json.loads(archive.read("models/checkpoints/Tiny_COCO+1+1+0.1+8+8.training.json"))
            # the path of the training machine is reduced to its last folder; nothing else changes
            self.assertEqual(published, dict(record, splits_dir="data_full"))
            for name in archive.namelist():
                self.assertNotIn(b"/home/someone", archive.read(name), name)
        with zipfile.ZipFile(os.path.join(assets, "phlame-models-vww-v9.9.9.zip")) as archive:
            self.assertEqual([n for n in archive.namelist() if n.endswith((".json", ".txt"))], [])
        # the record on disk is left as it was
        self.assertEqual(json.loads(Path(self.root, "models/checkpoints/Tiny_COCO+1+1+0.1+8+8.training.json")
                                    .read_text(encoding="utf-8")), record)
        for line in Path(assets, "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
            digest, name = line.split("  ")
            self.assertEqual(hashlib.sha256(Path(assets, name).read_bytes()).hexdigest(), digest)

        # a reader's clone: the list is there, the models are not
        clone = os.path.join(self.tmp.name, "clone")
        self.make_root(clone, with_models=False)
        base_url = Path(assets).as_uri()
        fetched = run("--root", clone, "--download", "v9.9.9", "--base_url", base_url, "--set", "coco")
        self.assertEqual(fetched.returncode, 0, fetched.stdout)
        self.assertEqual(Path(clone, "models/tflite/Tiny_COCO+1+1+0.1+8+8_int8.tflite").read_bytes(),
                         b"coco int8 model")
        # the records come with the models
        self.assertEqual(Path(clone, "models/tflite/Tiny_COCO+1+1+0.1+8+8_int8.classes.txt").read_bytes(),
                         b"cat\ndog\n")
        self.assertEqual(json.loads(Path(clone, "models/checkpoints/Tiny_COCO+1+1+0.1+8+8.training.json")
                                    .read_text(encoding="utf-8"))["splits_dir"], "data_full")
        self.assertFalse(os.path.exists(os.path.join(clone, "models/tflite/Tiny_VWW+1+1+0.1+8+8_int8.tflite")))
        self.assertEqual(run("--root", clone, "--set", "coco", "--require_all").returncode, 0)

        # a model already there and different is kept unless --overwrite is given
        own = Path(clone, "models/tflite/Tiny_COCO+1+1+0.1+8+8_int8.tflite")
        own.write_bytes(b"my retrained model")
        kept = run("--root", clone, "--download", "v9.9.9", "--base_url", base_url, "--set", "coco")
        self.assertEqual(kept.returncode, 1)
        self.assertEqual(own.read_bytes(), b"my retrained model")
        replaced = run("--root", clone, "--download", "v9.9.9", "--base_url", base_url, "--set", "coco",
                       "--overwrite")
        self.assertEqual(replaced.returncode, 0, replaced.stdout)
        self.assertEqual(own.read_bytes(), b"coco int8 model")

    def test_download_writes_nothing_the_list_does_not_name(self):
        assets = os.path.join(self.tmp.name, "assets")
        os.makedirs(assets)
        rel = "models/tflite/Tiny_VWW+1+1+0.1+8+8_int8.tflite"
        with zipfile.ZipFile(os.path.join(assets, "phlame-models-vww-v9.9.9.zip"), "w") as archive:
            archive.writestr(rel, b"tampered model")
            archive.writestr("../outside.txt", b"should never be written")
            archive.writestr("src/cli.py", b"should never be written")
        clone = os.path.join(self.tmp.name, "clone")
        self.make_root(clone, with_models=False)
        result = run("--root", clone, "--download", "v9.9.9", "--base_url", Path(assets).as_uri(), "--set", "vww")
        self.assertEqual(result.returncode, 1)
        self.assertIn("MISMATCH", result.stdout)
        written = sorted(str(p.relative_to(self.tmp.name)).replace("\\", "/")
                         for p in Path(self.tmp.name).rglob("*") if p.is_file() and "assets" not in p.parts)
        self.assertEqual([p for p in written if p.startswith("clone/")], ["clone/models/checksums.sha256"])
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "outside.txt")))

    def test_a_release_that_is_not_there_is_reported(self):
        clone = os.path.join(self.tmp.name, "clone")
        self.make_root(clone, with_models=False)
        missing = run("--root", clone, "--download", "v9.9.9", "--set", "vww",
                      "--base_url", Path(self.tmp.name, "no_such_release").as_uri())
        self.assertEqual(missing.returncode, 1)
        self.assertIn("FAILED", missing.stdout)


if __name__ == "__main__":
    unittest.main()
