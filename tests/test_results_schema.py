# -*- coding: utf-8 -*-
"""Check docs/results_schema.md against every published results JSON.

For both schemas (metrics_*.json and conditions_*.json):
  (a) every key found in a JSON must be documented;
  (b) every documented key must appear in at least one JSON.

Run from anywhere with: python -m unittest discover -s tests -v
"""

import glob
import json
import re
import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DOC = REPO_ROOT / "docs" / "results_schema.md"
RESULTS_DIR = REPO_ROOT / "results"

# Parents whose direct children are dynamic, and the placeholder that replaces them.
DYNAMIC_CHILDREN = {
    "latency.per_field": "<FIELD>",
    "per_class_metrics": "<CLASS>",
    "prediction_counts": "<CLASS>",
}
# Keys a bench writes that no published file carries yet. A key documented from
# what a bench writes (which tests/test_hil_session.py and tests/test_pil_session.py
# check against a simulated board) goes in one of these lists until a published
# file has it; then it must leave the list: the last test of this file fails until
# it does. The lists are empty since 2026-11-15: the keys the HIL bench gained in
# November were published with the COCO HIL sessions, and those of the PIL bench
# with the runs of results/experiments/.
NOT_YET_PUBLISHED_METRICS = set()                                                        # HIL
NOT_YET_PUBLISHED_PIL_METRICS = set()
NOT_YET_PUBLISHED_CONDITIONS = set()
METRICS_HEADING = re.compile(r"^## .*`metrics_\*\.json`")
CONDITIONS_HEADING = re.compile(r"^## .*`conditions_\*\.json`")
KEY_ROW = re.compile(r"^\|\s*`([^`]+)`\s*\|")


def documented_paths():
    """Return (metrics_paths, conditions_paths) from the first column of the tables."""
    metrics, conditions = set(), set()
    current = None
    for line in SCHEMA_DOC.read_text(encoding="utf-8").splitlines():
        if METRICS_HEADING.match(line):
            current = metrics
        elif CONDITIONS_HEADING.match(line):
            current = conditions
        elif current is not None:
            row = KEY_ROW.match(line)
            if row:
                current.add(row.group(1))
    return metrics, conditions


def collect_paths(node, prefix, paths):
    """Add every dotted key path of a JSON object to paths. Lists are not descended."""
    for key, value in node.items():
        if prefix in DYNAMIC_CHILDREN:
            path = "%s.%s" % (prefix, DYNAMIC_CHILDREN[prefix])
        elif prefix:
            path = "%s.%s" % (prefix, key)
        else:
            path = key
        paths.add(path)
        if isinstance(value, dict):
            collect_paths(value, path, paths)


# Git-ignored scratch folder where the README's own commands write: its files are not published results.
SCRATCH_DIR = RESULTS_DIR / "smoke_test"


def published(files):
    """The published result files: those git tracks, so that runs of a user's own models (which the
    usage steps of the README write next to the published ones) are not held to this check. Without
    git, every file outside the scratch folder."""
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), "ls-files", "-z", "--", "results"],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True).stdout
        tracked = {REPO_ROOT / name for name in out.decode("utf-8", "replace").split("\0") if name}
    except (OSError, subprocess.CalledProcessError):
        tracked = set()
    if tracked:
        return [name for name in files if Path(name) in tracked]
    return [name for name in files if SCRATCH_DIR not in Path(name).parents]


def real_paths(pattern):
    """Return (paths, n_files) for the published JSON files matching pattern under results/."""
    files = published(glob.glob(str(RESULTS_DIR / "**" / pattern), recursive=True))
    paths = set()
    for name in files:
        with open(name, encoding="utf-8") as fh:
            collect_paths(json.load(fh), "", paths)
    return paths, len(files)


class ResultsSchemaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc_metrics, cls.doc_conditions = documented_paths()
        cls.real_metrics, cls.n_metrics = real_paths("metrics_*.json")
        cls.real_conditions, cls.n_conditions = real_paths("conditions_*.json")

    def check_inputs_found(self):
        self.assertGreater(self.n_metrics, 0, "no metrics_*.json found under results/")
        self.assertGreater(self.n_conditions, 0, "no conditions_*.json found under results/")

    def assert_none(self, keys, message):
        self.assertFalse(keys, "%s: %s" % (message, ", ".join(sorted(keys))))

    def test_every_metrics_key_is_documented(self):
        self.check_inputs_found()
        self.assert_none(self.real_metrics - self.doc_metrics, "metrics keys missing from the docs")

    def test_every_documented_metrics_key_exists(self):
        self.check_inputs_found()
        self.assert_none(self.doc_metrics - self.real_metrics - NOT_YET_PUBLISHED_METRICS
                         - NOT_YET_PUBLISHED_PIL_METRICS,
                         "documented metrics keys found in no JSON")

    def test_every_conditions_key_is_documented(self):
        self.check_inputs_found()
        self.assert_none(self.real_conditions - self.doc_conditions, "conditions keys missing from the docs")

    def test_every_documented_conditions_key_exists(self):
        self.check_inputs_found()
        self.assert_none(self.doc_conditions - self.real_conditions - NOT_YET_PUBLISHED_CONDITIONS,
                         "documented conditions keys found in no JSON")

    def test_keys_listed_as_not_yet_published_are_documented_and_still_unpublished(self):
        self.check_inputs_found()
        for listed, documented, published in (
                (NOT_YET_PUBLISHED_METRICS | NOT_YET_PUBLISHED_PIL_METRICS, self.doc_metrics, self.real_metrics),
                (NOT_YET_PUBLISHED_CONDITIONS, self.doc_conditions, self.real_conditions)):
            self.assert_none(listed - documented, "listed as not yet published but not documented")
            self.assert_none(listed & published,
                             "now in a published file: remove from the NOT_YET_PUBLISHED list of this test")


if __name__ == "__main__":
    unittest.main()
