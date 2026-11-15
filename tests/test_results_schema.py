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
# Keys the HIL and PIL benches write that no published file carries yet: they
# gained them in November 2026 and nothing has been measured with them since.
# They are documented from what the benches write, which tests/test_hil_session.py
# and tests/test_pil_session.py check against a simulated board. When a published
# file has one of them it must leave this list: the last test of this file fails
# until it does.
NOT_YET_PUBLISHED_METRICS = {"run.completed", "dataset.n_stimuli_planned"}              # HIL
NOT_YET_PUBLISHED_PIL_METRICS = {"run.completed", "run.stopped_because", "run.input_shape_verified",
                                 "run.max_unanswered", "dataset.n_images_planned", "model.arena_used_bytes"}
# The 'software' block (versions behind a result), which the four levels write and
# tests/test_software_versions.py checks. Each list holds the keys of its own bench; the
# keys MIL and SIL write are all among them.
NOT_YET_PUBLISHED_METRICS |= {"software", "software.python", "software.platform", "software.numpy",
                              "software.scikit_learn", "software.pyserial", "software.opencv",
                              "software.tensorflow", "software.keras", "software.gpus"}            # HIL
NOT_YET_PUBLISHED_PIL_METRICS |= {"software", "software.python", "software.platform", "software.numpy",
                                  "software.scikit_learn", "software.pyserial", "software.pillow"}
NOT_YET_PUBLISHED_CONDITIONS = {"n_stimuli_planned", "completed", "stopped_because", "stimulus_monitor",
                                "stimulus_monitor.index", "stimulus_monitor.geometry",
                                "stimulus_monitor.monitors"}
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


def real_paths(pattern):
    """Return (paths, n_files) for the published JSON files matching pattern under results/."""
    files = [name for name in glob.glob(str(RESULTS_DIR / "**" / pattern), recursive=True)
             if SCRATCH_DIR not in Path(name).parents]
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
