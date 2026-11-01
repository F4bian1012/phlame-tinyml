# -*- coding: utf-8 -*-
"""Check docs/results_schema.md against every published results JSON.

For both schemas (metrics_*.json and hil_conditions_*.json):
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
    "latencia.por_campo": "<FIELD>",
    "metricas_por_clase": "<CLASS>",
    "conteo_predicciones": "<CLASS>",
}
METRICS_HEADING = re.compile(r"^## .*`metrics_\*\.json`")
CONDITIONS_HEADING = re.compile(r"^## .*`hil_conditions_\*\.json`")
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


def real_paths(pattern):
    """Return (paths, n_files) for the JSON files matching pattern under results/."""
    files = glob.glob(str(RESULTS_DIR / "**" / pattern), recursive=True)
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
        cls.real_conditions, cls.n_conditions = real_paths("hil_conditions_*.json")

    def check_inputs_found(self):
        self.assertGreater(self.n_metrics, 0, "no metrics_*.json found under results/")
        self.assertGreater(self.n_conditions, 0, "no hil_conditions_*.json found under results/")

    def assert_none(self, keys, message):
        self.assertFalse(keys, "%s: %s" % (message, ", ".join(sorted(keys))))

    def test_every_metrics_key_is_documented(self):
        self.check_inputs_found()
        self.assert_none(self.real_metrics - self.doc_metrics, "metrics keys missing from the docs")

    def test_every_documented_metrics_key_exists(self):
        self.check_inputs_found()
        self.assert_none(self.doc_metrics - self.real_metrics, "documented metrics keys found in no JSON")

    def test_every_conditions_key_is_documented(self):
        self.check_inputs_found()
        self.assert_none(self.real_conditions - self.doc_conditions, "hil_conditions keys missing from the docs")

    def test_every_documented_conditions_key_exists(self):
        self.check_inputs_found()
        self.assert_none(self.doc_conditions - self.real_conditions, "documented hil_conditions keys found in no JSON")


if __name__ == "__main__":
    unittest.main()
