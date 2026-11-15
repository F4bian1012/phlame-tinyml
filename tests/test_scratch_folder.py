#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
results/smoke_test/ is the git-ignored folder the README's own commands write to
(--output_dir results/smoke_test/...). What a user leaves there is not a
published result: the checks of the published results must not read it, or
following the README and then running the tests would fail.

Standard library only.

    python -m unittest discover -s tests -v
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))

import verify_models  # noqa: E402
import test_results_schema  # noqa: E402


def write_metrics(path, model_path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"level": "MIL", "model": {"path": model_path, "sha256": "00"}}), encoding="utf-8")


class ScratchFolderTest(unittest.TestCase):
    def test_model_check_leaves_the_scratch_folder_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_metrics(root / "results" / "mil" / "metrics_A.json", "models/checkpoints/A.keras")
            write_metrics(root / "results" / "nicla" / "pil" / "metrics_B.json", "models/tflite/B_int8.tflite")
            write_metrics(root / "results" / "smoke_test" / "mil" / "metrics_Mine.json", "models/checkpoints/Mine.keras")
            write_metrics(root / "results" / "smoke_test" / "metrics_Top.json", "models/checkpoints/Top.keras")
            found = sorted(ref[0] for ref in verify_models.result_references(str(root)))
            self.assertEqual(found, ["results/mil/metrics_A.json", "results/nicla/pil/metrics_B.json"])

    def test_schema_check_leaves_the_scratch_folder_out(self):
        scratch = test_results_schema.SCRATCH_DIR
        self.assertEqual(scratch, REPO / "results" / "smoke_test")
        ignored = (REPO / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("results/smoke_test/", ignored)
        # whatever is in the scratch folder now, the schema check counts the published files only:
        # none of the scratch folder, and (in a git checkout) none that git does not track, such as
        # the results of a session that is being measured
        _, n_files = test_results_schema.real_paths("metrics_*.json")
        on_disk = [str(p) for p in (REPO / "results").rglob("metrics_*.json")]
        published = test_results_schema.published(on_disk)
        self.assertEqual(n_files, len(published))
        self.assertEqual([p for p in published if scratch in Path(p).parents], [])
        self.assertLessEqual(len(published), len([p for p in on_disk if scratch not in Path(p).parents]))


if __name__ == "__main__":
    unittest.main()
