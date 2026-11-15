#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Checks the 'software' block the four levels write to their metrics_*.json:
the versions of Python, of the operating system and of the packages that can
change a figure (src/artifacts.py, software_versions()).

  - the helper: which version it reports for a package that is running, one
    that is only installed and one that is missing; that it imports nothing
    (the board benches must keep running without TensorFlow); that the block
    carries versions and no path;
  - PIL and HIL against a simulated board: the block holds the packages of the
    host side, with the versions of the modules that ran;
  - MIL and SIL on a small model: the block holds the versions a separate
    Python process reads from the same packages.

Expected values come from the packages themselves (their __version__), not
from the helper. TensorFlow only runs in child processes, so this file does
not load it into the process of the other tests; the MIL and SIL part is
skipped when TensorFlow is not installed, the bench parts when the packages of
the benches are missing.

    python -m unittest discover -s tests -v
"""
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("MPLBACKEND", "Agg")   # the benches draw the matrix; no window

import artifacts  # noqa: E402
import test_results_schema as schema  # noqa: E402
from test_boards import PilFolderCase, pil_benchmark  # noqa: E402

HAVE_METRICS = all(importlib.util.find_spec(name) for name in ("sklearn", "seaborn", "matplotlib"))
PIL_READY = pil_benchmark is not None and HAVE_METRICS
HAVE_TF = importlib.util.find_spec("tensorflow") is not None

# A version as packages write it: digits and dots first, then whatever follows (rc1, +cpu, .93).
VERSION = re.compile(r"^\d+(\.\d+)*[A-Za-z0-9.+_-]*$")
PACKAGE_MODULES = ("tensorflow", "keras", "numpy", "sklearn", "PIL", "serial", "cv2")
W, H = 16, 12
ENV = dict(os.environ, TF_CPP_MIN_LOG_LEVEL="3", PYTHONIOENCODING="utf-8")


def documented(block):
    """Key paths of a software block that docs/results_schema.md does not list."""
    doc_metrics, _ = schema.documented_paths()
    return sorted(set(["software"] + ["software.%s" % key for key in block]) - doc_metrics)


def assert_versions_only(case, block):
    """Every value is a version, the platform string, a list of device names or None: no path."""
    case.assertEqual(block["python"], platform.python_version())
    case.assertEqual(block["platform"], platform.platform())
    for key, value in block.items():
        if key in ("platform", "gpus") or value is None:
            continue
        case.assertRegex(value, VERSION, key)
    text = json.dumps(block)
    for forbidden in (sys.executable, os.path.expanduser("~"), os.getcwd()):
        case.assertNotIn(forbidden, text)
        case.assertNotIn(forbidden.replace("\\", "\\\\"), text)


class HelperTest(unittest.TestCase):
    def test_block_has_python_the_platform_and_the_packages_asked_for(self):
        block = artifacts.software_versions(["numpy", "pillow"])
        self.assertEqual(list(block), ["python", "numpy", "pillow", "platform"])
        assert_versions_only(self, block)

    def test_version_of_an_installed_package_is_the_one_its_module_gives(self):
        import numpy
        import PIL
        block = artifacts.software_versions(["numpy", "pillow"])
        self.assertEqual((block["numpy"], block["pillow"]), (numpy.__version__, PIL.__version__))

    def test_the_running_module_is_asked_before_the_installed_distribution(self):
        running = types.ModuleType("phlame_fake_running_module")
        running.__version__ = "9.8.7"
        table = {"fake": ("phlame_fake_running_module", ("numpy",))}   # numpy: a distribution that is installed
        with mock.patch.dict(artifacts.SOFTWARE_PACKAGES, table), \
                mock.patch.dict(sys.modules, {"phlame_fake_running_module": running}):
            self.assertEqual(artifacts.software_versions(["fake"])["fake"], "9.8.7")

    def test_a_package_that_was_not_imported_is_read_from_its_distribution(self):
        import numpy
        table = {"fake": ("phlame_module_nobody_imports", ("phlame-no-such-distribution", "numpy"))}
        with mock.patch.dict(artifacts.SOFTWARE_PACKAGES, table):
            # the first name is not installed, the second is: the search goes on
            self.assertEqual(artifacts.software_versions(["fake"])["fake"], numpy.__version__)

    def test_a_package_that_is_not_installed_is_recorded_as_none(self):
        table = {"fake": ("phlame_module_nobody_imports", ("phlame-no-such-distribution",))}
        with mock.patch.dict(artifacts.SOFTWARE_PACKAGES, table):
            block = artifacts.software_versions(["fake"])
        self.assertIn("fake", block)
        self.assertIsNone(block["fake"])

    def test_if_loaded_is_none_unless_this_process_imported_the_module(self):
        table = {"fake": ("phlame_fake_running_module", ("numpy",))}   # installed, but not imported
        with mock.patch.dict(artifacts.SOFTWARE_PACKAGES, table):
            self.assertIsNone(artifacts.software_versions([], if_loaded=["fake"])["fake"])
            running = types.ModuleType("phlame_fake_running_module")
            running.__version__ = "1.2.3"
            with mock.patch.dict(sys.modules, {"phlame_fake_running_module": running}):
                self.assertEqual(artifacts.software_versions([], if_loaded=["fake"])["fake"], "1.2.3")

    def test_gpus_are_those_a_loaded_tensorflow_sees(self):
        self.assertNotIn("tensorflow", sys.modules, "this test needs a process that has not imported TensorFlow")
        self.assertIsNone(artifacts.software_versions([], gpus=True)["gpus"])
        asked = []

        def list_physical_devices(kind):
            asked.append(kind)
            return [types.SimpleNamespace(name="/physical_device:GPU:0")]

        fake = types.ModuleType("tensorflow")
        fake.__version__ = "9.9.9"
        fake.config = types.SimpleNamespace(list_physical_devices=list_physical_devices)
        with mock.patch.dict(sys.modules, {"tensorflow": fake}):
            block = artifacts.software_versions(["tensorflow"], gpus=True)
        self.assertEqual((block["tensorflow"], block["gpus"], asked), ("9.9.9", ["/physical_device:GPU:0"], ["GPU"]))
        self.assertNotIn("gpus", artifacts.software_versions(["numpy"]))

    def test_the_helper_imports_no_package(self):
        before = set(sys.modules)
        artifacts.software_versions(list(artifacts.SOFTWARE_PACKAGES), gpus=True)
        imported = set(sys.modules) - before
        self.assertFalse([name for name in imported if name.split(".")[0] in PACKAGE_MODULES], sorted(imported))

    def test_every_key_the_helper_can_write_is_documented(self):
        block = artifacts.software_versions(list(artifacts.SOFTWARE_PACKAGES), gpus=True)
        self.assertEqual(documented(block), [])


@unittest.skipUnless(PIL_READY, "the PIL bench needs pyserial, Pillow, numpy, scikit-learn and seaborn")
class PilBenchSoftwareTest(PilFolderCase):
    def test_block_holds_the_host_packages_of_the_pil_level(self):
        code, log = self.run_folder(self.board())
        self.assertIsNone(code, log)
        block = self.metrics()["software"]
        import numpy
        import PIL
        import serial
        import sklearn
        self.assertEqual(block, {"python": platform.python_version(), "numpy": numpy.__version__,
                                 "pillow": PIL.__version__, "pyserial": serial.__version__,
                                 "scikit_learn": sklearn.__version__, "platform": platform.platform()})
        assert_versions_only(self, block)
        self.assertEqual(documented(block), [])

    def test_pil_level_neither_records_nor_loads_tensorflow(self):
        code, log = self.run_folder(self.board())
        self.assertIsNone(code, log)
        self.assertNotIn("tensorflow", self.metrics()["software"])
        self.assertNotIn("tensorflow", sys.modules)

    def test_run_cut_short_still_records_the_software(self):
        code, log = self.run_folder(self.board(silent_from=2), max_unanswered=1)
        self.assertEqual(code, 3, log)
        metrics = self.metrics()
        self.assertFalse(metrics["run"]["completed"])
        self.assertEqual(metrics["software"]["python"], platform.python_version())


from test_hil_session import READY as HIL_READY, HilSessionCase, hil_benchmark  # noqa: E402


@unittest.skipUnless(HIL_READY, "the HIL bench needs pyserial, OpenCV, numpy, scikit-learn and seaborn")
class HilBenchSoftwareTest(HilSessionCase):
    def test_block_holds_the_host_packages_of_the_hil_level(self):
        code, log, _ = self.run_bench(self.board())
        self.assertIsNone(code, log)
        block = self.load("metrics")["software"]
        import numpy
        import serial
        import sklearn
        self.assertEqual(block, {"python": platform.python_version(), "numpy": numpy.__version__,
                                 "opencv": hil_benchmark.cv2.__version__, "pyserial": serial.__version__,
                                 "scikit_learn": sklearn.__version__,
                                 # no --sil_model or --mil_model: nothing was classified on the PC
                                 "tensorflow": None, "keras": None, "gpus": None,
                                 "platform": platform.platform()})
        assert_versions_only(self, block)
        self.assertEqual(documented(block), [])
        self.assertNotIn("tensorflow", sys.modules)


# A small classifier for 16x12 grayscale images, saved as .keras and as an INT8 .tflite; the
# last line printed is what the packages say about themselves.
BUILD_MODELS = """
import json, platform, sys
import numpy as np
import tensorflow as tf
import PIL, sklearn
tf.keras.utils.set_random_seed(0)
model = tf.keras.Sequential([
    tf.keras.layers.Input((12, 16, 1)),
    tf.keras.layers.Conv2D(4, 3, activation="relu"),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(2, activation="softmax"),
])
model.save(sys.argv[1])
rng = np.random.default_rng(0)
def representative():
    for _ in range(8):
        yield [rng.integers(0, 256, size=(1, 12, 16, 1)).astype(np.float32)]
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
open(sys.argv[2], "wb").write(converter.convert())
print("VERSIONS " + json.dumps({
    "python": platform.python_version(), "tensorflow": tf.__version__, "keras": tf.keras.__version__,
    "numpy": np.__version__, "scikit_learn": sklearn.__version__, "pillow": PIL.__version__,
    "gpus": [device.name for device in tf.config.list_physical_devices("GPU")],
    "platform": platform.platform()}))
"""


@unittest.skipUnless(HAVE_TF and HAVE_METRICS, "the MIL and SIL levels need TensorFlow, scikit-learn and seaborn")
class MilSilSoftwareTest(unittest.TestCase):
    STEM = "Tiny_test+1+1+0.1+16+12"

    @classmethod
    def setUpClass(cls):
        import numpy as np
        from PIL import Image
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.root = Path(cls.tmp.name)
        keras_path = cls.root / (cls.STEM + ".keras")
        tflite_path = cls.root / (cls.STEM + "_int8.tflite")
        built = subprocess.run([sys.executable, "-c", BUILD_MODELS, str(keras_path), str(tflite_path)],
                               cwd=cls.root, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8")
        lines = [line for line in built.stdout.splitlines() if line.startswith("VERSIONS ")]
        if built.returncode != 0 or not lines or not tflite_path.is_file():
            raise RuntimeError("could not build the test models:\n" + built.stdout[-2000:])
        cls.packages = json.loads(lines[-1][len("VERSIONS "):])
        rng = np.random.default_rng(1)
        for name in ("non_person", "person"):
            (cls.root / "test" / name).mkdir(parents=True)
            for i in range(2):
                Image.fromarray(rng.integers(0, 256, size=(H, W), dtype=np.uint8), "L").save(
                    cls.root / "test" / name / ("img_%d.png" % i))
        cls.metrics = {}
        for level, script, model in (("mil", "mil_benchmark.py", keras_path),
                                     ("sil", "sil_benchmark.py", tflite_path)):
            out = cls.root / ("out_" + level)
            ran = subprocess.run(
                [sys.executable, str(SRC / script), "--model_path", str(model), "--data_dir", str(cls.root / "test"),
                 "--width", str(W), "--height", str(H), "--output_dir", str(out)],
                cwd=cls.root, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                encoding="utf-8")
            if ran.returncode != 0:
                raise RuntimeError("%s failed:\n%s" % (script, ran.stdout[-2000:]))
            with open(out / ("metrics_%s.json" % model.stem), encoding="utf-8") as fh:
                cls.metrics[level] = json.load(fh)

    def test_mil_block_holds_what_ran_the_float_model(self):
        block = self.metrics["mil"]["software"]
        self.assertEqual(block, {key: self.packages[key] for key in (
            "python", "tensorflow", "keras", "numpy", "scikit_learn", "pillow", "gpus", "platform")})
        self.assertIsInstance(block["gpus"], list)
        assert_versions_only(self, block)
        self.assertEqual(documented(block), [])

    def test_sil_block_holds_what_ran_the_int8_model(self):
        block = self.metrics["sil"]["software"]
        self.assertEqual(block, {key: self.packages[key] for key in (
            "python", "tensorflow", "numpy", "scikit_learn", "pillow", "platform")})
        assert_versions_only(self, block)
        self.assertEqual(documented(block), [])

    def test_levels_on_the_pc_do_not_list_the_packages_of_the_board_benches(self):
        for level in ("mil", "sil"):
            self.assertFalse({"pyserial", "opencv"} & set(self.metrics[level]["software"]), level)

    def test_this_file_does_not_load_tensorflow_into_the_test_process(self):
        self.assertNotIn("tensorflow", sys.modules)


if __name__ == "__main__":
    unittest.main()
