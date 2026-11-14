#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Checks the single image loader (src/images.py) and that every level of the
ladder reads its images through it.

The property that matters: for the same image file, MIL, SIL and the PIL bench
receive identical bytes. It is checked in two ways. By value: the array that
load_folder() gives (MIL and SIL) equals load_image() for each file, and the
payload the PIL bench sends to the board is that same array. By construction:
no training, calibration or evaluation script decodes an image with anything
else (TensorFlow, OpenCV or Pillow called directly).

Needs Pillow and numpy; TensorFlow is not imported. The bench test is skipped
when pyserial is missing.

    python -m unittest discover -s tests -v
"""
import ast
import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
sys.path.insert(0, str(SRC))

import images  # noqa: E402

W, H = 160, 120


def photo_like(seed, width=W, height=H):
    """A colour image with smooth gradients and noise, so JPEG coding is not trivial."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    base = np.stack([(xx * 255 // max(width - 1, 1)), (yy * 255 // max(height - 1, 1)),
                     ((xx + yy) * 255 // max(width + height - 2, 1))], axis=-1).astype(np.int16)
    noisy = np.clip(base + rng.integers(-40, 41, size=base.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(noisy, "RGB")


def make_dataset(root, classes=("non_person", "person"), per_class=3):
    for c_idx, cls in enumerate(classes):
        folder = Path(root) / cls
        folder.mkdir(parents=True)
        for i in range(per_class):
            photo_like(10 * c_idx + i).save(folder / ("img_%02d.jpg" % i), quality=90)


class LoadImageTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def test_grayscale_is_pillow_L_of_the_file(self):
        path = self.dir / "a.jpg"
        photo_like(1).save(path, quality=90)
        got = images.load_image(str(path), W, H)
        self.assertEqual((got.dtype, got.shape), (np.dtype("uint8"), (H, W)))
        with Image.open(path) as ref:
            self.assertTrue(np.array_equal(got, np.array(ref.convert("L"), dtype=np.uint8)))

    def test_lossless_file_keeps_its_pixels(self):
        gray = np.array(photo_like(2).convert("L"))
        path = self.dir / "a.png"
        Image.fromarray(gray, "L").save(path)
        self.assertTrue(np.array_equal(images.load_image(str(path), W, H), gray))

    def test_resizes_only_when_the_size_differs(self):
        path = self.dir / "big.png"
        big = photo_like(3, 320, 240).convert("L")
        big.save(path)
        got = images.load_image(str(path), W, H)
        self.assertEqual(got.shape, (H, W))
        self.assertTrue(np.array_equal(got, np.array(big.resize((W, H), Image.LANCZOS))))

    def test_colour(self):
        path = self.dir / "a.png"
        img = photo_like(4)
        img.save(path)
        got = images.load_image(str(path), W, H, grayscale=False)
        self.assertEqual(got.shape, (H, W, 3))
        self.assertTrue(np.array_equal(got, np.array(img)))

    def test_model_input_is_float32_with_a_channel_axis(self):
        batch = images.as_model_input(np.full((2, H, W), 7, dtype=np.uint8))
        self.assertEqual((batch.dtype, batch.shape), (np.dtype("float32"), (2, H, W, 1)))
        self.assertEqual(float(batch.max()), 7.0)
        self.assertEqual(images.as_model_input(np.zeros((2, H, W, 3), dtype=np.uint8)).shape, (2, H, W, 3))


class LoadFolderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name
        make_dataset(self.root)

    def test_order_labels_and_paths(self):
        (Path(self.root) / ".hidden").mkdir()
        (Path(self.root) / "person" / "notes.txt").write_text("not an image", encoding="utf-8")
        x, y, rel, classes = images.load_folder(self.root, W, H)
        self.assertEqual(classes, ["non_person", "person"])
        self.assertEqual(rel, ["non_person/img_00.jpg", "non_person/img_01.jpg", "non_person/img_02.jpg",
                               "person/img_00.jpg", "person/img_01.jpg", "person/img_02.jpg"])
        self.assertEqual(y.tolist(), [0, 0, 0, 1, 1, 1])
        self.assertEqual(x.shape, (6, H, W))
        self.assertTrue(all("\\" not in r for r in rel))

    def test_given_class_order_sets_the_labels(self):
        _, y, rel, classes = images.load_folder(self.root, W, H, class_names=["person", "non_person"])
        self.assertEqual(classes, ["person", "non_person"])
        self.assertTrue(rel[0].startswith("person/"))
        self.assertEqual(y.tolist(), [0, 0, 0, 1, 1, 1])

    def test_missing_class_folder_is_an_error(self):
        with self.assertRaises(FileNotFoundError):
            images.load_folder(self.root, W, H, class_names=["person", "car"])

    def test_png_and_jpeg_partitions_are_both_read(self):
        # The COCO partition is stored as lossless PNG (single channel); the earlier ones are JPEG.
        root = Path(self.root) / "mixed"
        gray = np.array(photo_like(5).convert("L"))
        for cls in ("non_person", "person"):
            (root / cls).mkdir(parents=True)
        Image.fromarray(gray, "L").save(root / "non_person" / "a.png")
        Image.fromarray(gray, "L").save(root / "person" / "b.PNG")
        photo_like(6).save(root / "person" / "a.jpg", quality=90)
        x, y, rel, _ = images.load_folder(str(root), W, H)
        self.assertEqual(rel, ["non_person/a.png", "person/a.jpg", "person/b.PNG"])
        self.assertEqual(y.tolist(), [0, 1, 1])
        self.assertTrue(np.array_equal(x[0], gray))      # a PNG gives back the stored pixels
        self.assertTrue(np.array_equal(x[2], gray))

    def test_result_does_not_depend_on_the_number_of_threads(self):
        a = images.load_folder(self.root, W, H, workers=1)[0]
        b = images.load_folder(self.root, W, H, workers=8)[0]
        self.assertTrue(np.array_equal(a, b))


class SameBytesAtEveryLevelTest(unittest.TestCase):
    """MIL and SIL take load_folder(); the PIL bench sends load_image() of each file."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name
        make_dataset(self.root)

    def test_folder_array_equals_the_file_by_file_array(self):
        x, _, rel, _ = images.load_folder(self.root, W, H)
        for row, r in zip(x, rel):
            self.assertTrue(np.array_equal(row, images.load_image(os.path.join(self.root, r), W, H)), r)

    def test_payload_of_the_pil_bench_is_that_array(self):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                import pil_benchmark
        except (ImportError, SystemExit):
            self.skipTest("pil_benchmark needs pyserial")
        x, _, rel, _ = images.load_folder(self.root, W, H)
        for row, r in zip(x, rel):
            with contextlib.redirect_stdout(io.StringIO()):
                payload = pil_benchmark.preprocess_image(os.path.join(self.root, r), W, H, True)
            self.assertEqual(payload, row.tobytes(), r)


# Scripts whose images must come from images.py, and the calls that would bypass it.
READERS = ["mil_benchmark.py", "sil_benchmark.py", "pil_benchmark.py", "quantize_int8_basic.py",
           "train_common.py", "train_mobilenet.py", "train_resnet.py", "train_squeezenet.py"]
BYPASS = {"image_dataset_from_directory", "decode_jpeg", "decode_png", "decode_image", "decode_bmp",
          "imread", "load_img", "open"}


def attribute_calls(path):
    """(object, attribute) of every attribute call in a script, e.g. ('Image', 'open')."""
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            owner = node.func.value
            found.add((owner.id if isinstance(owner, ast.Name) else getattr(owner, "attr", ""), node.func.attr))
    return found


class OneLoaderTest(unittest.TestCase):
    def test_no_script_decodes_images_by_itself(self):
        for name in READERS:
            calls = attribute_calls(SRC / name)
            direct = sorted("%s.%s" % c for c in calls
                            if c[1] in BYPASS and c[0] in ("Image", "cv2", "utils", "io", "image", "tf"))
            self.assertEqual(direct, [], "%s decodes images without images.py" % name)

    def test_the_levels_call_the_loader(self):
        uses = {name: {attr for owner, attr in attribute_calls(SRC / name) if owner == "images"}
                for name in READERS}
        for name in ("mil_benchmark.py", "sil_benchmark.py", "train_common.py"):
            self.assertIn("load_folder", uses[name], name)
        self.assertIn("load_image", uses["pil_benchmark.py"])
        self.assertIn("load_paths", uses["quantize_int8_basic.py"])
        for name in ("train_mobilenet.py", "train_resnet.py", "train_squeezenet.py"):
            self.assertIn(("train_common", "load_splits"), attribute_calls(SRC / name), name)


if __name__ == "__main__":
    unittest.main()
