#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
find_near_duplicates.py - Does the test split share photos with train/val?

A held-out test only measures generalisation if its images are not also in
the training data. split_dataset.py separates files, not photos: when the raw
data holds several copies of one photo (augmented, re-encoded, taken in a
burst), the copies can land in different splits. This script looks for them.

Typical usage (after split_dataset.py, with the images on disk)
---------------------------------------------------------------
    python src/find_near_duplicates.py --splits_dir data/splits
    python src/find_near_duplicates.py --splits_dir data/splits --region center
    python src/find_near_duplicates.py --splits_dir data/splits --embed pixels

How it works
------------
All images are compared in grayscale at 160x120.

1. Exact duplicates: identical pixels between a test image and a train/val one.
2. Candidate search: every image is embedded (ImageNet MobileNetV2, or raw
   pixels with --embed pixels); each test image is also embedded rotated by
   +/-12 degrees and mirrored. The most similar train/val images are kept.
3. Verification of each candidate with two independent measures:
     ncc      the candidate is aligned onto the test image (coarse search over
              rotation, scale and mirror, then ECC affine refinement) and the
              two are compared with normalised cross-correlation;
     inliers  SIFT keypoints of the two images are matched and the matches
              that agree with one similarity transform are counted (RANSAC).
   ncc is high for copies of one photo but also for different photos with the
   same layout (sky over sea, a field under a horizon); inliers ask for the
   same local structures in the same arrangement, which such photos lack.
4. Calibration on the data itself, for both measures:
     positive control  test images re-augmented here (rotation, scale, shift,
                       brightness/contrast, mirror, JPEG);
     negative control  random test/train pairs of the same class.
   Each threshold is the midpoint between the 1st percentile of the positives
   and the 99th percentile of the negatives. A measure whose controls overlap
   cannot tell copies apart on this dataset and is left out, with a warning.
5. Verdict per test image:
     near_duplicate  a train/val image passes both thresholds;
     review          a train/val image passes one, or comes close: look at it;
     distinct        nothing found.

--region center restricts steps 2 to 4 to the middle of the frame. Use it when
every photo shares its background (a fixed camera): the whole frame would
then make any two photos look alike.

Outputs (in --output_dir, default: the splits folder)
-----------------------------------------------------
    near_duplicates.csv              one row per test image: closest train/val
                                     image, both scores, verdict
    near_duplicates_within_test.csv  the same against the other test images
    near_duplicates_summary.json     counts, calibration, thresholds, versions
The three files hold file names and numbers only, and the same input gives the
same bytes. --review_sheet PATH also writes a picture of the pairs closest to
the thresholds; it shows the images unmasked, so it is never written unless
asked for, and it must not be published when the dataset is private.

The thresholds are heuristics calibrated on the dataset, not a proof: the
verdicts say where to look. What the method cannot detect is listed in
docs/data_independence.md.

Exit codes: 0 finished; 1 bad input, missing package or existing outputs
(--overwrite replaces them).
"""
import csv
import hashlib
import json
import os
import platform
import sys

import artifacts
import cli

try:
    import numpy as np
    import cv2
except ImportError:      # reported by main(); importing the module stays possible
    np = None
    cv2 = None

W_IMG, H_IMG = 160, 120
# rows, cols of the --region center window (the middle 70 % x 50 % of the frame)
CENTER = (slice(18, 102), slice(40, 120))
FEATURE_UPSCALE = 2          # SIFT finds few keypoints at 160x120; it runs on a 2x enlargement
MIN_REGION_PIXELS = 300
OUTPUT_NAMES = ("near_duplicates.csv", "near_duplicates_within_test.csv", "near_duplicates_summary.json")


# ----------------------------------------------------------------------------- input
def read_csv(path, sep=","):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter=sep))


def load_gray(path):
    """Image as 160x120 grayscale uint8."""
    buf = np.fromfile(path, dtype=np.uint8)          # works with non-ASCII paths on Windows
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError("cannot decode %s" % path)
    if img.shape != (H_IMG, W_IMG):
        img = cv2.resize(img, (W_IMG, H_IMG), interpolation=cv2.INTER_AREA)
    return img


def resolve(row, split, splits_dir):
    """Where the image of a manifest row is on disk, or None."""
    candidates = [os.path.join(splits_dir, split, row["class_name"], row["filename"]),
                  row.get("dst_path", ""), row.get("src_path", "")]
    for cand in candidates:
        cand = (cand or "").replace("\\", "/")
        if cand and os.path.isfile(cand):
            return cand
    return None


def load_splits(splits_dir):
    """Rows of the three manifests with the path of each image."""
    items, missing = [], []
    for split in ("train", "val", "test"):
        path = os.path.join(splits_dir, "%s_manifest.csv" % split)
        if not os.path.isfile(path):
            print("ERROR: %s not found. Run split_dataset.py first, or point --splits_dir "
                  "to the folder that holds the three manifests." % path)
            sys.exit(1)
        for row in read_csv(path):
            found = resolve(row, split, splits_dir)
            if found is None:
                missing.append("%s/%s/%s" % (split, row["class_name"], row["filename"]))
                continue
            items.append({"split": split, "cls": row["class_name"], "file": row["filename"], "path": found,
                          "hil": row.get("hil_subset", "0") in ("1", "True", "true")})
    if missing:
        print("ERROR: %d images listed in the manifests are not on disk, e.g. %s." % (len(missing), missing[:3]))
        print("The check compares pixels: the partition has to be materialized first (split_dataset.py).")
        sys.exit(1)
    if not any(it["split"] == "test" for it in items) or not any(it["split"] != "test" for it in items):
        print("ERROR: the manifests in %s list no test images or no train/val images." % splits_dir)
        sys.exit(1)
    return items


def region_mask(region):
    mask = np.zeros((H_IMG, W_IMG), bool)
    if region == "center":
        mask[CENTER] = True
    else:
        mask[:] = True
    return mask


def crop(img, region):
    return np.ascontiguousarray(img[CENTER]) if region == "center" else img


# ----------------------------------------------------------------------------- candidate search
def embed_mobilenet(imgs, batch):
    try:
        import tensorflow as tf
    except ImportError:
        print("ERROR: TensorFlow is not installed. Install the project requirements, or use "
              "--embed pixels (candidate search on raw pixels, weaker).")
        sys.exit(1)
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    model = MobileNetV2(include_top=False, weights="imagenet", pooling="avg", input_shape=(224, 224, 3))
    out = []
    for i in range(0, len(imgs), batch):
        b = np.stack([cv2.resize(a, (224, 224), interpolation=cv2.INTER_LINEAR) for a in imgs[i:i + batch]])
        b = np.repeat(b[..., None].astype(np.float32), 3, axis=-1)
        out.append(model.predict(preprocess_input(b), verbose=0))
        if (i // batch) % 20 == 0:
            print("  embedded %d/%d" % (min(i + batch, len(imgs)), len(imgs)), flush=True)
    return np.concatenate(out), {"embedding": "MobileNetV2/imagenet", "tensorflow": tf.__version__}


def embed_pixels(imgs, batch):
    out = []
    for a in imgs:
        s = cv2.resize(a, (20, 21), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
        out.append((s - s.mean()) / (s.std() + 1e-6))
    return np.stack(out), {"embedding": "pixels-20x21 (fallback, weaker)"}


def l2_normalise(E):
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)


# ----------------------------------------------------------------------------- measure 1: aligned NCC
def ncc(a, b, mask):
    if mask.sum() < MIN_REGION_PIXELS:
        return np.nan
    x = a[mask].astype(np.float64)
    y = b[mask].astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else np.nan


def rotate(a, angle, scale=1.0):
    M = cv2.getRotationMatrix2D((W_IMG / 2, H_IMG / 2), angle, scale)
    return cv2.warpAffine(a, M, (W_IMG, H_IMG), borderValue=0), M


def align_score(t, c, where):
    """Best NCC, inside the mask `where`, of candidate c aligned onto test image t.

    Coarse search over rotation (-16..16 degrees), scale (0.92-1.08) and
    horizontal mirror, with the translation of every pose from phase
    correlation; then ECC affine refinement from the three best poses. The
    image with more black border is the ECC template and its border is masked,
    so the wedges left by a rotation do not pull the alignment.
    """
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-5)
    tb = cv2.GaussianBlur(t, (5, 5), 0)
    tbf = tb.astype(np.float32)
    hann = cv2.createHanningWindow((W_IMG, H_IMG), cv2.CV_32F)
    full = np.full((H_IMG, W_IMG), 255, np.uint8)
    coarse = []
    for flip in (False, True):
        cc = np.ascontiguousarray(c[:, ::-1]) if flip else c
        cb = cv2.GaussianBlur(cc, (5, 5), 0)
        for scl in (0.92, 1.0, 1.08):
            for ang in range(-16, 17, 2):
                w, M = rotate(cb, ang, scl)
                (dx, dy), _ = cv2.phaseCorrelate(tbf, w.astype(np.float32), hann)   # translation of this pose
                M = M.copy()
                if abs(dx) < 15 and abs(dy) < 15:
                    M[0, 2] -= dx
                    M[1, 2] -= dy
                w = cv2.warpAffine(cb, M, (W_IMG, H_IMG), borderValue=0)
                ones = cv2.warpAffine(full, M, (W_IMG, H_IMG), borderValue=0)
                valid = (ones > 128) & (tb > 4) & (w > 4) & where
                score = ncc(tb, w, valid)
                if np.isfinite(score):
                    coarse.append((score, flip, ang, M))
    if not coarse:
        return {"ncc": np.nan, "flip": False, "angle": 0.0}
    coarse.sort(key=lambda x: -x[0])
    best = {"ncc": coarse[0][0], "flip": coarse[0][1], "angle": float(coarse[0][2])}
    for _, flip, ang, M in coarse[:3]:
        cc = np.ascontiguousarray(c[:, ::-1]) if flip else c
        ccw = cv2.warpAffine(cc, M, (W_IMG, H_IMG), borderValue=0)        # coarse pose applied
        swap = np.mean(ccw <= 4) > np.mean(t <= 4)
        tmpl, inp = (ccw, t) if swap else (t, ccw)
        mask = (tmpl > 4).astype(np.uint8)
        Wm = np.eye(2, 3, dtype=np.float32)
        try:
            _, Wm = cv2.findTransformECC(tmpl.astype(np.float32) / 255, inp.astype(np.float32) / 255, Wm,
                                         cv2.MOTION_AFFINE, crit, mask, 5)
        except cv2.error:
            continue
        scale = float(np.sqrt(abs(np.linalg.det(Wm[:, :2]))))
        if not (0.8 <= scale <= 1.25):
            continue
        warped = cv2.warpAffine(inp.astype(np.float32), Wm, (W_IMG, H_IMG),
                                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        ones = cv2.warpAffine(np.ones((H_IMG, W_IMG), np.float32), Wm, (W_IMG, H_IMG),
                              flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP) > 0.5
        valid = ones & (tmpl > 4) & (warped > 4)
        score = ncc(tmpl, warped, valid & where)
        if np.isfinite(score) and score > best["ncc"]:
            best = {"ncc": score, "flip": flip,
                    "angle": float(ang + np.degrees(np.arctan2(Wm[1, 0], Wm[0, 0])) * (-1 if swap else 1))}
    return best


def augment(a, rng):
    """Synthetic copy of an image: rotation, scale, shift, brightness/contrast, mirror, JPEG."""
    ang = rng.uniform(-15, 15)
    sc = rng.uniform(0.9, 1.1)
    M = cv2.getRotationMatrix2D((W_IMG / 2, H_IMG / 2), ang, sc)
    M[:, 2] += rng.uniform(-6, 6, 2)
    b = cv2.warpAffine(a, M, (W_IMG, H_IMG), borderValue=0)
    b = np.clip(b.astype(np.float32) * rng.uniform(0.8, 1.2) + rng.uniform(-20, 20), 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        b = np.ascontiguousarray(b[:, ::-1])
    _, enc = cv2.imencode(".jpg", b, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)


# ----------------------------------------------------------------------------- measure 2: local features
class Features:
    """SIFT keypoints of an image set, computed on demand and kept compact."""

    CACHE_LIMIT = 8000

    def __init__(self, where):
        self.sift = cv2.SIFT_create(nfeatures=400, contrastThreshold=0.02)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        big = (W_IMG * FEATURE_UPSCALE, H_IMG * FEATURE_UPSCALE)
        self.big = big
        self.mask = None if where.all() else cv2.resize(where.astype(np.uint8) * 255, big,
                                                        interpolation=cv2.INTER_NEAREST)
        self.cache = {}

    def describe(self, img, key=None):
        if key is not None and key in self.cache:
            return self.cache[key]
        enlarged = cv2.resize(img, self.big, interpolation=cv2.INTER_CUBIC)
        keypoints, des = self.sift.detectAndCompute(enlarged, self.mask)
        if des is None or len(keypoints) == 0:
            out = (np.zeros((0, 2), np.float32), np.zeros((0, 128), np.uint8))
        else:
            # SIFT descriptors are integers in 0..255 stored as float32: uint8 keeps them exactly
            out = (np.float32([k.pt for k in keypoints]), des.astype(np.uint8))
        if key is not None:
            if len(self.cache) >= self.CACHE_LIMIT:      # bounds the memory on large partitions
                self.cache.clear()
            self.cache[key] = out
        return out

    def inliers(self, a, b):
        """Matches between two described images that agree with one similarity transform."""
        (pa, da), (pb, db) = a, b
        if len(pa) < 4 or len(pb) < 4:
            return 0
        pairs = self.matcher.knnMatch(da.astype(np.float32), db.astype(np.float32), k=2)
        good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
        if len(good) < 4:
            return 0
        src = pa[[g.queryIdx for g in good]]
        dst = pb[[g.trainIdx for g in good]]
        _, keep = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0)
        return int(keep.sum()) if keep is not None else 0

    def score(self, plain, mirrored, other):
        """Inliers of a test image (described plain and mirrored) against another image."""
        return max(self.inliers(plain, other), self.inliers(mirrored, other))


def mirror(img):
    return np.ascontiguousarray(img[:, ::-1])


# ----------------------------------------------------------------------------- calibration and verdict
def calibrate(pos, neg, override, label):
    """Threshold between the two controls, and whether they are separated at all."""
    pos = np.array([x for x in pos if np.isfinite(x)], dtype=float)
    neg = np.array([x for x in neg if np.isfinite(x)], dtype=float)
    p01, n99 = float(np.percentile(pos, 1)), float(np.percentile(neg, 99))
    overlap = p01 <= n99
    if override is not None:
        threshold = float(override)
    elif not overlap:
        threshold = (p01 + n99) / 2
    else:
        threshold = float(np.percentile(pos, 5))
    info = {"positive_copies": {str(q): round(float(np.percentile(pos, q)), 4) for q in (1, 5, 50)},
            "negative_different_photos": {str(q): round(float(np.percentile(neg, q)), 4) for q in (50, 95, 99)},
            "negative_max": round(float(neg.max()), 4),
            "n_pos": int(len(pos)), "n_neg": int(len(neg)), "overlap": bool(overlap),
            "threshold": round(threshold, 4), "threshold_overridden": override is not None}
    print("[INFO] Calibration of %s: copies p1=%.3f p50=%.3f | different photos p50=%.3f p99=%.3f max=%.3f "
          "-> threshold %.3f%s" % (label, p01, float(np.median(pos)), float(np.median(neg)), n99, float(neg.max()),
                                   threshold, "  [WARNING: the controls overlap]" if overlap else ""), flush=True)
    return threshold, info


def verdict_of(score_ncc, score_inl, rule, exact=False):
    """near_duplicate / review / distinct for one pair of images."""
    if exact:            # identical pixels: no measure needed (a flat image has no keypoints to match)
        return "near_duplicate"
    ncc_copy = np.isfinite(score_ncc) and score_ncc >= rule["ncc"]
    ncc_close = np.isfinite(score_ncc) and score_ncc >= rule["ncc_review"]
    if not rule["use_features"]:
        return "near_duplicate" if ncc_copy else ("review" if ncc_close else "distinct")
    inl_copy = score_inl >= rule["inliers"]
    inl_close = score_inl >= rule["inliers_review"]
    if ncc_copy and inl_copy:
        return "near_duplicate"
    return "review" if (ncc_close or inl_close) else "distinct"


RANK = {"near_duplicate": 2, "review": 1, "distinct": 0}


# ----------------------------------------------------------------------------- review sheet
def review_sheet(pairs, imgs, path, rule):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    per_row = 3
    n = len(pairs)
    rows = (n + per_row - 1) // per_row
    fig, axes = plt.subplots(rows, 2 * per_row, figsize=(13, 2.2 * rows + 0.9))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    for k, pr in enumerate(pairs):
        r, c = divmod(k, per_row)
        for j, idx in enumerate((pr["ti"], pr["ci"])):
            axes[r, 2 * c + j].imshow(imgs[idx], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        col = {"near_duplicate": "#b30000", "review": "#a36b00"}.get(pr["verdict"], "#333333")
        axes[r, 2 * c].set_title("TEST %s\n%s\n" % (pr["test_file"], pr["test_class"]), fontsize=6, loc="left")
        axes[r, 2 * c + 1].set_title("%s %s (%s)\n%s  ncc %.3f  inliers %d\n"
                                     % (pr["match_split"].upper(), pr["match_file"], pr["match_class"],
                                        pr["verdict"], pr["ncc"], pr["feature_inliers"]),
                                     fontsize=6, loc="left", color=col)
    fig.suptitle("Review sheet: images shown unmasked. Do not publish it if the dataset is private. "
                 "(thresholds: ncc %.3f, inliers %s)" % (rule["ncc"], rule["inliers"] if rule["use_features"] else "-"),
                 color="red", fontsize=9, y=1 - 0.2 / (2.2 * rows + 0.9))
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=1 - 0.95 / (2.2 * rows + 0.9), wspace=0.06, hspace=0.45)
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def parse_args(argv=None):
    p = cli.ArgumentParser(description="Look for test images that are copies, or near copies, of train/val images.")
    p.add_argument("--splits_dir", default=os.path.join("data", "splits"),
                   help="Folder with the three manifests and the images of the partition (default: data/splits)")
    p.add_argument("--output_dir", default=None,
                   help="Where the three output files go (default: the splits folder)")
    p.add_argument("--region", choices=("full", "center"), default="full",
                   help="Part of the frame that is compared: the whole frame (default) or its middle, for "
                        "datasets where every photo shares the background (fixed camera)")
    p.add_argument("--embed", choices=("mobilenetv2", "pixels"), default="mobilenetv2",
                   help="Candidate search: ImageNet MobileNetV2 (default; needs TensorFlow and, the first time, "
                        "a download of its weights) or raw pixels (weaker)")
    p.add_argument("--topk", type=int, default=10,
                   help="Candidates per test image verified by alignment, about 80 ms each (default: 10)")
    p.add_argument("--topk_features", type=int, default=30,
                   help="Candidates per test image verified by local features (default: 30)")
    p.add_argument("--controls", type=int, default=300, help="Pairs in each calibration control (default: 300)")
    p.add_argument("--threshold", type=float, default=None, help="Set the ncc threshold instead of calibrating it")
    p.add_argument("--feature_threshold", type=float, default=None,
                   help="Set the inlier threshold instead of calibrating it")
    p.add_argument("--review_sheet", default=None, metavar="PATH",
                   help="Also write a picture of the pairs closest to the thresholds to PATH (.png). It shows the "
                        "images unmasked: keep it out of the repository if the dataset is private")
    p.add_argument("--sheet_pairs", type=int, default=48, help="Pairs shown in the review sheet (default: 48)")
    p.add_argument("--batch", type=int, default=64, help="Batch size of the embedding model (default: 64)")
    p.add_argument("--seed", type=int, default=20261102, help="Seed of the calibration controls (default: 20261102)")
    p.add_argument("--overwrite", action="store_true", help="Replace the output files if they already exist")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if np is None or cv2 is None:
        print("ERROR: numpy and opencv-python are needed. Run: pip install -r requirements.txt")
        sys.exit(1)
    output_dir = args.output_dir or args.splits_dir
    outputs = [os.path.join(output_dir, name) for name in OUTPUT_NAMES]
    protected = outputs + ([args.review_sheet] if args.review_sheet else [])
    artifacts.refuse_to_overwrite(protected, args.overwrite, hint="write somewhere else with --output_dir")

    rng = np.random.default_rng(args.seed)
    where = region_mask(args.region)
    items = load_splits(args.splits_dir)
    test = [i for i, it in enumerate(items) if it["split"] == "test"]
    ref = [i for i, it in enumerate(items) if it["split"] != "test"]
    ref_pos = {idx: k for k, idx in enumerate(ref)}
    print("[INFO] Images: %d (test %d, train+val %d); region compared: %s" % (len(items), len(test), len(ref), args.region))
    imgs = [load_gray(it["path"]) for it in items]

    # 1) exact duplicates
    digest = [hashlib.sha1(a.tobytes()).hexdigest() for a in imgs]
    ref_digest = {}
    for i in ref:
        ref_digest.setdefault(digest[i], []).append(i)
    exact = {i: ref_digest[digest[i]][0] for i in test if digest[i] in ref_digest}
    print("[INFO] Exact pixel duplicates test vs train/val: %d" % len(exact))

    # 2) candidates by embedding (test also rotated and mirrored)
    embed = embed_mobilenet if args.embed == "mobilenetv2" else embed_pixels
    print("[INFO] Embedding train/val/test...", flush=True)
    E, embed_info = embed([crop(a, args.region) for a in imgs], args.batch)
    E = l2_normalise(E)
    S = E[test] @ E[ref].T
    St = E[test] @ E[test].T
    for flip in (False, True):
        for ang in (-12.0, 0.0, 12.0):
            if not flip and ang == 0.0:
                continue
            print("[INFO] Embedding test variant: rotation %+.0f deg%s" % (ang, ", mirrored" if flip else ""), flush=True)
            variant = [crop(rotate(mirror(imgs[i]) if flip else imgs[i], ang)[0], args.region) for i in test]
            Ev, _ = embed(variant, args.batch)
            Ev = l2_normalise(Ev)
            S = np.maximum(S, Ev @ E[ref].T)
            St = np.maximum(St, Ev @ E[test].T)
    np.fill_diagonal(St, -1)
    order = np.argsort(-S, axis=1, kind="stable")
    top_ncc = order[:, :args.topk]
    top_feat = order[:, :max(args.topk, args.topk_features)]

    feats = Features(where)
    described = {}

    def test_features(i):
        if i not in described:
            described[i] = (feats.describe(imgs[i]), feats.describe(mirror(imgs[i])))
        return described[i]

    def inliers_between(i, j):
        plain, mirrored = test_features(i)
        return feats.score(plain, mirrored, feats.describe(imgs[j], key=j))

    # 3) calibration of both measures on the same control pairs
    print("[INFO] Calibrating on %d synthetic copies and %d pairs of different photos..."
          % (min(args.controls, len(test)), min(args.controls, len(test))), flush=True)
    pos_ncc, pos_inl, neg_ncc, neg_inl = [], [], [], []
    for i in rng.choice(test, min(args.controls, len(test)), replace=False):
        copy = augment(imgs[i], rng)
        pos_ncc.append(align_score(imgs[i], copy, where)["ncc"])
        plain, mirrored = test_features(int(i))
        pos_inl.append(feats.score(plain, mirrored, feats.describe(copy)))
    by_class = {}
    for i in ref:
        by_class.setdefault(items[i]["cls"], []).append(i)
    for i in rng.choice(test, min(args.controls, len(test)), replace=False):
        pool = by_class.get(items[i]["cls"]) or ref
        j = int(rng.choice(pool))
        neg_ncc.append(align_score(imgs[i], imgs[j], where)["ncc"])
        neg_inl.append(inliers_between(int(i), j))
    thr_ncc, cal_ncc = calibrate(pos_ncc, neg_ncc, args.threshold, "ncc")
    thr_inl, cal_inl = calibrate(pos_inl, neg_inl, args.feature_threshold, "inliers")
    use_features = (not cal_inl["overlap"]) or args.feature_threshold is not None
    if not use_features:
        print("[WARNING] Local features do not separate copies from different photos on this dataset; "
              "the verdicts use the ncc alone.")
    rule = {"ncc": thr_ncc, "ncc_review": min(cal_ncc["negative_different_photos"]["99"], thr_ncc),
            "use_features": use_features, "inliers": thr_inl,
            # more geometric agreement than any pair of different photos in the control
            "inliers_review": min(cal_inl["negative_max"] + 1, thr_inl)}

    # 4) verification of the candidates of every test image
    def verify(i, cand_ncc, cand_feat, score_of):
        """Scores every candidate of test image i and returns the strongest evidence."""
        scored = {}
        for j in list(cand_ncc) + list(cand_feat):
            scored.setdefault(j, {"ncc": np.nan, "flip": False, "angle": 0.0, "inl": 0,
                                  "exact": i in exact and j == exact[i]})
        if use_features:
            for j in scored:
                scored[j]["inl"] = inliers_between(i, j)
        for j in scored:
            wanted = j in cand_ncc or (use_features and scored[j]["inl"] >= rule["inliers_review"])
            if wanted:
                scored[j].update(align_score(imgs[i], imgs[j], where))
                if scored[j]["exact"]:
                    scored[j]["ncc"] = 1.0
        best_ncc = max((scored[j]["ncc"] for j in cand_ncc if np.isfinite(scored[j]["ncc"])), default=np.nan)

        def strength(j):
            s = scored[j]
            return (RANK[verdict_of(s["ncc"], s["inl"], rule, s["exact"])], s["inl"] if use_features else 0,
                    s["ncc"] if np.isfinite(s["ncc"]) else -2.0, score_of(j))
        j = max(scored, key=strength)
        return j, scored[j], best_ncc

    rows, picked = [], []
    ncc_only = {"at_or_above_threshold": 0, "review_band": 0}
    for k, i in enumerate(test):
        cand_ncc = [ref[j] for j in top_ncc[k]]
        if i in exact and exact[i] not in cand_ncc:
            cand_ncc = [exact[i]] + cand_ncc[:-1]
        cand_feat = [ref[j] for j in top_feat[k]]
        j, s, best_ncc = verify(i, cand_ncc, cand_feat, lambda j, k=k: float(S[k, ref_pos[j]]))
        if np.isfinite(best_ncc):
            if best_ncc >= rule["ncc"]:
                ncc_only["at_or_above_threshold"] += 1
            elif best_ncc >= rule["ncc_review"]:
                ncc_only["review_band"] += 1
        verdict = verdict_of(s["ncc"], s["inl"], rule, s["exact"])
        row = {"test_file": items[i]["file"], "test_class": items[i]["cls"], "hil_subset": int(items[i]["hil"]),
               "match_split": items[j]["split"], "match_file": items[j]["file"], "match_class": items[j]["cls"],
               "same_label": int(items[j]["cls"] == items[i]["cls"]), "exact": int(s["exact"]),
               "ncc": round(s["ncc"], 4) if np.isfinite(s["ncc"]) else "",
               "feature_inliers": int(s["inl"]) if use_features else "",
               "cosine": round(float(S[k, ref_pos[j]]), 4),
               "angle": round(s["angle"], 1), "mirrored": int(s["flip"]), "verdict": verdict}
        rows.append(row)
        picked.append({**row, "ti": i, "ci": j, "ncc": s["ncc"] if np.isfinite(s["ncc"]) else 0.0,
                       "feature_inliers": int(s["inl"])})
        if (k + 1) % 100 == 0:
            print("  verified %d/%d" % (k + 1, len(test)), flush=True)

    # copies inside the test split: they do not leak, but they make test images count twice
    test_pos = {idx: k for k, idx in enumerate(test)}
    within = []
    within_order = np.argsort(-St, axis=1, kind="stable")
    for k, i in enumerate(test):
        cand = [test[j] for j in within_order[k, :5] if test[j] != i]
        j, s, _ = verify(i, cand[:1], cand, lambda j, k=k: float(St[k, test_pos[j]]))
        within.append({"test_file": items[i]["file"], "other_test_file": items[j]["file"],
                       "ncc": round(s["ncc"], 4) if np.isfinite(s["ncc"]) else "",
                       "feature_inliers": int(s["inl"]) if use_features else "",
                       "verdict": verdict_of(s["ncc"], s["inl"], rule, s["exact"])})

    # 5) outputs
    os.makedirs(output_dir, exist_ok=True)
    for path, table in ((outputs[0], rows), (outputs[1], within)):
        with open(path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(table[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(table)

    def count(table, key="verdict"):
        return {v: sum(r[key] == v for r in table) for v in ("near_duplicate", "review", "distinct")}

    flagged = [r for r in rows if r["verdict"] == "near_duplicate"]
    summary = {
        "n_test": len(test), "n_trainval": len(ref),
        "exact_pixel_duplicates": len(exact),
        "verdicts": count(rows),
        "near_duplicate_by_class": {c: sum(r["test_class"] == c for r in flagged)
                                    for c in sorted({r["test_class"] for r in rows})},
        "near_duplicate_in_hil_subset": sum(r["hil_subset"] for r in flagged),
        "near_duplicate_match_split": {s: sum(r["match_split"] == s for r in flagged) for s in ("train", "val")},
        "near_duplicate_label_mismatch": sum(not r["same_label"] for r in flagged),
        "within_test": count(within),
        # what the ncc alone would say: the count of the first version of this check
        "ncc_alone": ncc_only,
        "calibration": {"ncc": cal_ncc, "feature_inliers": cal_inl},
        "rule": {"ncc_threshold": round(rule["ncc"], 4), "ncc_review_from": round(rule["ncc_review"], 4),
                 "features_used": use_features,
                 "inliers_threshold": round(rule["inliers"], 4) if use_features else None,
                 "inliers_review_from": round(rule["inliers_review"], 4) if use_features else None},
        "parameters": {"region": args.region, "embed": args.embed, "topk": args.topk,
                       "topk_features": args.topk_features, "controls": args.controls, "seed": args.seed,
                       "working_size": [W_IMG, H_IMG]},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__,
                        **embed_info},
    }
    with open(outputs[2], "w", encoding="utf-8", newline="\n") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")

    if args.review_sheet:
        ordered = sorted(picked, key=lambda r: (-RANK[r["verdict"]], -r["feature_inliers"], -r["ncc"]))
        review_sheet(ordered[:args.sheet_pairs], imgs, args.review_sheet, rule)
        print("[INFO] Review sheet written to %s (unmasked images: keep it private if the dataset is)."
              % args.review_sheet)

    v = summary["verdicts"]
    print("[RESULT] Test images with a near-duplicate in train/val: %d / %d (review: %d) | in hil_subset: %d | "
          "with a different label: %d | near-duplicates inside test: %d"
          % (v["near_duplicate"], len(test), v["review"], summary["near_duplicate_in_hil_subset"],
             summary["near_duplicate_label_mismatch"], summary["within_test"]["near_duplicate"]))
    print("[INFO] Wrote %s to %s. The verdicts say where to look; check the pairs before trusting the counts."
          % (", ".join(OUTPUT_NAMES), output_dir))


if __name__ == "__main__":
    main()
