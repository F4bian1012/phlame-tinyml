# Independence of the Test Split

A test split measures generalisation only if its images are not also in the training data.
`split_dataset.py` separates files, not photos: if the raw data holds several copies of one photo
(augmented, re-encoded, taken in a burst), the copies can land in different splits and the test
accuracy is then partly a memory test. This page says how the repository checks for that, what the
check found on each published partition, and what it cannot find.

## Method

`src/find_near_duplicates.py` compares every test image with the train and validation images, in
grayscale at 160×120:

1. **Exact duplicates.** Identical pixels between a test image and a train/val image.
2. **Candidate search.** Every image is embedded with an ImageNet MobileNetV2 (or with its raw
   pixels, `--embed pixels`); each test image is also embedded rotated by ±12° and mirrored. The
   most similar train/val images of each test image are the candidates.
3. **Verification** of each candidate with two independent measures:
   - `ncc`: the candidate is aligned onto the test image (search over rotation, scale and mirror,
     then ECC affine refinement) and the two are compared with normalised cross-correlation. The
     10 best candidates are verified this way (`--topk`).
   - `feature_inliers`: SIFT keypoints of the two images are matched and the matches that agree
     with one similarity transform are counted (RANSAC). The 30 best candidates are verified this
     way (`--topk_features`).
4. **Calibration on the partition itself**, for both measures: 300 test images re-augmented by the
   script (rotation, scale, shift, brightness and contrast, mirror, JPEG) are the copies; 300 random
   test/train pairs of the same class are the different photos. Each threshold is the midpoint
   between the 1st percentile of the copies and the 99th percentile of the different photos. A
   measure whose two controls overlap is left out, with a warning.
5. **Verdict** per test image: `near_duplicate` when a train/val image passes both thresholds (or
   has identical pixels); `review` when a train/val image passes only one of the two, or comes
   close to one (`ncc` above the 99th percentile of the different photos, or more inliers than any
   pair of different photos in the control); `distinct` otherwise.

Two measures are needed because each fails alone. The cross-correlation is high for copies of one
photo, but also for different photos that share a layout: sky over sea, a field under a horizon, a
room with a window on the left. The feature count asks for the same local structures in the same
arrangement, which such photos do not have, but it needs texture: a flat image has few keypoints.

`--region center` restricts steps 2 to 4 to the middle of the frame (rows 18–102, columns 40–120).
It is meant for datasets in which every photo shares its background, as with a fixed camera: over
the whole frame any two photos of such a dataset look alike.

The verdicts say where to look; they are not a proof. The thresholds are heuristics, and whether two
photos are "the same" is in the end decided by a person looking at them.

## Checking Your Own Partition

After `split_dataset.py`, with the images on disk:

```bash
python src/find_near_duplicates.py --splits_dir data/splits
```

It takes 20 to 25 minutes for 13,000 images (1,000 of them in the test) on a desktop CPU, and the
first run downloads the ImageNet weights of MobileNetV2. It writes three files into the splits folder
(or `--output_dir`) and refuses to replace them unless `--overwrite` is given:

| File | Content |
|---|---|
| `near_duplicates.csv` | One row per test image: the train/val image with the strongest evidence (`match_split`, `match_file`, `match_class`), `same_label`, `exact`, `ncc`, `feature_inliers`, the embedding `cosine`, the `angle` and `mirrored` flag of the alignment, and the `verdict`. `ncc` is empty when that image was verified by features only (it was not among the 10 verified by alignment and had too few inliers to be aligned). |
| `near_duplicates_within_test.csv` | The same against the other test images. Copies inside the test do not leak into training, but they make one photo count twice. |
| `near_duplicates_summary.json` | Counts, the calibration percentiles, the thresholds, the parameters and the package versions. |

The files hold file names and numbers only, and the same input gives the same bytes. To look at the
pairs, add `--review_sheet some/path.png`: it draws the pairs with the strongest evidence side by
side. The sheet shows the images unmasked, so it is written only on request; keep it out of the
repository if the dataset is private.

Look at every `near_duplicate` and `review` pair. If you want the decision recorded, write it to
`near_duplicates_review.csv` next to the other files, one row per pair looked at, with the columns
`test_file`, `match_file`, `checked` (`same_photo`, `same_scene` or `different`) and `note`.

Then measure what the shared images change:

```bash
python src/independence_report.py --splits_dir data/splits
```

It recomputes accuracy and macro-F1 of every result file of the partition (`predictions_*.csv` of
MIL, SIL and PIL, `latency_metrics_*.csv` of HIL) on the full test and on two cleaned versions of
it, and writes `results/tables/independence_<name>.csv`, where `<name>` is the splits folder without
its `splits_` prefix (`independence_coco.csv` for `data/splits_coco`). A result file is used when
all its images belong to the test split of the partition, so results of other partitions in the same
folders are left out.

- `confirmed`: without the test images that have a near-duplicate in train/val, as corrected by the
  review file when there is one;
- `strict`: without every test image the detector did not call `distinct`. It is an upper bound of
  what the screen could be hiding, since most of the review band are different photos.

`delta` is cleaned minus full. Its 95 % interval comes from resampling the test images 10,000 times
(seeded) and computing both figures on each resample, so it is paired.

The two published partitions below were checked with these commands (the images have to be on disk):

```bash
python src/find_near_duplicates.py --splits_dir data/splits_coco --overwrite
python src/independence_report.py  --splits_dir data/splits_coco --overwrite
python src/find_near_duplicates.py --splits_dir data/splits_plates --region center --overwrite
python src/independence_report.py  --splits_dir data/splits_plates --overwrite
```

If the test shares photos with train/val, the remedy is upstream: remove the copies from the raw
data, or keep all the copies of one photo in the same split (`split_dataset.py` does not do that by
itself), then split and train again.

## COCO (Main Benchmark)

**Partition.** Train and validation come from COCO train2017 (6,000 images per class, split 85/15
into 10,200 and 1,800) and the test from COCO val2017 (500 per class). The two COCO splits are
different files, so no file can be on both sides: 0 shared file names, 0 exact pixel duplicates.
Different files can still show the same scene, because COCO holds photos taken moments apart.

**Detector.** `find_near_duplicates.py` on the whole frame (thresholds: `ncc` 0.8105 and 40.84
inliers; review from `ncc` 0.6941 or 23 inliers) gives 1 `near_duplicate`, 190 `review` and 809
`distinct` test images; inside the test, 0 `near_duplicate` and 46 `review`. The tables are in
`data/splits_coco/`.

**Check by eye.** Three sets of pairs were looked at side by side in the original colour images:
the 191 pairs above; the 134 pairs of a first check that used the cross-correlation alone on the
middle of the frame (42 above its threshold, 92 in its review band; `--region center` reproduces
both counts in `ncc_alone` of the summary); and the 40 pairs with most geometric agreement of an
independent second search (colour MobileNetV2 embeddings of the full-size originals, SIFT on
400-pixel-wide images). That is 350 different pairs over 245 test images, listed with the decision
in `data/splits_coco/near_duplicates_review.csv`. A pair was counted as the same scene when the two
photos show the same objects in the same arrangement, taken moments apart or framed slightly
differently; the same kind of scene (another beach, another giraffe) was not.

**Result: 4 of the 1,000 test images show the same scene as a train/val image.**

| Test image | Train/val image | Split | Class of both | `ncc` | Inliers | Detector | What the two photos show |
|---|---|---|---|---|---|---|---|
| `000000368684.jpg` | `000000247130.jpg` | train | `non_person` | 0.9345 | 95 | `near_duplicate` | The same room and furniture, moments apart |
| `000000023781.jpg` | `000000038083.jpg` | train | `non_person` | 0.7987 | 54 | `review` | The same market stall, framed slightly differently |
| `000000409211.jpg` | `000000441363.jpg` | train | `person` | 0.7213 | 50 | `review` | The same tennis match: player, court and scoreboard at another instant |
| `000000338901.jpg` | `000000227428.jpg` | val | `non_person` | 0.8188 | 9 | `review` | The same dog on the same sofa, in another pose |

The labels agree in the four pairs and none of the four test images is in the `hil_subset`. The
other 346 pairs are different photos; in two of them the subject may be the same seen from another
viewpoint (a bear, a plate of food), and they are counted as different. None of the 46 pairs inside
the test shows the same scene.

All four are among the pairs the detector flags or puts in its review band, and the independent
search ranks them among its strongest pairs. The first check, on the cross-correlation alone, had
reported 42 near-duplicates (9 of them with a different label, 9 in the
`hil_subset`) and 5 copies inside the test: 40 of the 42 and the 5 are different photos that share a
layout, the label mismatches and the rig images are all among those 40, and two of the four real
pairs were missed.

**Effect on the results.** `independence_report.py` on the 18 COCO runs that exist (MIL and SIL for
seven architectures, PIL for four); macro-F1, Δ = cleaned − full, 95 % paired bootstrap interval;
full table with accuracy in `results/tables/independence_coco.csv`:

| Level | Architecture | F1 full (n=1000) | F1 `confirmed` (n=996) | Δ [95 % interval] | F1 `strict` (n=809) | Δ [95 % interval] |
|---|---|---|---|---|---|---|
| MIL | MobileNet | 0.7460 | 0.7460 | +0.0000 [−0.0015, +0.0019] | 0.7451 | −0.0009 [−0.0138, +0.0126] |
| MIL | MobileNetV2 | 0.7777 | 0.7778 | +0.0001 [−0.0012, +0.0021] | 0.7696 | −0.0081 [−0.0202, +0.0042] |
| MIL | MobileNetV3Large | 0.8188 | 0.8190 | +0.0003 [−0.0011, +0.0023] | 0.8146 | −0.0042 [−0.0153, +0.0072] |
| MIL | MobileNetV3Small | 0.7723 | 0.7724 | +0.0001 [−0.0013, +0.0021] | 0.7706 | −0.0017 [−0.0141, +0.0109] |
| MIL | ResNet18 | 0.6230 | 0.6225 | −0.0005 [−0.0023, +0.0013] | 0.6052 | −0.0178 [−0.0323, −0.0035] |
| MIL | ResNet8 | 0.6083 | 0.6087 | +0.0004 [−0.0016, +0.0026] | 0.5865 | −0.0218 [−0.0373, −0.0063] |
| MIL | SqueezeNet | 0.3333 | 0.3329 | −0.0004 [−0.0014, +0.0004] | 0.3270 | −0.0064 [−0.0133, +0.0005] |
| SIL | MobileNet | 0.7309 | 0.7308 | −0.0001 [−0.0016, +0.0017] | 0.7256 | −0.0053 [−0.0185, +0.0082] |
| SIL | MobileNetV2 | 0.7825 | 0.7827 | +0.0002 [−0.0012, +0.0022] | 0.7770 | −0.0056 [−0.0173, +0.0065] |
| SIL | MobileNetV3Large | 0.7800 | 0.7801 | +0.0001 [−0.0013, +0.0021] | 0.7773 | −0.0027 [−0.0148, +0.0097] |
| SIL | MobileNetV3Small | 0.6604 | 0.6601 | −0.0003 [−0.0020, +0.0016] | 0.6582 | −0.0022 [−0.0157, +0.0113] |
| SIL | ResNet18 | 0.6229 | 0.6224 | −0.0005 [−0.0023, +0.0013] | 0.5981 | −0.0249 [−0.0389, −0.0107] |
| SIL | ResNet8 | 0.6105 | 0.6109 | +0.0004 [−0.0015, +0.0026] | 0.5873 | −0.0232 [−0.0386, −0.0082] |
| SIL | SqueezeNet | 0.3333 | 0.3329 | −0.0004 [−0.0014, +0.0004] | 0.3270 | −0.0064 [−0.0133, +0.0005] |
| PIL | MobileNet | 0.7438 | 0.7438 | +0.0000 [−0.0015, +0.0019] | 0.7417 | −0.0021 [−0.0151, +0.0111] |
| PIL | MobileNetV2 | 0.7778 | 0.7779 | +0.0001 [−0.0012, +0.0021] | 0.7750 | −0.0027 [−0.0148, +0.0095] |
| PIL | MobileNetV3Small | 0.6550 | 0.6566 | +0.0016 [−0.0004, +0.0041] | 0.6497 | −0.0053 [−0.0196, +0.0086] |
| PIL | ResNet8 | 0.6112 | 0.6116 | +0.0004 [−0.0015, +0.0026] | 0.5898 | −0.0214 [−0.0360, −0.0066] |

- **Without the four confirmed images** macro-F1 and accuracy change by at most 0.0016 in every
  run, no interval excludes zero, the order of the architectures is the same at each level, and the
  gap between two levels of one architecture moves by at most 0.0019.
- **Without all 191 suspect images** macro-F1 changes by −0.0009 to −0.0081 for the MobileNet
  family and SqueezeNet (every interval includes zero) and by −0.018 to −0.025 for ResNet8 and
  ResNet18 (the intervals exclude zero). 187 of those 191 images are different photos from their
  closest train/val image, so this is not shared data: the two residual networks, the weakest
  classifiers that learned something on this benchmark, do somewhat better on test images whose
  layout resembles a training image. The gap between two levels of one architecture moves by at
  most 0.0070; two pairs of architectures that were within 0.006 of each other swap places
  (MobileNetV2 and MobileNetV3Small at MIL, MobileNetV2 and MobileNetV3Large at SIL).

The results on this benchmark do not depend on the four shared scenes.

**Visual Wake Words partition.** The published VWW partition (`data/splits_vww`) was drawn from COCO
val2017 with the same script and seed as this test split, and 719 of the 1,000 test images above are
also in it (499 in its train split, 119 in its validation split, 101 in its test split). The two
benchmarks have their own models and are evaluated separately, so neither result is affected. But a
model trained on the VWW partition must not be evaluated on this test split, and the two partitions
must not be merged.

## Plates (Secondary Benchmark)

**The test split of the plates partition is not independent of its training data.** The partition
(`data/splits_plates`: 940 test images, 5,336 train and validation images) comes from a dataset that
holds several copies of many of its photos (identical, rotated or mirrored), and the split separated
the copies.

`find_near_duplicates.py --splits_dir data/splits_plates --region center` (fixed camera, so the
middle of the frame is compared; thresholds: `ncc` 0.8426 and 25.95 inliers) gives:

| | Test images | Of 940 |
|---|---|---|
| Identical pixels to a train/val image | 12 | 1.3 % |
| `near_duplicate` (both measures) | 726 | 77.2 % |
| `review` | 96 | 10.2 % |
| `distinct` | 118 | 12.6 % |

- Of the 726, 574 match a training image and 152 a validation image; 448 of the 714 that are not
  exact copies are mirrored or rotated by 3° or more with respect to their match; 11 carry the other
  class label than their match.
- With the cross-correlation alone, as in the first check of 2 November 2026, 763 test images
  (81.2 %) are above the threshold and 41 are in the review band; the script reproduces both counts.
- In the 200-image `hil_subset`: 148 `near_duplicate`, 25 `review`, 27 `distinct`.
- Inside the test: 274 images are a `near_duplicate` of another test image.

These are the verdicts of the detector; unlike for COCO, the pairs were not checked one by one for
this page. The controls are well separated on this partition (copies: 1st percentile `ncc` 0.92 and
34 inliers; different photos: 99th percentile 0.76 and 18), and the exact duplicates need no
judgement. The tables are in `data/splits_plates/`; they hold file names and scores, no image.

**Effect on the results** (`results/tables/independence_plates.csv`, the 24 published plates runs).
On the 118 `distinct` images macro-F1 differs from the published figure by −0.052 to +0.063 at MIL,
SIL and PIL, with intervals between ±0.03 and ±0.06 wide; five of the 19 exclude zero (MobileNet and
MobileNetV3Small score higher on the clean images). On the rig only 27 of the 200 images are
`distinct`, too few to say anything (intervals of ±0.05 to ±0.13).

The clean subset is too small to re-establish the plates results on independent images. They are
kept as a secondary benchmark with this reading: every level of the ladder is evaluated on the same
940 (or 200) images, so the differences between levels are comparisons on identical images and
remain valid as such; the accuracies themselves are measured on a test that shares most of its
photos with the training data and say little about vehicles the models have not seen.

## What the Check Cannot Detect

- **The same subject photographed again.** Two photos of one vehicle, person or place taken from
  different viewpoints or on different days are not related by a rotation, a scale and a mirror, so
  neither measure links them. Independence at the level of subjects (no vehicle, no person, no
  location on both sides) can only come from how the data were collected and split.
- **Copies beyond the search.** Only the 10 (alignment) and 30 (features) train/val images that the
  embedding ranks closest to a test image are verified. A copy that the embedding ranks lower is
  never looked at. Strong crops, rotations beyond about 16°, overlays and occlusions also defeat the
  alignment; the feature count tolerates more, but one measure alone gives `review`, not
  `near_duplicate`.
- **Images without texture.** A flat or blurred image has few keypoints: the feature count cannot
  confirm a copy of it, which leaves it in `review` at most (identical pixels are still found).
- **Differences below the working size.** Everything is compared in grayscale at 160×120. Two files
  that differ only in colour, or only in detail lost at that size, count as the same image.
- **Thresholds from synthetic controls.** The copies of the calibration are made by the script with
  one family of transformations, and the different photos are random pairs, whereas the candidates
  that are verified are by construction the most similar images of the partition. The
  cross-correlation of different photos is therefore higher among the candidates than in the
  control, which is why it flags too much on its own.
- **Other partitions.** The check looks at one partition. It does not see that two published
  partitions share images (see the note on Visual Wake Words above), nor that a model was
  pre-trained elsewhere on images of the test.
- **Train against validation** is not checked: copies there affect model selection, not the test.
