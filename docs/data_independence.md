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

It takes 20 to 25 minutes for 13,000 images (1,000 of them in the test) on a desktop CPU, about 2 hours
20 minutes for 123,000 images (5,000 of them in the test) on the same machine while it ran other jobs, and the
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

Look at every `near_duplicate` and `review` pair: on the COCO partition below, three out of four shared
scenes were in the review band and not among the `near_duplicate` verdicts. If you want the decision recorded, write it to
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

**Partition.** Train and validation are all 118,287 images of COCO train2017, split 85/15 into
100,544 and 17,743; the test is all 5,000 images of COCO val2017. The labels follow the rule of the
original Visual Wake Words script (`person` when a person annotation, crowd included, covers more
than 0.5 % of the image), which gives 56,670 `person` and 61,617 `non_person` images in train2017
and 2,363 and 2,637 in val2017. The images are stored as 160×120 grayscale PNG. The two COCO splits
are different files, so no file can be on both sides: 0 shared file names, 0 exact pixel
duplicates. Different files can still show the same scene, because COCO holds photos taken moments
apart, and a few photographs that are in both of its splits.

**Detector.** `find_near_duplicates.py` on the whole frame (thresholds: `ncc` 0.7954 and 37
inliers; review from `ncc` 0.6606 or 24 inliers) gives 32 `near_duplicate`, 1,279 `review` and
3,689 `distinct` test images; inside the test, 8 `near_duplicate` and 334 `review` images, which
are 294 different pairs. None of the 32 is in the `hil_subset` and none has another label than its
match; 53 of the 1,279 are in the `hil_subset`. The run took about 2 hours 20 minutes on a desktop
CPU that was also running other jobs. The tables are in `data/splits_coco/`.

**Check by eye.** Every flagged pair was looked at side by side in the original colour photographs:
the 1,311 pairs against train/val (32 + 1,279) and the 294 pairs inside the test. The pairs were
drawn on contact sheets, the 247 with most evidence (verdict `near_duplicate`, or 24 inliers or
more) at 240×180 pixels per photo and the others at 160×120; 138 pairs were then looked at again
at up to 480×360, among them every pair whose two photos looked alike on a sheet and every pair
counted as the same from a 160×120 sheet. Two more sets were looked at to measure what the
detector leaves out (see below): the 294 `distinct` test images whose best match has 15 to 23
inliers, and a random sample of 240 of the other 3,395. That is 2,139 pairs on 102 sheets, each
listed with the decision in `data/splits_coco/near_duplicates_review.csv` (1,845 pairs against
train/val) and `near_duplicates_within_test_review.csv` (294 pairs).

The criterion is the one of the first partition: a pair counts as the same scene when the two
photos show the same objects in the same arrangement, taken moments apart or framed differently;
the same kind of scene (another beach, another giraffe), the same landmark on another day and the
same subject photographed on another occasion do not. A pair is `same_photo` when the two files
are the same photograph.

**Result: 129 of the 5,000 test images (2.6 %) show the same photo or scene as a train/val image.**

| Where the pair comes from | Pairs looked at | Same photo or scene |
|---|---|---|
| `near_duplicate`: both thresholds passed | 32 | 32 |
| `review`: 37 inliers or more, `ncc` below its threshold | 44 | 14 |
| `review`: 24 to 36 inliers | 124 | 19 |
| `review`: `ncc` ≥ 0.7954, fewer than 24 inliers | 151 | 25 |
| `review`: `ncc` 0.6606 to 0.7954, fewer than 24 inliers | 960 | 31 |
| **All flagged pairs** | **1,311** | **121** |
| Not flagged (`distinct`), 15 to 23 inliers | 294 | 8 |
| Not flagged (`distinct`), fewer than 15 inliers: random sample of 3,395 | 240 | 0 |

- **What the 129 are.** Five are the same photograph in both COCO splits, two of them with another
  colour processing. The other 124 are the same scene: the same room, table, animal, vehicle or
  match photographed again moments later or from a slightly different position (the same tennis
  match with its scoreboard, the same cat in front of the same television, the same train at the
  same signal). In a few of them the place and the framing are the same and the subject is not
  (another bus at the same street corner, another train on the same curve, the other player on the
  same court). Five are marked "probably" in the notes, where the scene has little that identifies
  it (a snow ridge, a surf break, bears on a slope, a hotel room, cows in a field).
- **By class and split.** 71 of the 2,637 `non_person` test images (2.7 %) and 58 of the 2,363
  `person` images (2.5 %); 110 match a training image and 19 a validation image. The 129 test
  images match 126 different train/val images.
- **Labels.** The two photos have the same label in 128 pairs. In one, the same cake is
  `non_person` in train and `person` in the test, where the frame includes an arm.
- **`hil_subset`.** One of its 200 images is among the 129 (`000000273712.png`, a refrigerator in a
  room; `review` for the detector, 15 inliers).
- **Agreement with the first partition.** Its test was a part of this one, and the four pairs
  confirmed there are all among the 129, with the same train/val image.

The other 1,190 flagged pairs are different photos. In 22 of them the two photos come from the same
place or event without showing the same scene (the same landmark on another day, another player of
the same tournament, another bathroom with the same fittings); they are counted as different, with
a note.

**Inside the test.** 14 of the 294 pairs show the same scene (the four the detector calls
`near_duplicate` and ten of its `review` pairs), so 28 test images count one scene twice. None of
the 28 is in the `hil_subset`; 6 of them are also among the 129. In one pair the labels differ: the
same surfer on the same wave is `person` in one frame and `non_person` in the other.

**What the detector leaves out.** The 129 are a lower bound.
- All 32 `near_duplicate` verdicts are right, but they are a quarter of what was found: 89 of the
  121 confirmed among the flagged pairs were in the `review` band, and 8 more were not flagged at
  all. On this partition the verdict `near_duplicate` alone understates the shared scenes by a
  factor of four, and the review band cannot be skipped.
- Below the review band, 8 of the 294 pairs with 15 to 23 inliers are the same scene (2.7 %); they
  are among the 129. In a random sample of 240 of the remaining 3,395 `distinct` test images
  (seed 20261114) none is: with 95 % confidence fewer than 1.3 % of those images, at most about 42,
  share their scene with the train/val image the detector kept as their best match. Two pairs of
  the sample show the same subject on another occasion (a person at the same mirror on another
  day, possibly the same bird feeder).
- Both figures concern the one train/val image recorded for each test image. A test image whose
  scene is shared with another training image, one the embedding did not rank among its candidates,
  is not seen by the detector nor by this check.

**Effect on the results.** `independence_report.py` on the 19 COCO runs (MIL and SIL for seven
architectures, PIL on the Portenta H7 for the five that fit; the PIL runs on the Nicla Vision give the
same prediction for every image); macro-F1, Δ = cleaned − full, 95 % paired bootstrap interval; full
table with accuracy in `results/tables/independence_coco.csv`:

| Level | Architecture | F1 full (n=5000) | F1 `confirmed` (n=4871) | Δ [95 % interval] | F1 `strict` (n=3681) | Δ [95 % interval] |
|---|---|---|---|---|---|---|
| MIL | MobileNet | 0.8462 | 0.8461 | −0.0002 [−0.0017, +0.0015] | 0.8397 | −0.0066 [−0.0124, −0.0009] |
| MIL | MobileNetV2 | 0.8575 | 0.8554 | −0.0021 [−0.0032, −0.0009] | 0.8490 | −0.0085 [−0.0139, −0.0030] |
| MIL | MobileNetV3Large | 0.8806 | 0.8791 | −0.0015 [−0.0026, −0.0003] | 0.8731 | −0.0075 [−0.0125, −0.0024] |
| MIL | MobileNetV3Small | 0.8366 | 0.8362 | −0.0004 [−0.0020, +0.0013] | 0.8313 | −0.0052 [−0.0112, +0.0007] |
| MIL | ResNet18 | 0.8511 | 0.8494 | −0.0017 [−0.0029, −0.0003] | 0.8401 | −0.0110 [−0.0164, −0.0054] |
| MIL | ResNet8 | 0.8340 | 0.8327 | −0.0013 [−0.0027, +0.0003] | 0.8250 | −0.0090 [−0.0148, −0.0033] |
| MIL | SqueezeNet | 0.8359 | 0.8339 | −0.0021 [−0.0034, −0.0007] | 0.8281 | −0.0079 [−0.0136, −0.0021] |
| SIL | MobileNet | 0.8431 | 0.8426 | −0.0005 [−0.0020, +0.0012] | 0.8326 | −0.0105 [−0.0161, −0.0047] |
| SIL | MobileNetV2 | 0.8573 | 0.8556 | −0.0017 [−0.0029, −0.0004] | 0.8500 | −0.0074 [−0.0127, −0.0020] |
| SIL | MobileNetV3Large | 0.8644 | 0.8633 | −0.0011 [−0.0024, +0.0003] | 0.8612 | −0.0032 [−0.0088, +0.0023] |
| SIL | MobileNetV3Small | 0.7842 | 0.7834 | −0.0008 [−0.0025, +0.0010] | 0.7781 | −0.0061 [−0.0128, +0.0005] |
| SIL | ResNet18 | 0.8534 | 0.8520 | −0.0014 [−0.0027, +0.0000] | 0.8425 | −0.0109 [−0.0165, −0.0054] |
| SIL | ResNet8 | 0.8344 | 0.8329 | −0.0015 [−0.0029, +0.0001] | 0.8261 | −0.0083 [−0.0141, −0.0025] |
| SIL | SqueezeNet | 0.8340 | 0.8318 | −0.0021 [−0.0034, −0.0007] | 0.8258 | −0.0082 [−0.0141, −0.0024] |
| PIL | MobileNet | 0.8421 | 0.8417 | −0.0005 [−0.0020, +0.0012] | 0.8310 | −0.0112 [−0.0169, −0.0054] |
| PIL | MobileNetV2 | 0.8575 | 0.8558 | −0.0017 [−0.0029, −0.0003] | 0.8502 | −0.0073 [−0.0127, −0.0017] |
| PIL | MobileNetV3Small | 0.7788 | 0.7779 | −0.0009 [−0.0027, +0.0009] | 0.7742 | −0.0046 [−0.0113, +0.0021] |
| PIL | ResNet8 | 0.8337 | 0.8325 | −0.0013 [−0.0027, +0.0003] | 0.8258 | −0.0080 [−0.0138, −0.0021] |
| PIL | SqueezeNet | 0.8344 | 0.8322 | −0.0021 [−0.0034, −0.0007] | 0.8263 | −0.0081 [−0.0138, −0.0022] |

- **Without the 129 confirmed images** macro-F1 changes by −0.0021 to −0.0002 and accuracy by
  −0.0021 to −0.0001; 8 of the 19 intervals exclude zero, the order of the architectures is the same at MIL and SIL (at PIL `SqueezeNet` and `ResNet8`, 0.0007 apart, swap places), and the gap between two levels of one
  architecture moves by at most 0.0004.
- **Without all 1,319 suspect images** macro-F1 changes by −0.0112 to −0.0032; 15 of the 19 intervals exclude zero,
  the order of the architectures is the same at each level, and the gap between two levels of one architecture moves by at most 0.0043.
  1,190 of those images are different photos from their closest train/val image, so this drop is not
  an effect of shared data alone: test images that resemble some training image are slightly easier
  for the models.

Removing the 129 shared scenes changes no macro-F1 of this benchmark by more than 0.0021.

**Visual Wake Words partition.** The published VWW partition (`data/splits_vww`, 2,000 images) was
drawn from COCO val2017, and this test is all of val2017: every image of that partition is in this
test (1,400 from its train split, 300 from its validation split, 300 from its test split), and none
is in this train/val. The two partitions were labelled with different rules, and 53 of the 2,000
images have another label here. The two benchmarks have their own models and are evaluated
separately, so neither result is affected. But a model trained on the VWW partition has seen 1,400
images of this test and must not be evaluated on it, and the two partitions must not be merged.

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
- **Shared scenes below the review band.** A scene photographed twice from different positions has
  few matching keypoints and a low cross-correlation, and can fall below both review limits. On the
  COCO partition, 8 of the 294 `distinct` test images with 15 to 23 inliers were the same scene as
  their match, and none of a random sample of 240 with fewer.
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
