# Vehicle Benchmark: an Application Case on Private Images

The benchmark of this repository is the public one, on MS-COCO: see [Results](../README.md#results) in the
README. This page holds the benchmark the framework was developed on. It is kept as an application case, outside
the main results, and its figures are not to be read as the results of the framework: read the three reasons
below before any number on this page.

## What It Is

- **Task and data.** A two-class classification of vehicle photographs taken by a fixed camera, at 160×120
  grayscale. The dataset has 7,525 images (4,387 / 3,138), balanced by undersampling to 6,276 (3,138 per class)
  and split with seed 42 into 4,394 train, 942 validation and 940 test images (`data/splits_plates/`); 200 of the
  test images, 100 per class, are flagged as the `hil_subset`. The classes are published as `Class1` and `Class2`.
- **Models.** Seven architectures, in the runs named `{architecture}+32+20+0.0001+160+120` (batch 32, 20 epochs,
  learning rate 1e-4, 160×120): the names a run takes without a `--tag`.
- **Levels.** The seven models at MIL and SIL; the five that fit the flash at PIL (940 images) and at HIL (200
  images), all on one Arduino Portenta H7 with the Vision Shield (HM01B0 camera). The PIL runs are dated 12
  September 2026 and the HIL sessions 12 and 13 September 2026, at 10 lux, 24 cm and 24 °C, with each stimulus
  1.5 s on screen before its capture (`--settle 1.5`, the default of the bench). The MIL and SIL files are dated
  16 October 2026.
- **Repeatability.** Four more HIL sessions of `MobileNetV2` on the same 200 images
  (`results/hil/rep1..rep4/`): three with the presentation order of the first session (`seed=42`) and one with
  another order (`seed=7`).
- **Statistics.** Eight tables in `results/tables/`, rebuilt from the per-image result files (see
  [Statistical Tables](#statistical-tables)).

## Why It Is Not the Reference Benchmark

1. **Its images are private.** The photographs show identifiable licence plates and cannot be redistributed (see
   [Data Availability](#data-availability)), so nobody else can run a level of this benchmark. What the
   repository allows is to rebuild its tables from the per-image results.
2. **Its test split is not independent of its training data.** 726 of its 940 test images are near-duplicates of
   a train/val image, 12 of them exact copies, so its accuracies are measured largely on photos seen in training
   (see [`data_independence.md`](data_independence.md#plates-secondary-benchmark)).
3. **It was measured before three checks the framework has today:** before the single image loader, so MIL and
   SIL did not read the same pixels as PIL; with the default TFLite interpreter at SIL, whose kernels are not
   those of the chip; and without the model fingerprint at PIL, so its PIL files do not prove which model was on
   the board. Each is detailed under [Limitations Specific to This Benchmark](#limitations-specific-to-this-benchmark).

## Why It Is Kept

- It is, so far, the only benchmark of the repository measured at the four levels for several architectures, and
  the only one with repeated HIL sessions and with the eight statistical tables. The Visual Wake Words run
  ([Results](../README.md#results), section 2) has the four levels for one model, and the COCO benchmark stops at
  PIL.
- For traceability: the framework, its result format and its statistics were developed on it, its result files
  and tables are in the repository as they were measured, `python src/build_ladder_tables.py` rebuilds its tables
  when it is given no benchmark, and the test suite recomputes them (`tests/test_published_tables.py`).
- What it does support: every level is evaluated on the same 940 (or 200) images, so the differences between
  levels are comparisons on identical images and remain valid as such. The accuracies themselves say little about
  vehicles the models have not seen.

## Four-Level Ladder (Seven Architectures, 160×120)

> **The test split of this benchmark is not independent of its training data.** 726 of its 940 test images are near-duplicates of a train/val image (12 of them exact copies), so the accuracies below are measured largely on photos seen in training (see [`data_independence.md`](data_independence.md#plates-secondary-benchmark)). The comparison between levels still holds, because every level is evaluated on the same images.

Evaluated on the balanced binary partition (`data/splits_plates/test`, n = 940 test images for MIL/SIL/PIL; n = 200 `hil_subset == 1` stimuli at 10 lux, 24 cm, 24 °C for HIL). All metrics are **F1-macro** (which equals weighted F1 and accuracy to within ≤ 0.002 on this 50/50 split); latencies are measured on-chip via the Cortex-M7 `DWT->CYCCNT` cycle counter at `SystemCoreClock = 400.0 MHz`:

| Architecture | `.tflite` (INT8) | Arena used | **MIL** (n=940) | **SIL** (n=940) | **PIL** (n=940 / n=200) | **HIL** (n=200) | HIL `INF` (ms) | HIL `TOTAL` (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `MobileNet` (α=0.25) | 311.5 KiB | 177.2 KiB | 0.9202 | 0.9159 | 0.9170 / 0.9399 | 0.8736 | 236.49 | 241.66 |
| `MobileNetV2` (α=0.35) | 630.5 KiB | 348.1 KiB | 0.9478 | 0.9446 | 0.9468 / 0.9450 | 0.8785 | 396.00 | 401.17 |
| `MobileNetV3Small` (α=0.75) | 817.3 KiB | 220.5 KiB | 0.9479 | 0.9181 | 0.9266 / 0.9400 | 0.9200 | 478.35 | 483.64 |
| `SqueezeNet` | 808.2 KiB | 404.7 KiB | 0.9159 | 0.9159 | 0.9180 / 0.9099 | 0.8899 | 1126.24 | 1131.56 |
| `ResNet8` | 1249.2 KiB | 388.8 KiB | 0.9318 | 0.9308 | **0.9308 / 0.9600**† | 0.8632 | 2814.02 | 2819.17 |
| `MobileNetV3Large` (α=0.75) | 2180.8 KiB | — | 0.9489 | 0.9553 | *Flash OOM* | *Flash OOM* | — | — |
| `ResNet18` | 11057.3 KiB | — | 0.9393 | 0.9393 | *Flash OOM* | *Flash OOM* | — | — |

> †**Subset representativeness note (`ResNet8` PIL on n=940 vs. n=200):** Both PIL figures are reported explicitly because the fixed 200-image `hil_subset` happens to be slightly easier for `ResNet8` than the full 940-image test split (`0.9600` on n=200 vs. `0.9308` on n=940, a +0.0292 sampling offset). Computing the physical-frontend gap (PIL → HIL) always requires comparing paired predictions on the **exact same n=200 images** (`0.9600 → 0.8632`, Δ = −0.0968, exact McNemar p < 0.001, Holm-adjusted p < 0.001), rather than mixing n=940 with n=200.

`MobileNetV3Large` and `ResNet18` do not fit the internal flash of the board (see
[Hardware Details](../README.md#hardware-details) in the README), so they stop at SIL.

## Paired 2×3 Decomposition on the HIL Subset (n = 200, `hil_subset == 1`)

Cross-running the exact same 200 test stimuli as **digital files** (MIL₂₀₀, SIL₂₀₀, PIL₂₀₀) and as **HM01B0 camera-captured frames** (MIL\*₂₀₀, SIL\*₂₀₀, HIL₂₀₀) isolates where degradation enters the pipeline (F1-macro shown; exact McNemar p-values with Holm–Bonferroni adjustment across the 5 architectures in parentheses):

| Architecture | MIL₂₀₀ (FP32, file) | SIL₂₀₀ (INT8, file) | PIL₂₀₀ (Chip, file) | MIL\*₂₀₀ (FP32, frame) | SIL\*₂₀₀ (INT8, frame) | HIL₂₀₀ (Chip, frame) | SIL\*₂₀₀ = HIL₂₀₀ agreement | Physical gap Δ(PIL₂₀₀ → HIL₂₀₀) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `MobileNet` | 0.9449 | 0.9349 | 0.9399 | 0.8995 | 0.8736 | 0.8736 | **200 / 200 (100.0%)** | −0.0663 (p = 0.002, Holm 0.009) |
| `MobileNetV2` | 0.9450 | 0.9400 | 0.9450 | 0.9093 | 0.8733 | 0.8785 | **195 / 200 (97.5%)** | −0.0664 (p = 0.007, Holm 0.022) |
| `MobileNetV3Small` | 0.9500 | 0.9150 | 0.9400 | 0.9148 | 0.9100 | 0.9200 | **194 / 200 (97.0%)** | −0.0200 (n.s.) |
| `SqueezeNet` | 0.8949 | 0.8949 | 0.9099 | 0.8949 | 0.8899 | 0.8899 | **200 / 200 (100.0%)** | −0.0200 (n.s.) |
| `ResNet8` | 0.9600 | 0.9600 | 0.9600 | 0.8527 | 0.8527 | 0.8632 | **198 / 200 (99.0%)** | −0.0968 (p < 0.001, Holm < 0.001) |

- **On-chip execution fidelity (SIL ↔ PIL and SIL\* ↔ HIL):** Across all 10 paired comparisons (5 architectures × 2 input domains), moving from PC INT8 simulation to the physical Cortex-M7 chip does not lower F1-macro in any of them (Δ from 0.0000 to +0.0250), and no difference is significant: at most 9 of the 200 images change between right and wrong, and the exact McNemar p is 0.18 or more before any correction. That is no measurable effect on these images, not proof that the chip never degrades: in the Visual Wake Words run ([Results](../README.md#results), section 2) the same step on captured frames is −0.0097 (0.6892 → 0.6795, not significant either). F1 can hide changes that cancel out, so image-level agreement is the stricter check: SIL ↔ PIL on the 940 test files is 99.4%–99.6%, except `MobileNetV3Small` (95.96%, 38 images change), and SIL\* ↔ HIL on captured frames is 97.0%–100.0%.
- **Optical front end in FP32 (MIL₂₀₀ → MIL\*₂₀₀):** Replacing the file with the camera frame changes F1-macro by −0.1073 in `ResNet8` (Holm p < 0.001), −0.0454 in `MobileNet` (Holm p = 0.156), −0.0351 in `MobileNetV3Small` (Holm p = 0.156), −0.0357 in `MobileNetV2` (Holm p = 0.287) and 0.0000 in `SqueezeNet` (Holm p = 1.000). Only `ResNet8` is individually significant.
- **Physical gap on the chip (PIL₂₀₀ → HIL₂₀₀):** This is a different quantity, shown in the last column of the table: significant in `ResNet8` and `MobileNet`, and −0.0200 (not significant) in `MobileNetV3Small` and `SqueezeNet`. In `MobileNetV2` the gap is negative in all five sessions measured (−0.0509 to −0.0664), but whether it stays significant after the Holm correction depends on the session: Holm p = 0.022 with the session of the table and with `rep4`, and 0.080, 0.058 and 0.093 with `rep1`, `rep2` and `rep3` (Holm over the five architectures, the other four unchanged; unadjusted p from 0.007 to 0.031).
- **Bench repeatability (`results/hil/rep1..rep4/`, `MobileNetV2`):** Across 3 fixed-seed (`seed=42`) sessions under identical physical conditions, the PIL → HIL gap on the fixed 200-image subset is −0.0561, −0.0563, and −0.0509 (repeatability sd = 0.0030, CV = 5.6%, pairwise image agreement 96.5%–97.5%; sd = 0.0065 across all 4 runs when including `rep4`, the one session with another presentation order (`seed=7`), at −0.0664, with 189/200 stimuli or 94.5% receiving an identical prediction across all 4 sessions). The session of the tables, which used the order of the replicates and also streamed every frame back, gives −0.0664 as well, so the data do not single out the order as the cause of the wider spread. Combining 200-image subset sampling variance with bench repeatability yields an overall `MobileNetV2` frontend gap of **−0.057 [−0.098, −0.017] (95% CI)**.

## Statistical Tables

`results/tables/ladder_tables.md` (and `.tex`) holds the eight tables; each is also a CSV with unrounded values, and
`ladder_summary.json` records the headline numbers, the SHA-256 of every input file and the library versions.

| Table | File | Content |
|---|---|---|
| 1 | `table1_ladder.csv` | Fidelity ladder MIL → SIL → PIL → HIL (macro-F1) |
| 2 | `table2_decomposition.csv` | Paired decomposition input × engine on the 200 rig images |
| 3 | `table3_latency.csv` | Per-phase latency and memory on the Cortex-M7 |
| 4 | `table4_repeatability.csv` | HIL repeatability (`MobileNetV2`, same 200 images) |
| 5 | `table5_per_class.csv` | Per-class sensitivity |
| 6 | `table6_gap_contrasts.csv` | Between-architecture contrasts of the PIL → HIL gap |
| 7 | `table7_attribution.csv` | PIL attribution check: per-image agreement of each PIL run with the SIL of every model |
| 8 | `table8_representativeness.csv` | Representativeness of the rig subset (F1 on 200 − F1 on 940) |

The statistics are paired bootstrap 95% intervals (10,000 resamples, seed 20261016), exact McNemar tests with
Holm correction and per-class Fisher tests; the script is deterministic. The tables are rebuilt in about 10
seconds without a board, a dataset or trained models. This writes to a git-ignored folder, and the second command
says whether the rebuilt tables are the published ones:

```bash
python src/build_ladder_tables.py --output_dir results/smoke_test/tables
python tests/tables_check.py results/tables results/smoke_test/tables
```

`results/tables/independence_plates.csv` recomputes the 24 runs without the test images shared with the training
data (see [`data_independence.md`](data_independence.md#plates-secondary-benchmark)).

## The Same Model on a Second Board

The first run of the PIL bench on an Arduino Nicla Vision (14 November 2026, `results/nicla/pil/`) used the
`MobileNetV2` of this benchmark. The board answered the 940 test images, with the same prediction as a Portenta H7
measured the same day for all of them and 19.7% fewer inference cycles. That Portenta H7 run is not among the
published files: against the one in `results/pil/`, measured in September, the Nicla Vision run agrees on 937 of
the 940 images and takes 19.3% fewer cycles. The three images that differ come from the host, not from the board:
the pixels it decoded from three JPEG files were no longer the same (see
[Boards and Memory](hardware.md#boards-and-memory) and [Serial Protocol](hardware.md#serial-protocol-pil-bench)).
The two boards carry the same microcontroller, so this is a consistency check between boards, not evidence of
portability to another microcontroller family. The COCO benchmark was then measured on both boards
([Results](../README.md#results), section 1).

## Limitations Specific to This Benchmark

- **Test split not independent of the training data.** Of the 940 test images, 12 have the same pixels as a
  train/val image, 726 are near-duplicates of one, 96 are in the review band and 118 are distinct; in the
  200-image `hil_subset`, 148 are near-duplicates, 25 in the review band and 27 distinct. The clean subset is too
  small to re-establish the results on independent images. Method, per-run effect and reading are in
  [`data_independence.md`](data_independence.md#plates-secondary-benchmark).
- **Private images.** No level can be run again by someone who does not hold the photographs; only the tables can
  be rebuilt, from the per-image files.
- **Models are files, not recipes.** The models were trained before the training scripts had a seed, so training
  again does not give the same weights. Their `SqueezeNet` was trained without batch normalisation, which the
  COCO benchmark needed.
- **Two decoders.** MIL and SIL were measured with TensorFlow decoding the JPEG files and PIL with Pillow, which do
  not give the same pixels, so the SIL-to-PIL differences mix the decoder with the chip. Re-running the models
  through the one loader the scripts use today (`src/images.py`) changes the prediction of 0 to 7 of the 940 test
  images at MIL and of 4 to 33 at SIL, depending on the model. The files are kept as measured, and a re-run of
  one of these MIL or SIL results differs from them by that decoder effect. The partition stays JPEG.
- **Default TFLite interpreter at SIL.** The SIL runs used the default interpreter of TFLite (XNNPACK delegate),
  whose kernels do not give the chip's prediction for every image: on the COCO benchmark it differs from the chip
  in 27 to 80 of 5,000 images, for the same files on which the reference kernels of TFLite differ in 1 to 25
  ([Results](../README.md#results), section 1). Part of what the SIL-to-PIL comparison shows here is therefore the
  PC, not the chip.
- **PIL files without the on-chip model verification.** The PIL runs were measured when `pil_benchmark.py` hashed
  the `.tflite` passed with `--model_path` on the host and never compared it with the board, so their `sha256`
  does not prove which model was flashed and they carry no `model.identity_verified`. A board running another
  model than the one named is ruled out only indirectly: each architecture has its own inference cycle count, and
  in Table 7 every PIL run agrees most with the SIL of its own model (96.0% to 99.6% of the 940 images, against
  91.3% to 95.7% with the SIL of any other model). The HIL sessions did verify the fingerprint of the model on the
  board.
- **Session-to-session variation of the HIL gap, known from few runs:** The three strict replicates (`seed=42`) give sd = 0.0030 F1-macro on the fixed 200-image subset. The one session with another presentation order (`rep4`, `seed=7`) gives −0.0664 against their mean of −0.0544 (4-run sd = 0.0065), but so does the session of the tables, which used the order of the replicates (it was measured two hours earlier and streamed every frame back): with one run of each kind, the data do not separate an effect of the order from the variation between sessions. Accounting for both 200-image sampling variance and bench repeatability yields the conservative frontend interval of −0.057 [−0.098, −0.017] (95% CI).
- **Non-interchangeability of PIL and HIL inference cycle counts:** Across all evaluated models, `CYC_INF` in HIL is systematically 0.81%–2.77% lower than in PIL despite an identical `400.0 MHz` clock and SDRAM arena layout (cause not identified: arena and cache configuration are the same in both firmwares). PIL and HIL latency baselines should therefore be reported separately rather than mixed.
- **One board and one rig.** Every PIL and HIL result of the tables comes from one Arduino Portenta H7 with the
  Vision Shield, in one rig configuration (10 lux, 24 cm, 24 °C, `--settle 1.5`).

## Data Availability

- **In the repository:** the partition manifests, the split report and the near-duplicate tables
  (`data/splits_plates/`: file names and scores, no image), every per-image prediction and latency table of the
  runs named without a tag (`results/{mil,sil,pil,hil}/`, `results/hil/rep1..rep4/`, and the `MobileNetV2` run in
  `results/nicla/pil/`), and the tables (`results/tables/`).
- **Model files:** distributed like the other published models (see
  [Data Availability](../README.md#data-availability) in the README); `python src/verify_models.py --set plates`
  checks them against `models/checksums.sha256`.
- **Licence-plate photographs:** withheld. The photographs of this benchmark show identifiable licence plates,
  which are personal data under Colombian data-protection law (**Law 1581 of 2012**), so they are neither in this
  repository nor in its releases. They are available from the author upon reasonable academic request, subject to
  data-protection compliance.
