# The COCO Benchmark: `person` / `non_person` on MS-COCO 2017

This is the benchmark of the repository. The [README](../README.md#results) has its two result tables and a short
list of what they show. This page holds what the README leaves out: the complete recipe to reproduce the benchmark,
the notes on its results in full, and the limitation that concerns its PIL files. The partition and the check of
its independence are described in [`data_independence.md`](data_independence.md#coco-main-benchmark), and the full
table that the two tables of the README are condensed from is `results/tables/benchmark_COCO.md`.

- [Reproducing the Benchmark](#reproducing-the-benchmark)
- [Notes on the Results](#notes-on-the-results)
- [The Published PIL Files and the Model on the Board](#the-published-pil-files-and-the-model-on-the-board)

## Reproducing the Benchmark

The benchmark needs no private data, and its PC levels need no board. The README gives the three steps in short;
these are the same steps with every command, followed by the board levels and the training. From a fresh clone:

### 1. Rebuild the Dataset

All 118,287 images of COCO `train2017` feed `train` and `val` (100,544 and 17,743), and all 5,000 images of `val2017` are the test partition, so no file is on both sides of the evaluation. That is the amount of data the published results on this task are trained with. The labels follow the rule of the original Visual Wake Words script: `person` when a person annotation covers more than 0.5% of the image, and no image is left out. The two downloads take about 20 GB of disk and the later steps about 9 GB more; the partition itself, 160×120 grayscale images stored as lossless PNG, is 1.9 GB of them. As measured here: 33 minutes to download `train2017`, one and a half to three minutes for `val2017`, and about 23 minutes to prepare the images.
```bash
python src/download_vww_subset.py --coco_split train2017 --full_split --output_dir data/raw_coco_train
python src/download_vww_subset.py --coco_split val2017 --full_split --output_dir data/raw_coco_test
python src/process_images.py --raw_path data/raw_coco_train --path_processed data/processed/coco_train_gray
python src/process_images.py --raw_path data/raw_coco_test --path_processed data/processed/coco_test_gray
python src/reshape_images.py --width 160 --height 120 --format png --input_dir data/processed/coco_train_gray --output_dir data/processed/coco_train_160x120
python src/reshape_images.py --width 160 --height 120 --format png --input_dir data/processed/coco_test_gray --output_dir data/processed/coco_test_160x120
python src/split_dataset.py --input_dir data/processed/coco_train_160x120 --test_input_dir data/processed/coco_test_160x120 --train 0.85 --val 0.15 --test 0 --output_dir data/splits_coco --force
```
Each download prints, when it ends, the selection hash stored in `data/splits_coco/subset_report_train2017.json` and `subset_report_val2017.json` (`d12ce39deb42` and `a0ed2d0b8236`); with `--dry_run` it gives the image counts without downloading anything. The last command ends with `same partition as the manifests already in data/splits_coco`, and `git status` shows no change: the rebuilt partition is the published one. Checked from a fresh clone, the rebuilt image files were also identical, byte for byte, to those the published runs were measured on.

### 2. Get the Published Models

The seven `.keras` checkpoints and their `_int8.tflite` files, named `{arch}_COCO+32+50+0.0003+160+120`, are distributed with the release, which has not been published yet; until it is, they can be requested from the author (see [Data Availability](../README.md#data-availability)). Place them in `models/checkpoints/` and `models/tflite/`, each with its `.classes.txt` file when it comes with one (it holds the label order; without it the benches take the class names from the folders of the test partition and say so); `python src/verify_models.py --set coco` checks them against `models/checksums.sha256`, which holds the digest recorded in every result file.

### 3. Run the PC Levels and Compare

The runs write to a git-ignored folder, and `compare_results.py` checks them against the published files: same model file, same metrics, same confusion matrix and the same prediction for every image. The published SIL runs use the reference kernels of TFLite (see [Notes on the Results](#notes-on-the-results)):
```bash
python src/mil_benchmark.py --width 160 --height 120 --model_path models/checkpoints/MobileNetV2_COCO+32+50+0.0003+160+120.keras --splits_dir data/splits_coco --output_dir results/smoke_test/mil
python src/sil_benchmark.py --model_path models/tflite/MobileNetV2_COCO+32+50+0.0003+160+120_int8.tflite --splits_dir data/splits_coco --kernels reference --output_dir results/smoke_test/sil
python src/compare_results.py results/smoke_test results
python src/build_benchmark_table.py --tag COCO --output_dir results/smoke_test/tables
```
The last command rebuilds the table of this benchmark (`results/tables/benchmark_COCO.md` and `.csv`) from the published result files alone; `git diff --no-index --ignore-cr-at-eol results/tables/benchmark_COCO.csv results/smoke_test/tables/benchmark_COCO.csv` prints nothing when the rebuilt table is the published one. Checked from a fresh clone: the fourteen MIL and SIL runs came out identical to the published ones, image by image, in about half an hour.

### Board Levels

PIL and HIL use the commands of steps 5 to 7 of [Usage](../README.md#usage-full-pipeline) with `--splits_dir data/splits_coco`, on either board. The published PIL files of these models are protected against replacement, so add `--output_dir` (or `--overwrite`). A PIL run of the 5,000 images takes from 65 minutes (`MobileNet` on the Nicla Vision) to almost 5 hours (`ResNet8` on the Portenta H7). The published HIL sessions are those of the Portenta H7 on the 200 images of the `hil_subset` ([Results](../README.md#results), section 1).

### Training

To train the models instead of downloading them, this is the command of the published `MobileNetV2` (the options are explained in step 2 of [Usage](usage.md#2-model-training)); use a tag of your own instead of `COCO`, so that your runs do not take the names of the published ones (the scripts refuse to replace their files):
```bash
python src/train_mobilenet.py --base_model MobileNetV2 --trainable_backbone --width 160 --height 120 --batch_size 32 --epochs 50 --learning_rate 0.0003 --lr_schedule stepped --augmentation light --seed 42 --splits_dir data/splits_coco --tag COCO
```
The other six differ only in the script and its first options: `train_mobilenet.py --base_model MobileNet`, `MobileNetV3Small` or `MobileNetV3Large` with `--trainable_backbone`; `train_resnet.py --base_model ResNet8` or `ResNet18`; and `train_squeezenet.py --batch_norm`. Each run writes its recipe, per-epoch history and software versions to `{model}.training.json` next to the checkpoint. The seven trainings took 5.4 hours on one GPU (RTX 2080 SUPER). The seed repeats a run exactly on a CPU but not on a GPU, so training again gives other weights and slightly other numbers: the published numbers are reproduced with the published model files.

## Notes on the Results

These are the notes on the two tables of MIL, SIL and PIL of [Results](../README.md#results), section 1, in full.
The notes on the HIL table are in the README only.

- **The two boards give the same answers.** For each of the five models that fit, the Portenta H7 and the Nicla Vision give the same prediction for all 5,000 images. Every run answered the 5,000 images without a dropped serial link.
- **SIL = PIL when the PC uses the same arithmetic.** MIL, SIL and the PIL bench read each image through one loader, and the partition is lossless PNG, so the three levels receive the same bytes. The SIL column uses the reference kernels of TFLite (`sil_benchmark.py --kernels reference`), the ones TFLite-Micro is derived from. With them the PC and the chip agree on 4,999 of 5,000 images for `MobileNetV2` (the one that differs is an exact tie at the output), 4,996 for `SqueezeNet`, 4,993 for `ResNet8` and 4,975 for `MobileNet`. The default interpreter of TFLite (XNNPACK delegate), which the earlier Visual Wake Words run was measured with, differs from the chip in 27 to 80 images of the same files, and the built-in kernels without the delegate in 9 to 53: most of what a SIL-to-PIL comparison shows with the default interpreter is the PC, not the chip. The runs with the other kernels are in `results/sil_kernels/`, and the second table of `benchmark_COCO.md` compares them. On the chip, the convolutions, fully connected layers, additions, multiplications, pooling and softmax run the CMSIS-NN kernels of the library, whose headers describe them as bit-accurate with TensorFlow Lite Micro (see [Boards and Memory](hardware.md#boards-and-memory)); these figures are a check of that on real hardware.
- **`MobileNetV3Small` is the exception.** It loses five points at quantization (0.8388 → 0.7856), the chip and the reference-kernel SIL agree on 4,754 of 5,000 images only, and its predictions depend on the kernels (324 images differ between the built-in and the reference kernels). One of its squeeze-excite activations is zero for all 3,000 images sampled from the three splits (it is nearly so in the ImageNet weights already) and gets a degenerate scale of 7.8e-9 at conversion. The default interpreter of the PC refuses the file for that reason; the reference kernels and both boards run it. Its agreement figures say more about that file than about the chip.
- **The Nicla Vision is faster.** With the same MCU, clock and model file it needs 11.4% to 19.9% fewer cycles per inference than the Portenta H7. Its tensor arena is in internal RAM and the Portenta's in external SDRAM, and that is the cause. On the 200 images of the HIL subset, the Portenta H7 runs an inference of `MobileNet` in 193.85 ms with the arena in internal RAM (`compile_upload_arduino.py --arena_memory internal`) and in 237.29 ms with the same sources and the arena in SDRAM; the Nicla Vision takes 193.46 ms and the published Portenta H7 run, an earlier build, 241.40 ms. The external SDRAM costs that model about 18% of its inference time, and moving the arena leaves the two boards 0.2% apart (`results/experiments/arena_internal/pil_cycles_*.md`, built by `compare_pil_cycles.py`). Only `MobileNet` was measured this way. `SqueezeNet` and `ResNet8` need a larger arena than the default on the Nicla Vision (`--arena_kb 408` and `392`), which leaves 27 KB and 43 KB of its RAM for everything else; both ran the 5,000 images.
- **Two models stop at SIL.** The INT8 files of `MobileNetV3Large` (2,179.5 KiB) and `ResNet18` (11,057.3 KiB) are larger than the flash left for a model (about 1,374 KiB on the Portenta H7, see [Flash Budget](hardware.md#flash-budget)).
- **`SqueezeNet` is trained with batch normalisation** (`train_squeezenet.py --batch_norm`). Without it the network does not learn on COCO: with three learning rates it answers one class for every image (accuracy 0.500). The batch-normalisation layers are folded into the convolutions at conversion, so the INT8 file has the same operators and is 1,200 bytes larger.
- **Test split and training data.** The training and validation images come from COCO `train2017` and the test images from `val2017`, so no file is on both sides. Different files can still show the same scene: after looking at every pair flagged by `src/find_near_duplicates.py`, 129 of the 5,000 test images (2.6%) show the same photo or scene as a train/val image (one of them in the `hil_subset`), and removing them changes macro-F1 by at most 0.0021 in any run. Method, per-run figures and limits are in [`data_independence.md`](data_independence.md#coco-main-benchmark).
- **Where these numbers stand.** Published results on this task, trained on the same amount of data, report about 86% for a MobileNetV1 (α=0.25) at 96×96 in colour (the MLPerf Tiny reference model) and 88.75% for a MobileNetV2 at 160×160 in grayscale (MicroNets). The models here take 160×120 grayscale images and one recipe was used for the seven; no model was tuned on the test partition.

## The Earlier Visual Wake Words Run

Section 2 of [Results](../README.md#results) in the README gives this run in three sentences; this is its paragraph and its table in full.

This smaller run predates the COCO benchmark. Its 2,000 images were drawn from `val2017`, which is the COCO test split, so a `_VWW` model must not be evaluated on the COCO test (the overlap is described in [`docs/data_independence.md`](data_independence.md#coco-main-benchmark)). To demonstrate dataset-agnostic pipeline reuse on a standard *MLPerf Tiny* task (`person` vs. `non_person` from MS-COCO `val2017`, 2,000 balanced images prepared via `src/download_vww_subset.py` after excluding 226 ambiguous images with `< 0.5%` person area or `iscrowd == 1`; `data/splits_vww/test` n = 300, `hil_subset == 1` n = 200 at 14 lux, 24 cm, 24 °C; linear classification head trained for 20 epochs over the frozen ImageNet `MobileNetV2` α = 0.35 backbone on 1,400 grayscale 160×120 images as a pipeline-reuse demonstration rather than a leaderboard-tuned VWW model):

| Scope | Level / Domain | Accuracy | F1-macro | Latency `INF` / `TOTAL` (ms) | Notes |
|---|---|---:|---:|---:|---|
| **Full test (n=300)** | **MIL** (FP32, PC) | 0.6900 | 0.6892 | — | `2,220.7 KiB` `.keras` |
| **Full test (n=300)** | **SIL** (INT8, PC) | 0.6767 | 0.6765 | — | `630.5 KiB` `.tflite` (3.52× smaller) |
| **Full test (n=300)** | **PIL** (INT8, Portenta H7) | 0.6867 | 0.6865 | 399.06 / 402.26 | `285/300` (95.0%) agreement with SIL |
| **Paired 2×3 (n=200)** | MIL₂₀₀ → SIL₂₀₀ → PIL₂₀₀ (file) | 0.7300 → 0.7150 → 0.7200 | 0.7290 → 0.7150 → 0.7196 | 399.06 / 402.26 | Δ(SIL₂₀₀ → PIL₂₀₀) = +0.0046 |
| **Paired 2×3 (n=200)** | MIL\*₂₀₀ → SIL\*₂₀₀ → HIL₂₀₀ (HM01B0) | 0.6750 → 0.6900 → 0.6800 | 0.6748 → 0.6892 → 0.6795 | 395.84 / 400.99 | **`196/200` (98.0%) SIL\* = HIL agreement** |

Three things limit what this table says:

- **It was measured before the single image loader.** MIL and SIL decoded the JPEG files with TensorFlow and the PIL bench with Pillow, which do not give the same pixels, and SIL ran on the default TFLite interpreter (XNNPACK delegate), whose kernels do not always give the chip's prediction. Its SIL-to-PIL difference therefore mixes the decoder and the PC kernels with the chip. Re-running the model through the one loader the scripts use today (`src/images.py`) changes the prediction of 5 of the 300 test images at MIL and of 14 at SIL, so a re-run of these two results differs from the published files by that much. The files are kept as measured, and the partition stays JPEG ([usage notes](scripts.md#usage-notes)).
- **Its PIL file does not prove which model was on the board.** It was measured before the PIL firmware announced the fingerprint of its model, and with a single model there is no SIL run of another model to compare it with (see [below](#the-published-pil-files-and-the-model-on-the-board)). The HIL session did verify the fingerprint.
- **One training and one session.** The model was trained before the training scripts had a seed, so training again does not give the same weights, and the HIL figures come from one session on one Portenta H7, with each stimulus 1.0 s on screen before its capture.

## The Published PIL Files and the Model on the Board

The limitations of the framework and of its published results are listed under
[Known Limitations](../README.md#known-limitations) in the README. This one is given there in short:

- **The published PIL files predate the on-chip model verification in PIL:** Their `sha256` is that of the `.tflite` given to the bench on the host, which was not compared with the board, so it does not prove which model was flashed and the files carry no `model.identity_verified`. A board running another model than the one named is ruled out for the COCO runs indirectly: each PIL run is compared, image by image, with the SIL run of every model of the benchmark, and its own model must be the best match by a wide margin. That is the table "Which model each PIL run agrees with" of `results/tables/benchmark_COCO.md` (`results/tables/benchmark_COCO_attribution.csv`), for both boards, and Table 7 of the ladder tables (`results/tables/coco/`, `results/nicla/tables/coco/`). The PIL file of the Visual Wake Words run, a run with one model, has no such comparison. Runs measured from now on are checked directly, by the fingerprint the PIL firmware announces before the first image (checked on both boards, see [Serial Protocol](hardware.md#serial-protocol-pil-bench)).
