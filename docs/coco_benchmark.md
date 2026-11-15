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

PIL and HIL use the commands of steps 5 to 7 of [Usage](../README.md#usage-full-pipeline) with `--splits_dir data/splits_coco`, on either board. The published PIL files of these models are protected against replacement, so add `--output_dir` (or `--overwrite`). A PIL run of the 5,000 images takes from 65 minutes (`MobileNet` on the Nicla Vision) to almost 5 hours (`ResNet8` on the Portenta H7). Which levels have been measured on this benchmark so far is stated in the first note of [Results](../README.md#results), section 1, in the README.

### Training

To train the models instead of downloading them, this is the command of the published `MobileNetV2` (the options are explained in step 2 of [Usage](usage.md#2-model-training)); use a tag of your own instead of `COCO`, so that your runs do not take the names of the published ones (the scripts refuse to replace their files):
```bash
python src/train_mobilenet.py --base_model MobileNetV2 --trainable_backbone --width 160 --height 120 --batch_size 32 --epochs 50 --learning_rate 0.0003 --lr_schedule stepped --augmentation light --seed 42 --splits_dir data/splits_coco --tag COCO
```
The other six differ only in the script and its first options: `train_mobilenet.py --base_model MobileNet`, `MobileNetV3Small` or `MobileNetV3Large` with `--trainable_backbone`; `train_resnet.py --base_model ResNet8` or `ResNet18`; and `train_squeezenet.py --batch_norm`. Each run writes its recipe, per-epoch history and software versions to `{model}.training.json` next to the checkpoint. The seven trainings took 5.4 hours on one GPU (RTX 2080 SUPER). The seed repeats a run exactly on a CPU but not on a GPU, so training again gives other weights and slightly other numbers: the published numbers are reproduced with the published model files.

## Notes on the Results

These are the notes on the two tables of [Results](../README.md#results), section 1, in full. The first note of the
README, on the levels that have been measured, is kept there only.

- **The two boards give the same answers.** For each of the five models that fit, the Portenta H7 and the Nicla Vision give the same prediction for all 5,000 images. Every run answered the 5,000 images without a dropped serial link.
- **SIL = PIL when the PC uses the same arithmetic.** MIL, SIL and the PIL bench read each image through one loader, and the partition is lossless PNG, so the three levels receive the same bytes. The SIL column uses the reference kernels of TFLite (`sil_benchmark.py --kernels reference`), the ones TFLite-Micro is derived from. With them the PC and the chip agree on 4,999 of 5,000 images for `MobileNetV2` (the one that differs is an exact tie at the output), 4,996 for `SqueezeNet`, 4,993 for `ResNet8` and 4,975 for `MobileNet`. The default interpreter of TFLite (XNNPACK delegate), which every earlier result of this repository was measured with, differs from the chip in 27 to 80 images of the same files, and the built-in kernels without the delegate in 9 to 53: most of what a SIL-to-PIL comparison shows with the default interpreter is the PC, not the chip. The runs with the other kernels are in `results/sil_kernels/`, and the second table of `benchmark_COCO.md` compares them. On the chip, the convolutions, fully connected layers, additions, multiplications, pooling and softmax run the CMSIS-NN kernels of the library, whose headers describe them as bit-accurate with TensorFlow Lite Micro (see [Boards and Memory](hardware.md#boards-and-memory)); these figures are a check of that on real hardware.
- **`MobileNetV3Small` is the exception.** It loses five points at quantization (0.8388 → 0.7856), the chip and the reference-kernel SIL agree on 4,754 of 5,000 images only, and its predictions depend on the kernels (324 images differ between the built-in and the reference kernels). One of its squeeze-excite activations is zero for all 3,000 images sampled from the three splits (it is nearly so in the ImageNet weights already) and gets a degenerate scale of 7.8e-9 at conversion. The default interpreter of the PC refuses the file for that reason; the reference kernels and both boards run it. Its agreement figures say more about that file than about the chip.
- **The Nicla Vision is faster.** With the same MCU, clock and model file it needs 11.4% to 19.9% fewer cycles per inference than the Portenta H7. Its tensor arena is in internal RAM and the Portenta's in external SDRAM, which is the likely cause; it has not been isolated. `SqueezeNet` and `ResNet8` need a larger arena than the default on the Nicla Vision (`--arena_kb 408` and `392`), which leaves 27 KB and 43 KB of its RAM for everything else; both ran the 5,000 images.
- **Two models stop at SIL.** The INT8 files of `MobileNetV3Large` (2,179.5 KiB) and `ResNet18` (11,057.3 KiB) are larger than the flash left for a model (about 1,374 KiB on the Portenta H7, see [Flash Budget](hardware.md#flash-budget)).
- **`SqueezeNet` is trained with batch normalisation** (`train_squeezenet.py --batch_norm`). The network as trained for the vehicle benchmark does not learn on COCO: with three learning rates it answers one class for every image (accuracy 0.500). The batch-normalisation layers are folded into the convolutions at conversion, so the INT8 file has the same operators and is 1,200 bytes larger.
- **Test split and training data.** The training and validation images come from COCO `train2017` and the test images from `val2017`, so no file is on both sides. Different files can still show the same scene: after looking at every pair flagged by `src/find_near_duplicates.py`, 129 of the 5,000 test images (2.6%) show the same photo or scene as a train/val image (one of them in the `hil_subset`), and removing them changes macro-F1 by at most 0.0021 in any run. Method, per-run figures and limits are in [`data_independence.md`](data_independence.md#coco-main-benchmark).
- **Where these numbers stand.** Published results on this task, trained on the same amount of data, report about 86% for a MobileNetV1 (α=0.25) at 96×96 in colour (the MLPerf Tiny reference model) and 88.75% for a MobileNetV2 at 160×160 in grayscale (MicroNets). The models here take 160×120 grayscale images and one recipe was used for the seven; no model was tuned on the test partition.

## The Published PIL Files and the Model on the Board

The limitations of the framework and of its published results are listed under
[Known Limitations](../README.md#known-limitations) in the README. This one is given there in short:

- **The published PIL files predate the on-chip model verification in PIL:** Their `sha256` is that of the `.tflite` given to the bench on the host, which was not compared with the board, so it does not prove which model was flashed and the files carry no `model.identity_verified`. A board running another model than the one named is ruled out for them indirectly: each PIL run is compared, image by image, with the SIL run of every model of its benchmark, and its own model must be the best match by a wide margin. For the COCO runs on both boards that is the table "Which model each PIL run agrees with" of `results/tables/benchmark_COCO.md` (`results/tables/benchmark_COCO_attribution.csv`); for the runs of the vehicle application case it is Table 7 of its tables ([`vehicle_benchmark.md`](vehicle_benchmark.md#statistical-tables)). Runs measured from now on are checked directly, by the fingerprint the PIL firmware announces before the first image (checked on both boards, see [Serial Protocol](hardware.md#serial-protocol-pil-bench)).
