# Usage: the Full Pipeline in Detail

The [README](../README.md#usage-full-pipeline) gives the seven steps as one short sequence of commands. This page goes through the same steps with their options, what each script checks before it runs and what it leaves behind, and ends with the two board levels on the Arduino Nicla Vision and a note on the two requirements files. The reference of every script is [`scripts.md`](scripts.md).

Every step needs your own dataset in `data/raw/`; steps 6 and 7 also need the board. The four levels share their options (`--model_path`, `--splits_dir`, `--output_dir`, `--overwrite`), and every multi-word option can be written with `_` or `-`. Extra options and details are in the [usage notes](scripts.md#usage-notes). The published COCO benchmark has its own recipe with the exact commands: see [`coco_benchmark.md`](coco_benchmark.md#reproducing-the-benchmark).

No dataset at hand? The public Visual Wake Words subset can stand in for it. This downloads 2,000 images into `data/raw/person` and `data/raw/non_person` in about a minute, and every command below then works unchanged:
```bash
python src/download_vww_subset.py --per_class 1000 --seed 42 --output_dir data/raw
```

## 1. Data Preparation
Place your raw image dataset inside `data/raw/` (organized by class subfolders):
```
data/raw/
├── Class1/
└── Class2/
```

**Supported formats.** `.jpg`, `.jpeg`, `.png` and `.bmp`, any resolution and colour
mode: `process_images.py` converts every image to 8-bit single-channel grayscale and
`reshape_images.py` resizes it to the target resolution (extensions are matched
case-insensitively, so `.JPG` or `.Png` also count). Each subfolder of `data/raw/` is one class, and
labels follow the alphabetical order of the folder names. Files with other
extensions (e.g. `.gif`, `.tif`, `.webp`, `.heic`) are skipped.

Then process (convert to grayscale) and resize them with recommended sizes: CAMERA_R160x120 (QQVGA), CAMERA_R320x240 (QVGA), CAMERA_R320x320.
```bash
python src/process_images.py --raw_path data/raw --path_processed data/processed/grayscale
python src/reshape_images.py --width 160 --height 120 --balance undersample
```

Finally, partition the resized dataset. This step is **required** before training, because it guarantees that every level evaluates exactly the same held-out images:
```bash
python src/split_dataset.py --input_dir data/processed/160x120 --output_dir data/splits --seed 42
```
`data/splits/` is a git-ignored working folder for your own partition. The partitions behind the published results are versioned in separate folders (`data/splits_coco/`, `data/splits_plates/`, `data/splits_vww/`) and these commands do not touch them. If your test images come from a separate source, give that folder with `--test_input_dir`: all of it becomes the test partition and `--input_dir` only feeds `train` and `val`.

## 2. Model Training
Train an architecture (e.g., MobileNetV2). Training reads `data/splits/train` and `data/splits/val`, and fails with instructions if the partition is missing.
```bash
python src/train_mobilenet.py --base_model MobileNetV2 --width 160 --height 120 --batch_size 32 --epochs 20 --learning_rate 0.0001 --tag demo
```
For ResNet or SqueezeNet, run `src/train_resnet.py` or `src/train_squeezenet.py`.

That command is the default recipe: a frozen ImageNet backbone with a new classifier, a constant learning rate and early stopping. Three options give a full-length training like the reference trainings of person detection on COCO: `--lr_schedule stepped` runs every epoch and halves the rate at 40% and at 60% of them, `--augmentation light` uses a milder augmentation (flip, rotation of 10°, shifts of 5%, zoom of 10%), and `--trainable_backbone` trains every layer of a MobileNet from the ImageNet weights:
```bash
python src/train_mobilenet.py --base_model MobileNetV2 --trainable_backbone --width 160 --height 120 --batch_size 32 --epochs 50 --learning_rate 0.0003 --lr_schedule stepped --augmentation light --tag demo_full
```
The run name does not carry these options, so give each recipe its own `--tag`. What each option does, and what it gave in a pilot on COCO, is in [`docs/scripts.md`](scripts.md#2-model-training-ml-pipelines).

`--tag` names the run: the checkpoint is `MobileNetV2_demo+32+20+0.0001+160+120.keras`, and the `.tflite` and every result file of that model inherit the name. Use a different tag for each dataset. Without a tag the run takes the names of the models of the vehicle application case ([`vehicle_benchmark.md`](vehicle_benchmark.md)), and the scripts stop instead of replacing an existing checkpoint, `.tflite` or result file; `--overwrite` allows it. The label order is saved next to the checkpoint (`.classes.txt`).

`--seed` (default 42) fixes the initial weights, the shuffling and the augmentation, and every run leaves `{model}.training.json` next to its checkpoint with the seed, the recipe, the per-epoch history and the software versions. With the same seed, two runs on a CPU gave bit-identical weights; on the GPU they did not, because some GPU kernels are not deterministic (a `ResNet8` trained from scratch differed from the first epoch, a `MobileNetV2` head only in the seventh decimal of its history). Compare the history in that record instead of assuming that two runs are equal; the measurements are in [`docs/scripts.md`](scripts.md#2-model-training-ml-pipelines).

## 3. Model-in-the-Loop (MIL) Evaluation
The floating-point `.keras` model runs on the PC over the held-out test partition. Outputs go to `results/mil/`:
```bash
python src/mil_benchmark.py --width 160 --height 120 --model_path models/checkpoints/MobileNetV2_demo+32+20+0.0001+160+120.keras
```

## 4. INT8 Quantization and Software-in-the-Loop (SIL) Evaluation
Convert the model to integer-only TFLite. Calibration samples come from `data/splits/train` only, so the evaluation partition stays untouched:
```bash
python src/quantize_int8_basic.py --model_path models/checkpoints/{model_name}.keras --calib_samples 100 --calib_seed 42
```
The INT8 `.tflite` model runs on the PC over the same test partition. Outputs go to `results/sil/`:
```bash
python src/sil_benchmark.py --model_path models/tflite/{model_name}_int8.tflite
```

## 5. Model Header for the Firmware
Convert the `.tflite` into a C header. It is written into both firmware sketches (`deployment/pil_firmware/` and `deployment/hil_firmware/`), so PIL and HIL run the same model:
```bash
python src/tflite_to_c.py models/tflite/{model_name}_int8.tflite
```
The two `model.h` files in the repository carry the `MobileNetV2` of the vehicle application case, the model of the build record [`flash_budget_build.txt`](flash_budget_build.txt); this command replaces them, so `git status` reports them as modified.

The board runs one firmware at a time: steps 6 and 7 each flash their own and then run their bench. `compile_upload_arduino.py` finds the board by itself; the benches take its port with `--port`, which `arduino-cli board list` shows once the firmware is running (it is not the port of the bootloader). If an upload fails, see [Upload Troubleshooting](hardware.md#upload-troubleshooting).

## 6. Processor-in-the-Loop (PIL) Evaluation
The real chip runs inference on test images sent over USB serial; the camera is not part of the loop.

Flash the PIL firmware:
```bash
python src/compile_upload_arduino.py --target pil
```
Run the bench. It streams the held-out test partition and writes to `results/pil/`:
```bash
python src/pil_benchmark.py --port COM9 --model_path models/tflite/{model_name}_int8.tflite --width 160 --height 120
```
Before the first image the bench asks the board what it is running, and it stops, with nothing written, unless the board is the one given with `--board`, the model in flash is the `.tflite` given with `--model_path` (same FNV-1a fingerprint and size) and the images have the input shape the firmware announces. `--width` and `--height` may be left out: they are then taken from the firmware. A board that stops answering ends the run after 5 images in a row without an answer (`--max_unanswered`) instead of waiting 30 s for each remaining image. A run that does not reach its last image keeps what it measured, marked `completed: false` in `metrics_*.json`, and ends with exit code 3. `--allow_model_mismatch` measures on a board whose model cannot be verified, which is the case of a firmware flashed before November 2026 (it reports no fingerprint), and records `model.identity_verified = false`.

## 7. Hardware-in-the-Loop (HIL) Evaluation
The real chip runs inference on frames from its own HM01B0 camera, which looks at each test image displayed full-screen on a monitor.

Flash the HIL firmware:
```bash
python src/compile_upload_arduino.py --target hil
```
Run the bench. It displays the `hil_subset` of the test partition and writes to `results/hil/`:
```bash
python src/hil_benchmark.py --port COM9 --model_path models/tflite/{model_name}_int8.tflite --lux 10 --distance_cm 24 --ambient_temp 24
```
The bench aborts if the fingerprint of the model on the board does not match `--model_path`, or if a rig condition (`--lux`, `--distance_cm`, `--ambient_temp`) is missing. Each stimulus stays on screen for `--settle` seconds before its capture: the default is 1.5 s, the value of the sessions of the vehicle application case (the Visual Wake Words session used 1.0 s). It waits for ENTER before the first stimulus, so the camera can be placed in front of the monitor; `--no_confirm` skips the wait, and is needed where there is no interactive terminal. Adding `--sil_model models/tflite/{model_name}_int8.tflite --mil_model models/checkpoints/{model_name}.keras` also classifies every captured frame on the PC, which separates the effect of the camera from the effect of the chip.

Before a session, `python src/calibrate_hil_rig.py --port COM9 --live` shows a test pattern and says how to move the camera until it frames the monitor, and `--report` writes a calibration report (geometry, tone response, vignetting, exposure stability and focus) to `results/hil/calibration/`. A session that is cut short keeps what it measured: every stimulus is written to the CSV as it is measured, and the other files are written marked `completed: false`, with exit code 3 (see [`docs/scripts.md`](scripts.md#5-camera-in-the-loop-hil-evaluation)).

## Running PIL and HIL on the Arduino Nicla Vision
The same two sketches build for the Nicla Vision; `--board nicla` selects it in the three scripts that deal with the board. Steps 1 to 5 do not change.

Flash the PIL firmware and run its bench:
```bash
python src/compile_upload_arduino.py --target pil --board nicla
```
```bash
python src/pil_benchmark.py --board nicla --port COM9 --model_path models/tflite/{model_name}_int8.tflite --width 160 --height 120
```
Flash the HIL firmware and run its bench:
```bash
python src/compile_upload_arduino.py --target hil --board nicla
```
```bash
python src/hil_benchmark.py --board nicla --port COM9 --model_path models/tflite/{model_name}_int8.tflite --lux 10 --distance_cm 24 --ambient_temp 24
```
The results go to `results/nicla/pil/` and `results/nicla/hil/`, and every file records the board it was measured on (`run.board`). The benches stop if the firmware announces a board other than the one given with `--board`.

Two things differ from the Portenta H7 (details in [Hardware and Porting](hardware.md#boards-and-memory)):
- **Memory.** The Nicla Vision has no external SDRAM, so the tensor arena is a static array in its 511 KiB of internal RAM: 368 KiB by default, sized for `MobileNetV2`. `--arena_kb` changes it at compile time, and `--compile_only` reports the RAM use without a board.
- **Camera.** Its GC2145 is a colour sensor: the firmware reads RGB565 and converts it to 8-bit luminance (BT.601 weights, as Pillow does) before the same preprocessing as on the Portenta H7.

> **Status.** The PIL level of the COCO benchmark was measured on a Nicla Vision: the five models that fit, 5,000 test images each, with the same prediction as the Portenta H7 for every image and 11.4% to 19.9% fewer inference cycles (`results/nicla/pil/`; see [Results](../README.md#results), section 1, and [Boards and Memory](hardware.md#boards-and-memory)). Both boards carry the same STM32H747, so that agreement is a consistency check between two boards, not evidence of portability to other microcontroller families. The first run on the board, with the `MobileNetV2` of the vehicle application case, is described in [`vehicle_benchmark.md`](vehicle_benchmark.md#the-same-model-on-a-second-board). The HIL sketch builds for the board and has not been run on it yet.

## The Two Requirements Files
`requirements.txt` declares *minimum* compatible
versions (`>=`): it installs on other Python versions and operating systems, even with newer
releases, and is what the installation commands of the [README](../README.md#installation) use. `requirements-lock.txt` pins the
*exact* versions of the project virtual environment, the likely source of the published results (Python
3.13.14, TensorFlow 2.21.0, see its header): use `pip install -r requirements-lock.txt`
when you need to reproduce the reported numbers rather than just run the
framework. One file cannot do both: exact pins stop installing as soon as a
release disappears or is missing for another Python version, and minimums alone
cannot rebuild the exact environment. `scikit-learn` and `seaborn` are not optional:
the evaluation scripts need them to write `metrics_*.json` and the confusion
matrices. GPU training is only possible under WSL2 (`tensorflow[and-cuda]`, see the
note in `requirements.txt`); the serial benches stay on Windows, where the COM port is.
The firmware toolchain is described under [Requirements](../README.md#requirements).
