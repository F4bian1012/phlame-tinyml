# Extending PHLAME

How to add an architecture and how to add a board. Both are written from what the code requires today, and each
section ends with what was checked by doing it and what only by reading the code.

A dataset of your own needs no change to the code: follow [Usage](../README.md#usage-full-pipeline) in the README
with your images and a `--tag`.

- [Adding an Architecture](#adding-an-architecture)
- [Adding a Board](#adding-a-board)

---

## Adding an Architecture

Nothing after training knows the list of architectures: quantization, the four levels and the tables work on a
model file and on its run name. An architecture is therefore added in the training script, and the rest of this
section is what the later steps expect from the model it saves.

### What Every Level Expects from a Model

| Property | Why |
|---|---|
| Input `(height, width, 1)`, raw pixels in `[0, 255]` | Training, the calibration of the quantizer and MIL read grayscale images through `src/images.py` and feed them unscaled; SIL, the PIL bench and the firmware quantize the same raw pixel with the scale and zero point of the input tensor. |
| The normalisation is a layer of the model | For the same reason: no level scales the pixels before the model. The three families do it with `Rescaling` or with the `preprocess_input` of the backbone. |
| Output: one probability per class (`softmax`), in label order | Every level, the firmware included, takes the `argmax`. The label order is the alphabetical order of the class folders and travels with the model in `{run}.classes.txt`. |
| Only layers that convert to integer-only TFLite | `quantize_int8_basic.py` converts with `TFLITE_BUILTINS_INT8` and INT8 input and output; a layer without an integer kernel stops the conversion. |
| Saved as `.keras` with layers Keras can load back | The quantizer and MIL load the checkpoint from the file. |
| Augmentation layers only for the `standard` augmentation | With `--augmentation light` the batches are augmented by `train_common.py` before they reach the model, so the model must not augment again. |

A three-channel input is outside what the pipeline does today: SIL, the PIL bench and both sketches handle one,
but training, calibration and MIL read grayscale only.

### In the Training Script

**A variant of a family that exists.** Add a branch where the family builds its network and the name to the
`choices` of `--base_model`:

- MobileNet family: `create_mobilenet_model()` in `src/train_mobilenet.py` (the backbone class, its
  `preprocess_input` and the width multiplier).
- ResNet family: `build_resnet()` in `src/train_resnet.py`. Its last branch builds `ResNet18` for any name that is
  not `ResNet8`, so give the new network a branch of its own.

**A new family.** Write `src/train_<family>.py` after the three that exist. What it has to do, in order:

1. Parse its options with `cli.ArgumentParser`, with at least `--width`, `--height`, `--batch_size`, `--epochs`,
   `--learning_rate`, `--splits_dir`, `--tag` (`type=artifacts.valid_tag`), `--overwrite` and `--seed`, and call
   `train_common.add_recipe_arguments(parser)` for `--lr_schedule` and `--augmentation`. Do it in a
   `parse_args()` that the script calls above the line that imports TensorFlow, as the three do, so that
   `--help` and a wrong option answer at once.
2. Take the names of its outputs from `artifacts.model_stem()` and `artifacts.training_curve_name()`, and call
   `artifacts.refuse_to_overwrite()` on them before doing any work.
3. Call `keras.utils.set_random_seed(seed)` before the data and the model exist.
4. Read the images with `train_common.load_splits()` and nothing else: it returns the two datasets, the class
   names and the description of the data for the training record.
5. Save the best epoch with `ModelCheckpoint(monitor="val_accuracy", save_best_only=True)` and add
   `train_common.schedule_callbacks()`.
6. After the fit, write `artifacts.write_class_names()` and `artifacts.write_training_record()` with
   `train_common.recipe_settings()`, `train_common.software_versions()` and the data description.
7. End with a non-zero exit code when the training fails.

The run name is `{architecture}[_{tag}]+{batch}+{epochs}+{learning rate}+{width}+{height}`. Keep it: the quantizer
reads the width and the height from its last two fields, and every result file is named after it.

The tests name the three training scripts where they check their options, their exit codes, their defaults and
that they read images through `images.py` (`tests/test_cli.py`, `tests/test_images.py`,
`tests/test_training_recipe.py`). A fourth script is only covered by the check that every script uses
`cli.ArgumentParser` until it is added to those lists.

### After Training

The commands are those of [Usage](../README.md#usage-full-pipeline), unchanged. What depends on the model:

| Step | Script | What to look at for a new architecture |
|---|---|---|
| MIL | `mil_benchmark.py` | Nothing: it loads the `.keras` file. |
| Quantization | `quantize_int8_basic.py` | The conversion fails here if a layer has no integer kernel. |
| SIL | `sil_benchmark.py` | The default interpreter refuses some valid INT8 files; `--kernels reference` or `builtin` runs them, and the choice is recorded. |
| Header | `tflite_to_c.py` | It writes `model.h` into both sketches. |
| Does it fit? | `compile_upload_arduino.py --compile_only` | Flash: the `.tflite` is linked into the firmware. Tensor arena: known only on a board, see below. |
| PIL, HIL | `pil_benchmark.py`, `hil_benchmark.py` | The firmware registers every TFLite-Micro operator (`AllOpsResolver`), so no operator list is edited. |
| Summary table | `build_benchmark_table.py --tag <tag>` | Nothing: it finds the runs by their tag. |
| Ladder tables | `build_ladder_tables.py` | Name the architectures: `--archs` (all, the first one fixes the image order) and `--deployable` (those with PIL and HIL runs). The defaults are the seven and the five of the published benchmarks. The tables handle two classes. |
| MACs | `count_macs.py` | An operator it knows neither as counted nor as free of products ends the run with exit code 3 and is named: add it to `COUNTED`, with its formula, or to `NO_MACS` in the script. |
| Checksums | `models/checksums.sha256`, `verify_models.py` | Only when its results go into the repository, see below. |

**Flash and tensor arena.** The limits and the figures of the published models are in
[Hardware Details](../README.md#hardware-details) and in [Boards and Memory](hardware.md#boards-and-memory). In short:

- The model shares the internal flash with the firmware. `compile_upload_arduino.py --target pil --board nicla
  --compile_only` prints the flash and RAM the build uses without a board. For the Portenta H7 arduino-cli prints no
  size through the script; [`flash_budget_build.txt`](flash_budget_build.txt) shows how the sizes were read from a
  build.
- The arena a model needs is printed by the firmware at boot (`ARENA_USED:<bytes>`), so it is known once the model
  has run on a board. On the Portenta H7 the default arena (4 MiB, external SDRAM) leaves room to find it out. On the
  Nicla Vision the arena is a static array in internal RAM, 368 KiB by default: build with `--arena_kb` for a model
  that needs more. Too small an arena stops at boot with a message; too large a one fails at link time.
- A model input larger than the image buffer stops at boot with a message; `--image_buffer_bytes` changes the buffer.
  The HIL sketch captures 160×120 and resizes to another input size by nearest neighbour, which the PIL path does not
  do (the host resizes there).

**Results in the repository.** A result file records the SHA-256 of the model it used, and
`models/checksums.sha256` lists every model file behind a published result. `python src/verify_models.py --results`
and `tests/test_release.py` fail while a published result (one that git tracks) names a model that is not in the
list. `python src/verify_models.py --update_list --set coco` (or `vww`, `plates`) rewrites the entries of that set
from the model files its published results name, sorted by path. The archive a model is released in
follows its tag: `_COCO` and `_VWW` have their own, every other name goes with the vehicle benchmark. If the new
files are the first published ones to carry a key that `tests/test_results_schema.py` lists as not yet
published, that test asks for the key to be taken off its list.

### How This Section Was Checked

By doing, on a scratch copy of the repository, with a fourth training script for a small depthwise-separable
network and a partition made from the eight images of `data/sample/`: the options of the new script, the data
path (`train_common.load_splits()`), the model, a checkpoint saved under the run name with its class names, and
then `quantize_int8_basic.py`, `mil_benchmark.py`, `sil_benchmark.py --kernels reference`, `tflite_to_c.py`,
`count_macs.py`, `build_benchmark_table.py`, `verify_models.py --results` (which reported the two unlisted files)
and `build_ladder_tables.py` (which stopped and listed the missing PIL and HIL runs). No script after training
was changed.

Not run: the training itself (the checkpoint held the initial weights, so the figures of that exercise mean
nothing), the training record a finished run writes, and everything that needs a board: the build with the new
`model.h`, the arena, PIL and HIL.

---

## Adding a Board

What is specific to a board sits in two places: the `BOARD SUPPORT` block at the top of each sketch, and one
entry of `src/boards.py`. The Nicla Vision was added that way to a framework written for the Portenta H7, and it
is the worked example below. [Portability](hardware.md#portability) says what the benches and the wire protocol
assume of any board; this section is the list of what to write.

### The Two Places

**`src/boards.py`, one entry of `BOARDS`.** The key is what the operator types after `--board`.

| Field | Used for |
|---|---|
| `label` | Messages, table headers and plot labels. |
| `announces` | The name the firmware prints in its `BOARD:<name>` line. The benches record it (`run.board`, `board`) and stop when it is not the board given with `--board`. |
| `camera` | Messages and the plot of the HIL bench, and a note of the ladder tables. |
| `core`, `fqbn` | What `compile_upload_arduino.py` installs and builds for. |
| `port_match` | Text looked for in the board names of `arduino-cli board list` to find the port. |
| `results_dir` | Root of the board's results: `results/<board>/{pil,hil}`. The Portenta H7 keeps `results/`, where the published files are. |
| `default_arena_kb` | Only for a board whose arena is fixed at build time; a test keeps it equal to the sketch's default. |

`--board` takes its choices from this table in `compile_upload_arduino.py`, the two benches,
`calibrate_hil_rig.py` and `build_ladder_tables.py`, so they need no edit. Two scripts do not follow it by
themselves: `build_benchmark_table.py` gathers the boards named in `--boards` (default `portenta,nicla`), and
`independence_report.py` reads one results root (`--results_dir results/<board>` for the PIL and HIL runs of
another board).

**The `BOARD SUPPORT` block of each sketch**, one `#elif defined(ARDUINO_<BOARD>)` branch:

| Macro | Sketch | Meaning |
|---|---|---|
| `BOARD_NAME` | both | The `announces` value of the table, exactly. |
| `BOARD_HAS_SDRAM`, `ARENA_MEMORY` | both | Whether the arena and the image buffer are allocated in external RAM or are static arrays, and the word the boot lines use for it. |
| `TENSOR_ARENA_KB`, `IMAGE_BUFFER_BYTES` | both | Defaults of the two sizes, inside `#ifndef` so that `--arena_kb` and `--image_buffer_bytes` can replace them at build time. |
| `CAMERA_NAME`, `CAMERA_FAIL_TEXT` | HIL | Texts of the boot lines. |
| `CAMERA_PIXEL_FORMAT`, `CAMERA_DELIVERS_RGB565` | HIL | The format asked of the camera, and whether the frame must be converted to luminance before the common path. |

Outside the block the HIL sketch also needs the include and the object of the camera driver. Everything else is
common and must stay as it is, because the host reads it: the `#...@` framing, the identity lines, `READY_PIL` and
`READY_HIL`, the end-of-inference banners and the latency lines (see
[Serial Protocol](hardware.md#serial-protocol-pil-bench)). The latency comes from the DWT cycle counter of the
Cortex-M, so a core without it needs another clock and that part of the sketches rewritten.

### The Nicla Vision, Step by Step

1. **Build target.** Core `arduino:mbed_nicla`, FQBN `arduino:mbed_nicla:nicla_vision`: the entry in `boards.py`.
2. **Memory.** The board has the MCU of the Portenta H7 and no SDRAM chip, so `BOARD_HAS_SDRAM` is 0 and the arena
   and the image buffer are static arrays. The linker then accounts for them: a size that does not fit fails the
   build. The default arena, 368 KiB, was sized for `MobileNetV2`; `--arena_kb` sets another.
3. **Camera.** The GC2145 is a colour sensor and its grayscale mode does not deliver luminance in the core used, so
   the sketch asks for RGB565 and converts it on the MCU with the BT.601 weights in integer arithmetic. The
   conversion was checked against Pillow's `convert('L')` for every RGB565 value, so the frame means the same as
   the grayscale images the other levels read. Its time is counted in the `PRE` phase.
4. **Host.** Nothing but the table entry: `--board nicla` builds, files the results under `results/nicla/` and
   checks the `BOARD:NICLA_VISION` the firmware announces.
5. **Sizes.** `--compile_only` gave the flash and RAM of each sketch, and the arena each model needs came from
   `ARENA_USED` on the Portenta H7. The figures are in [Boards and Memory](hardware.md#boards-and-memory).

### What the Tests Notice

- `tests/test_boards.py` checks that every entry of `BOARDS` has the fields the scripts use, that no two boards
  announce the same name, and that each sketch defines `BOARD_NAME` for every board of the table. An entry without
  its two sketch branches fails there.
- The simulated boards of the tests take the announced name as a parameter (`SimulatedBoard(level, board=...)`
  in `tests/test_boards.py`, which the PIL tests use, and `RigBoard(..., board=...)` in
  `tests/test_hil_session.py`), so the bench tests can be repeated for a new board without hardware.

### What Has Been Exercised

- **PIL on the Nicla Vision: run on the board.** The runs are in `results/nicla/pil/`: the `MobileNetV2` of the
  vehicle benchmark (940 images) and the COCO models that fit the board (5,000 test images each). For every COCO
  model measured on both boards, the two boards gave the same prediction for each of the 5,000 images
  (`python src/build_benchmark_table.py --tag COCO` recounts it). For the vehicle model see
  [Boards and Memory](hardware.md#boards-and-memory): the Portenta H7 run it is compared with there was not
  published.
- **HIL on the Nicla Vision: builds, not run.** What stays open until it runs is listed under
  [Portability](hardware.md#portability): the byte order of the RGB565 frame, the field of view at 160×120 and
  the RAM left for the camera frame.
- **A third board: not done.** The steps above are what adding the second one took; they have not been repeated
  for another MCU family, and a core other than `arduino:mbed_*` may need more than the two places named here.
