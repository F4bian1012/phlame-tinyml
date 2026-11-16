# Changelog

What changed in PHLAME, grouped by what a user of the repository notices. The commit history has the detail.

## Unreleased

This is the state being prepared as version 1.0.0. It has not been tagged or archived: there is no release and no
DOI yet (see [Citation](README.md#citation)).

### The Fidelity Ladder

- Four levels, one script each, named after the level: `mil_benchmark.py` (float model on the PC),
  `sil_benchmark.py` (INT8 model on the PC), `pil_benchmark.py` (INT8 model on the chip, images sent over serial)
  and `hil_benchmark.py` (INT8 model on the chip, frames from its own camera looking at a monitor).
- The four take the same options (`--model_path`, `--splits_dir`, `--output_dir`, `--overwrite`) and write files
  with one naming pattern: `confusion_matrix_`, `metrics_`, `predictions_`, `latency_metrics_` and `conditions_`
  followed by the run name. Every level writes the prediction of each image, so two levels can be compared image
  by image.
- On-chip latency per phase (pre-processing, inference, post-processing), from the cycle counter of the Cortex-M7.
- `build_ladder_tables.py` rebuilds the eight statistical tables (paired bootstrap intervals, exact McNemar test,
  Holm correction) from the result files, without hardware. It builds them for any benchmark that has the four
  levels: without options those of the COCO benchmark, in `results/tables/coco/`, and another one is named with
  `--tag`, `--run_suffix`, `--splits_dir` and `--board`.
- `build_benchmark_table.py` builds the summary table of a benchmark measured up to PIL, with the image-by-image
  agreement between levels and between boards.
- MIL and SIL compute their metrics and write their result files through one module, `src/evaluation.py`;
  the files they write did not change. The scripts that use TensorFlow answer `--help` and a wrong option
  without loading it.

### Your Own Dataset

- The classes are the subfolders of the dataset; nothing in the code names the classes of the published
  benchmarks.
- `--tag` gives each dataset its own run name, and the scripts stop instead of replacing an existing checkpoint,
  `.tflite` or result file unless `--overwrite` is given.
- The label order travels with each model (`{run}.classes.txt`).
- `data/splits/` is a working folder that git ignores; the published partitions live in `data/splits_coco/`
  and `data/splits_vww/`.
- `split_dataset.py --test_input_dir` takes the test images from a separate source, so that training and test
  cannot share a file.

### Public Benchmark on COCO

- `download_vww_subset.py` fetches `person` / `non_person` images from MS-COCO 2017, as a balanced subset or, with
  `--full_split`, a whole split labelled by the rule of the original Visual Wake Words script.
- The COCO benchmark was first published on 13,000 images (12,000 of `train2017` for training and validation,
  1,000 of `val2017` for the test) and then replaced by the whole of COCO: all 118,287 images of `train2017` and
  all 5,000 of `val2017`, stored as lossless PNG (`reshape_images.py --format png`). The results of the first
  partition were removed with it.
- Seven architectures were trained on the full partition with a recipe taken from the published work on this
  task. The options are in the three training scripts: `--lr_schedule stepped`, `--augmentation light`,
  `--trainable_backbone` for the MobileNets and `--batch_norm` for SqueezeNet.
- `compare_results.py` says whether a reproduced run gives the published result: same model file, same metrics,
  same prediction for every image.
- The benchmark has its fourth level: HIL sessions of the five models that fit, on the Portenta H7 and the 200
  images of the HIL subset, four repeatability sessions of `MobileNetV2`, and the eight ladder tables in
  `results/tables/coco/`, which a test and the CI now rebuild and compare. The rig report taken before the
  sessions is in `results/hil/calibration/`. A first set of sessions, measured the same day before the camera
  was aligned, was published and then withdrawn; it is in the history of the repository (commit 407a025).
- `find_near_duplicates.py` and `independence_report.py` look for test images that show the same photo or scene
  as a training image and recompute the results without them; what they found on the COCO partition is in
  [`docs/data_independence.md`](docs/data_independence.md).

### Two Boards

- One sketch per level now serves the Arduino Portenta H7 and the Arduino Nicla Vision. `--board` selects the
  board in `compile_upload_arduino.py`, the two benches, the rig calibrator and `build_ladder_tables.py`;
  `src/boards.py` holds what changes with the board.
- The results of the Nicla Vision go to `results/nicla/`, and every PIL and HIL file records the board the
  firmware announced (`run.board`). A bench stops when that is not the board given with `--board`.
- `compile_upload_arduino.py` takes `--arena_kb`, `--image_buffer_bytes` and `--compile_only` (build and report
  the sizes without a board).
- HIL of the COCO benchmark on the Nicla Vision: `MobileNetV2`, `MobileNet` and `MobileNetV3Small` over the 200
  images of the HIL subset, with the report of the calibrator before each session (`results/nicla/hil/`,
  tables in `results/nicla/tables/coco/`). With that camera, from 94 cm, no model loses a distinguishable amount
  from PIL to HIL; on the Portenta H7 two do. `SqueezeNet` and `ResNet8` cannot run HIL on this board.
- Three repeatability sessions of `MobileNetV2` at HIL on the Nicla Vision (`results/nicla/hil/rep1` to `rep3`):
  macro-F1 0.835 to 0.869, with the camera creeping up to 1.6 px within a session.
- A first attempt on the Nicla Vision was discarded: the board crept on its mount during three sessions measured
  behind a single calibration report. The documents now ask for a report before every session.
- Why the Nicla Vision needs fewer cycles than the Portenta H7 for the same model was measured: with the tensor
  arena in internal RAM (`--arena_memory internal`) the Portenta H7 runs `MobileNet` within 0.2% of the cycles
  of the Nicla Vision; the external SDRAM costs about 18% of the inference time of that model
  (`results/experiments/`).
- The firmware comments and serial messages are in English. The benches still accept the end-of-inference
  banners of the earlier builds, so a board does not have to be reflashed.

### Board Benches

- The PIL sketch announces the model it carries (FNV-1a fingerprint, size, input shape), and the PIL bench
  checks it against `--model_path` before it measures, takes the image size from the firmware, and refuses a
  wrong board, model or shape. HIL already checked the fingerprint.
- The PIL bench survives a dropped serial link: it reopens the port, sends the same image again and counts it
  (`run.n_reconnects`). It gives up on a board that stops answering (`--max_unanswered`).
- A PIL run or an HIL session that ends early keeps what it measured, is marked as not completed
  (`run.completed = false`) and ends with exit code 3.
- `calibrate_hil_rig.py` checks the alignment of the camera and the monitor before an HIL session, and
  `--monitor` chooses the monitor of the stimuli.
- The PIL sketch announces where its tensor arena is (`ARENA_MEMORY:SDRAM` or `INTERNAL`) and the PIL bench
  records it (`model.arena_memory`). `compile_upload_arduino.py --arena_memory internal` builds the Portenta H7
  firmware with the arena in internal RAM, where the Nicla Vision has it; `pil_benchmark.py --hil_subset` sends
  only the images of the HIL subset; and `compare_pil_cycles.py` puts the cycles per phase of several PIL runs of
  one model side by side. Together they allow the place of the arena to be measured on one board. No such run is
  published yet.
- The HIL bench keeps the captured frames of each session in a folder of its own,
  `data/hil_frames/<name of the session's files>/`, recorded in the conditions file, and stops before opening the
  port when that folder already holds frames, unless `--overwrite` is given. Before, every session wrote to
  `data/hil_frames/` and replaced the frames of the one before. A folder given with `--frames_dir` is used as it
  was.
- The `model.h` in git of the two sketches carries the `MobileNetV2` of the COCO benchmark, and the build record
  of the flash budget was made again with that header.

Which of these have run on a board and which only against the simulated board of the tests is kept up to date in
[`docs/hardware.md`](docs/hardware.md).

### The Same Bytes at Every Level

- One image loader, `src/images.py`, reads the files for training, the calibration of the quantizer, MIL, SIL and
  the PIL bench. Before it, MIL and SIL decoded each JPEG with TensorFlow and the PIL bench with Pillow, and the
  two did not give the same pixels.
- `sil_benchmark.py --kernels {default,builtin,reference}` chooses the TFLite kernels that run the INT8 model on
  the PC and records the choice (`model.kernels`). The kernels do not give the same prediction for every image,
  and the default interpreter refuses some valid INT8 files. The SIL results of the COCO benchmark use the
  reference kernels; the other two kinds are in `results/sil_kernels/`.

### Result Files

- The keys and fixed texts of the result files are in English. `docs/results_schema.md` has the table of
  former names, documents every key and says from when each newer key is written; a test keeps the document and
  the published files in step.
- Newer keys: the board (`run.board`), the reconnections of the PIL bench, the kernels of SIL, the identity of the
  model in PIL, whether a run was completed, and the software versions behind a result (`software`).
- `count_macs.py` counts the multiply-accumulate operations of a `.tflite` from its graph;
  `results/tables/macs_COCO.csv` has the count of the seven COCO models.
- The results, tables and partition manifests of a private application case the framework was first developed on
  were removed from the repository on 2026-11-15; they remain in its history up to commit 645875d.

### Reproducible Training

- `--seed` in the three training scripts fixes the initial weights, the shuffling and the augmentation. Each run
  leaves `{run}.training.json` next to its checkpoint: seed, recipe, per-epoch history and software versions. Two
  runs with one seed are identical on a CPU and not on a GPU; the record is what lets them be compared.
- A step that fails ends with a non-zero exit code, so a chain of commands does not carry on with the files of an
  earlier run.

### Release Tooling

- `models/checksums.sha256` lists the SHA-256 of every model file behind a published result, and
  `verify_models.py` checks model files against it, checks the result files against it, fetches the model
  archives of a release and builds them.
- The model archives also carry the training record and the class order of each model, with the path of the
  training machine reduced to the name of its folder; `verify_models.py --download` writes them next to the
  models.
- `CITATION.cff` and `.zenodo.json` describe version 1.0.0 without a release date or a DOI, and the README says
  what is published today; tests keep the three in agreement.
- The Apache-2.0 licence text that the two firmware sketches refer to is in `deployment/`.
- `pyproject.toml` states the project metadata (name, version, licence, dependencies). It installs no code:
  the scripts are still run from a checkout as `python src/<script>.py`.

### Tests and Continuous Integration

- The test suite now has more than three hundred tests. It needs no board, dataset or model: the benches run
  against a simulated board, the statistics of the tables are checked against hand-worked values and the
  published tables are rebuilt and compared.
- A workflow, `.github/workflows/tests.yml`, runs the suite and the table comparison on Linux and Windows for
  every push to `main` and every pull request.

### Documentation

- PHLAME is a name: the README no longer expands it as an acronym.
- [`docs/data_independence.md`](docs/data_independence.md) says what the one image of the HIL subset that shares
  its scene with a training image does to the PIL and HIL figures (0.0014 of macro-F1 at most).
- [`docs/extending.md`](docs/extending.md): how to add an architecture and a board.
  [`CONTRIBUTING.md`](CONTRIBUTING.md): tests and conventions.
- [`docs/hardware.md`](docs/hardware.md), [`docs/scripts.md`](docs/scripts.md),
  [`docs/results_schema.md`](docs/results_schema.md), [`docs/data_independence.md`](docs/data_independence.md)
  and the build record of the flash budget, [`docs/flash_budget_build.txt`](docs/flash_budget_build.txt).
- Several statements of the README were reworded to say what the data show.
- The COCO benchmark is the benchmark of the repository: it is what `build_ladder_tables.py` builds without
  options and the model in the `model.h` of the sketches (see above).
- The header of the README names the two boards: PIL is measured on the Arduino Portenta H7 and on the Arduino
  Nicla Vision, HIL on the Portenta H7.
- The README is shorter. The detail of the seven usage steps and of the Arduino Nicla Vision moved to
  [`docs/usage.md`](docs/usage.md); the README keeps the sequence of commands.
- The README was cut a second time. It keeps the two tables of the COCO benchmark, a short list of what they
  show, the commands that reproduce the PC levels and each limitation in one or two sentences. What left it is in
  the documents: the full recipe and the full notes of the COCO benchmark in the new
  [`docs/coco_benchmark.md`](docs/coco_benchmark.md); the hardware details and the flash budget in
  [`docs/hardware.md`](docs/hardware.md); the index of the scripts and the layout of the repository in
  [`docs/scripts.md`](docs/scripts.md); the table of the result files, how to read them and the Python snippet in
  [`docs/results_schema.md`](docs/results_schema.md); and the note on the two requirements files in
  [`docs/usage.md`](docs/usage.md). No number changed.
- [`docs/related_work.md`](docs/related_work.md) compares PHLAME with MLPerf Tiny and with other benchmarks and
  tools: what each measures, what PHLAME does that was not found in them, and what it does not do (no energy
  measurement, one microcontroller, no standardised submission process, no release yet). Each reference was then
  looked up at its primary source and the statements that cite it were checked against it (full text in the HTML
  rendering of arXiv, documentation and repositories; the PDF files were not opened, and three sources could not
  be checked in full, as its status note says). That narrowed what the document says was not found elsewhere:
  ST's `validate` command, ML-EXray and EdgeMark each measure some difference between a reference on the PC and
  the device, so the claim is now the count, for a microcontroller and over the test set of a public benchmark,
  of the images predicted differently.

### Renamed

For anyone who used an earlier commit:

| Before | Now |
|---|---|
| `src/test_model.py` | `src/mil_benchmark.py` |
| `src/test_tflite_model.py` | `src/sil_benchmark.py` |
| `src/hil_camera_benchmark.py` | `src/hil_benchmark.py` |
| `deployment/hil_camera_firmware/` | `deployment/hil_firmware/` |
| `Matriz_*.png`, `HIL_Confusion_Matrix_*.png` | `confusion_matrix_*.png` |
| `hil_latencies_*.csv` | `latency_metrics_*.csv` |
| `hil_conditions_*.json` | `conditions_*.json` |
| Spanish keys in the result files (`nivel`, `modelo`, ...) | English keys (`level`, `model`, ...): table in `docs/results_schema.md` |
| `--path_proyecto` | `--sketch_path` (the former name is still accepted) |
| `--tag_suffix` of `build_ladder_tables.py` | `--run_suffix` (the former name is still accepted) |
| `build_ladder_tables.py --tag <tag>` with the run suffix `+32+20+0.0001+160+120` implied | add `--run_suffix +32+20+0.0001+160+120` (the suffix implied is now that of the COCO benchmark) |
| Frames of every HIL session in `data/hil_frames/` | `data/hil_frames/<name of the session's files>/` (`--frames_dir data/hil_frames` gives the former place) |

### Starting Point

What the repository already had at the beginning of November 2026, before the changes above: the three families
of training scripts with INT8 quantization, the PIL bench with per-phase latency, the camera-in-the-loop HIL bench
with the model fingerprint, a four-level run on Visual Wake Words, and the script that builds the statistical
tables.
