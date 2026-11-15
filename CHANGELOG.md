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
  Holm correction) from the result files, without hardware. It now builds them for any benchmark that has the
  four levels (`--tag`, `--run_suffix`, `--splits_dir`, `--board`), not only for the vehicle benchmark.
- `build_benchmark_table.py` builds the summary table of a benchmark measured up to PIL, with the image-by-image
  agreement between levels and between boards.

### Your Own Dataset

- The classes are the subfolders of the dataset; nothing in the code names the classes of the published
  benchmarks.
- `--tag` gives each dataset its own run name, and the scripts stop instead of replacing an existing checkpoint,
  `.tflite` or result file unless `--overwrite` is given.
- The label order travels with each model (`{run}.classes.txt`).
- `data/splits/` is a working folder that git ignores; the published partitions live in `data/splits_coco/`,
  `data/splits_plates/` and `data/splits_vww/`.
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
- `find_near_duplicates.py` and `independence_report.py` look for test images that show the same photo or scene
  as a training image and recompute the results without them; what they found on each published partition is in
  [`docs/data_independence.md`](docs/data_independence.md).

### Two Boards

- One sketch per level now serves the Arduino Portenta H7 and the Arduino Nicla Vision. `--board` selects the
  board in `compile_upload_arduino.py`, the two benches, the rig calibrator and `build_ladder_tables.py`;
  `src/boards.py` holds what changes with the board.
- The results of the Nicla Vision go to `results/nicla/`, and every PIL and HIL file records the board the
  firmware announced (`run.board`). A bench stops when that is not the board given with `--board`.
- `compile_upload_arduino.py` takes `--arena_kb`, `--image_buffer_bytes` and `--compile_only` (build and report
  the sizes without a board).
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
- `CITATION.cff` and `.zenodo.json` describe version 1.0.0 without a release date or a DOI, and the README says
  what is published today; tests keep the three in agreement.
- The Apache-2.0 licence text that the two firmware sketches refer to is in `deployment/`.

### Tests and Continuous Integration

- The test suite now has more than three hundred tests. It needs no board, dataset or model: the benches run
  against a simulated board, the statistics of the tables are checked against hand-worked values and the
  published tables are rebuilt and compared.
- A workflow, `.github/workflows/tests.yml`, runs the suite and the table comparison on Linux and Windows for
  every push to `main` and every pull request.

### Documentation

- [`docs/extending.md`](docs/extending.md): how to add an architecture and a board.
  [`CONTRIBUTING.md`](CONTRIBUTING.md): tests and conventions.
- [`docs/hardware.md`](docs/hardware.md), [`docs/scripts.md`](docs/scripts.md),
  [`docs/results_schema.md`](docs/results_schema.md), [`docs/data_independence.md`](docs/data_independence.md)
  and the build record of the flash budget, [`docs/flash_budget_build.txt`](docs/flash_budget_build.txt).
- Several statements of the README were reworded to say what the data show, among them the significance of the
  PIL-to-HIL gap of `MobileNetV2`, which depends on the session.

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
| `data/splits/` manifests of the vehicle benchmark | `data/splits_plates/` |

### Starting Point

What the repository already had at the beginning of November 2026, before the changes above: the three families
of training scripts with INT8 quantization, the PIL bench with per-phase latency, the camera-in-the-loop HIL bench
with the model fingerprint, the results of the vehicle benchmark at four levels with its repeatability sessions,
a four-level run on Visual Wake Words, and the script that builds the statistical tables.
