# Contributing to PHLAME

Bug reports, questions and changes are welcome. This file says how to report a problem, how to run the tests and
which conventions the repository follows, so that a change fits in without a round of corrections.

## Reporting a Problem

Open an issue at [GitHub Issues](https://github.com/F4bian1012/phlame-tinyml/issues); the README's
[Support](README.md#support) section lists the other way to reach the maintainer. A report is quick to act on when
it gives:

- the commit you are on, the operating system and the Python version;
- the exact command and its complete output;
- for a figure that looks wrong, the `metrics_*.json` of the run: it records the SHA-256 of the model, the board
  for PIL and HIL and, in files written from November 2026 on, the software versions;
- for a board problem, the board, the version of its Arduino core and the lines the firmware prints at boot.

Do not attach photographs that show people or other personal data.

## Setting Up

```bash
git clone https://github.com/F4bian1012/phlame-tinyml.git
cd phlame-tinyml
python -m venv venv
# activate it (see Installation in the README), then:
pip install -r requirements.txt
```

`requirements.txt` gives minimum versions; `requirements-lock.txt` pins the exact versions of the project
environment, for reproducing the published figures. The Arduino toolchain is needed only to build or flash the firmware
([Requirements](README.md#requirements)).

## Running the Tests

```bash
python -m unittest discover -s tests
```

The suite needs no board, no dataset and no trained model, and takes a minute or two. It runs the benches
against a simulated board, rebuilds the published tables from the result files, recomputes their statistics from
hand-worked values and checks every published result file against the documented schema.

- A test whose package is missing is skipped, not failed: read `OK (skipped=N)` as "that part was not checked".
  TensorFlow is needed only by `tests/test_sil_kernels.py` and by part of `tests/test_count_macs.py` and
  `tests/test_software_versions.py`; the bench tests need pyserial, Pillow, OpenCV, scikit-learn and seaborn; the
  table tests need numpy and scipy.
- To compare the tables yourself (the ladder tables of the COCO benchmark, in `results/tables/coco/`):
  ```bash
  python src/build_ladder_tables.py --output_dir results/smoke_test/tables/coco
  python tests/tables_check.py results/tables/coco results/smoke_test/tables/coco
  ```
- `.github/workflows/tests.yml` runs both steps on Linux and on Windows for every push to `main` and every pull
  request.

Nothing in the suite trains a model or talks to a board. A change to a sketch or to what a bench exchanges with
it has to be tried on a board as well; say in the pull request what was run on hardware and what was not, as
[`docs/hardware.md`](docs/hardware.md) does.

## Conventions

These are what the code and the history already follow.

- **English everywhere**: code, comments, messages of the scripts and of the firmware, keys of the result files,
  documents and commit messages.
- **Run names.** Every file of a model derives from
  `{architecture}[_{tag}]+{batch}+{epochs}+{learning rate}+{width}+{height}` (`src/artifacts.py`). Give each
  dataset its own `--tag`; `COCO` and `VWW` are those of the published runs.
- **Command line.** Scripts parse their options with `cli.ArgumentParser` (`src/cli.py`), which accepts every
  multi-word option with `_` or `-`. Declare options with `_`. The four benches share `--model_path`,
  `--splits_dir`, `--output_dir` and `--overwrite`; the two board benches also share `--port`, `--board`,
  `--baud`, `--count`, `--delay`, `--gap` and `--folder`.
- **No silent overwrite.** A script stops before replacing an existing output unless `--overwrite` is given
  (`artifacts.refuse_to_overwrite()`). Scratch outputs go to `results/smoke_test/`, which git ignores.
- **One image loader.** Training, calibration, MIL, SIL and the PIL bench read image files through
  `src/images.py` and nothing else, so every level receives the same bytes.
- **One evaluation for the PC levels.** MIL and SIL compute their metrics and write their result files
  through `src/evaluation.py`; a script only adds what is its own (how the model runs, its own keys).
- **The command line comes before TensorFlow.** A script that uses TensorFlow defines `parse_args()` at the
  top and calls it, when run as a script, above the line that imports TensorFlow, so `--help` and a wrong
  option answer at once (`tests/test_help_without_tensorflow.py`).
- **Failures are visible.** A step that reports an error ends with a non-zero exit code.
- **Result files have a documented schema.** Every key of `metrics_*.json` and `conditions_*.json` is in
  [`docs/results_schema.md`](docs/results_schema.md), and `tests/test_results_schema.py` checks the document and
  the published files against each other in both directions. A new key is added to the document with the date it
  is written from; while no published file carries it, it is listed in that test, which fails once one does.
- **Published results are not edited.** A file under `results/` is what a script wrote for a measurement. To
  change one, measure again or regenerate it with the script that wrote it. The few values added to published
  files after the measurement are named as such in the schema document.
- **Text the host reads from the firmware lives in one place.** The banners, the identity lines and the board
  names are in `src/boards.py`, and `tests/test_boards.py` reads the sketches to keep them in step.
- **Commit messages** start with what kind of change it is (`feat:`, `fix:`, `docs:`, `test:`, `results:`,
  `data:`, `ci:`, `refactor:`) and say what changed and what was measured.

## What Does Not Go into Git

- Model files (`.keras`, `.tflite`): they are released as archives, and `models/checksums.sha256` lists the
  digest of every model behind a published result (`python src/verify_models.py`).
- Image files of the datasets. The partitions are published as manifests and reports (`data/splits_*/`); the
  images are rebuilt from their public source. The eight photographs of `data/sample/` are the only ones in the
  repository.
- Photographs with personal data, in any form, figures and previews included.
- Local paths and user names in result files and documents.

## Extending

[`docs/extending.md`](docs/extending.md) says how to add an architecture and how to add a board, and what has and
has not been tried. [`CHANGELOG.md`](CHANGELOG.md) lists what changed.
