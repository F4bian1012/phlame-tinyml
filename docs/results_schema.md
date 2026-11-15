# Results Schema

Single source of truth for the machine-readable result files in `results/`. Run
`python -m unittest discover -s tests -v` to check this document against every published JSON:
`tests/test_results_schema.py` fails if a key exists in a JSON but not here, or here but in no JSON.

**Table convention.** In every table of the two schema sections (`metrics_*.json` and `conditions_*.json`),
the first column holds the full path of exactly one key between backticks, with dots for nesting. The
`metrics_*.json` sections give paths from the root of the metrics JSON; the `conditions_*.json` section gives
paths from the root of the conditions JSON.

Dynamic keys use placeholders:

- `<FIELD>`: a telemetry field under `latency.per_field` (listed in section 7).
- `<CLASS>` under `per_class_metrics`: the class names (`Class1`, `Class2`; `person`, `non_person` in
  VWW) and also the aggregate entries added by scikit-learn: `macro avg` and `weighted avg` (objects with
  `precision`, `recall`, `f1-score`, `support`) and `accuracy` (a number, not an object).
- `<CLASS>` under `prediction_counts`: a predicted class index as a string (`"0"`, `"1"`, ...).

## Result Files

Every level writes into its own folder, `results/{mil,sil,pil,hil}/`, with the same file names. `{model}` is the model name, which ends in `_int8` from SIL onwards:

| File | Content | MIL | SIL | PIL | HIL |
|---|---|:-:|:-:|:-:|:-:|
| `confusion_matrix_{model}.png` | Confusion matrix, for visual inspection | ✓ | ✓ | ✓ | ✓ |
| `metrics_{model}.json` | Global and per-class metrics, model hash, run settings | ✓ | ✓ | ✓ | ✓ |
| `predictions_{model}.csv` | True and predicted class of every test image | ✓ | ✓ | ✓ | — |
| `latency_metrics_{model}.csv` | Per-image latency by phase, measured on the chip | — | — | ✓ | ✓ |
| `conditions_{model}.json` | Rig conditions and the model fingerprint reported by the board | — | — | — | ✓ |

In HIL the per-image predictions are columns of `latency_metrics_{model}.csv`, next to the latencies of the same capture.

- **Same `sha256` across SIL, PIL and HIL:** Verifies that the same `.tflite` file was evaluated on the host. MIL records the hash of the `.keras` file, so it never matches the other levels. Whether the board ran that file is a separate check: the firmware announces the FNV-1a fingerprint of the model it carries and the bench stops unless it is that of the `.tflite` (`model.identity_verified`). Every published HIL file has that check. The published PIL files were measured before the PIL firmware announced the fingerprint, and a board running another model is ruled out for them indirectly (see [Known Limitations](../README.md#known-limitations)).
- **`predictions_*.csv` (`;`-separated):** Columns: `Image;RelPath;TrueLabel;TrueClass;PredLabel;PredClass;Correct`. `RelPath` disambiguates files sharing identical names across class subdirectories. In PIL and HIL, empty cells differentiate between unlabelled stimuli and hardware dropouts. Joining on `RelPath` isolates the exact images that flip prediction across ladder rungs.

## Former Key Names

Until November 2026 the keys of `metrics_*.json` were written in Spanish. They were renamed in the
scripts and in every published file; no measured value changed. Code written against the former names
needs the new ones:

| Former key | Current key |
|---|---|
| `nivel` | `level` |
| `generado_utc` | `generated_utc` |
| `modelo` | `model` |
| `nombre` | `name` |
| `identidad_verificada` | `identity_verified` |
| `nota` | `note` |
| `corrida` | `run` |
| `puerto` | `port` |
| `condiciones_completas` | `conditions_complete` |
| `nota_correccion` | `correction_note` |
| `n_samples_pedidas` | `n_samples_requested` |
| `n_estimulos` | `n_stimuli` |
| `n_respondidas` | `n_answered` |
| `n_sin_respuesta` | `n_unanswered` |
| `n_muestras` | `n_samples` |
| `n_imagenes` | `n_images` |
| `metricas_globales` | `global_metrics` |
| `metricas_por_clase` | `per_class_metrics` |
| `matriz_confusion` | `confusion_matrix` |
| `matriz` | `matrix` |
| `orden` | `order` |
| `latencia` | `latency` |
| `fuente` | `source` |
| `n_imagenes_con_medida` | `n_images_measured` |
| `metodo_p95` | `p95_method` |
| `por_campo` | `per_field` |
| `media` | `mean` |
| `mediana` | `median` |
| `desv_std` | `std` |
| `validacion_cruzada` | `cross_validation` |
| `sil_en_pc` | `sil_on_pc` |
| `mil_en_pc` | `mil_on_pc` |
| `n_con_sil` | `n_with_sil` |
| `n_con_mil` | `n_with_mil` |
| `artefactos` | `artifacts` |
| `matriz_png` | `confusion_matrix_png` |
| `latencias_csv` | `latency_csv` |
| `condiciones_json` | `conditions_json` |
| `predicciones_csv` | `predictions_csv` |
| `conteo_predicciones` | `prediction_counts` |

The fixed descriptive texts (`confusion_matrix.order`, `latency.source`, `latency.p95_method`,
`model.note`, `rig`) and the hand-written session notes (`notes`, `run.correction_note`) were translated
to English in the same change.

## Metrics Schema Glossary (`metrics_*.json`)

Every evaluation script writes a `metrics_{model}[_int8].json` file capturing classification accuracy, execution metadata, and latency.

### 1. Root & Metadata
| Key | Level | Type | Description |
|---|---|---|---|
| `level` | All | `string` | Evaluation level in the fidelity ladder: `"MIL"`, `"SIL"`, `"PIL"`, or `"HIL"`. |
| `generated_utc` | All | `string` | ISO 8601 UTC timestamp when the evaluation run was completed. |
| `model` | All | `object` | Model identification, checksums, and on-chip hardware verification (see Section 2). |
| `run` | PIL, HIL | `object` | Serial connection parameters and physical rig environment settings (see Section 4). |
| `dataset` | All | `object` | Dataset partition directories, sample counts, and class definitions (see Section 3). |
| `global_metrics` | All | `object` | Global classification summary: accuracy, weighted/macro F1, precision, recall (see Section 5). |
| `per_class_metrics` | All | `object` | Detailed classification breakdown per class (`precision`, `recall`, `f1-score`, `support`), plus macro/weighted averages (see Section 5). |
| `confusion_matrix` | All | `object` | Confusion matrix structure (`labels`, `matrix`, and layout orientation `order`; see Section 6). |
| `latency` | PIL, HIL | `object` | Hardware cycle and microsecond timer statistics per inference phase (see Section 7). |
| `prediction_counts` | PIL | `object` | Frequency dictionary mapping predicted class index strings (`"0"`, `"1"`, etc.) to total counts (see Section 6). |
| `cross_validation` | HIL | `object` | Summary of paired PC MIL/SIL evaluation on the exact camera-captured frames (see Section 8). |
| `artifacts` | All | `object` | Relative filesystem paths to companion outputs (PNG heatmap, CSV tables, conditions JSON; see Section 9). |
| `software` | All | `object` | Versions of the software on the PC that produced the file (see Section 10). Written from November 2026 on; absent in earlier files. |

### 2. Model Traceability (`model`)
| Key | Level | Type | Description |
|---|---|---|---|
| `model.name` | All | `string` | Model architecture tag / run identifier (e.g., `"MobileNetV2+32+20+0.0001+160+120_int8"`). |
| `model.path` | All | `string` | Relative path to the evaluated `.keras` or `.tflite` model artifact. |
| `model.sha256` | All | `string` | 64-character SHA-256 hash of the host model file: the `.keras` file in MIL and the `.tflite` file in SIL, PIL and HIL. By itself it does not prove which model is flashed on the board: `model.identity_verified` says whether the bench checked that. |
| `model.bytes` | All | `integer` | Model file size on disk in bytes. |
| `model.fnv1a_tflite` | PIL, HIL | `string\|null` | 8-character hex 32-bit FNV-1a hash computed directly from the host `.tflite` binary file. In PIL: written from November 2026 on, absent in earlier files; `null` when `--model_path` was not a file (a run allowed with `--allow_model_mismatch`). |
| `model.fnv1a_chip` | PIL, HIL | `string\|null` | 8-character hex 32-bit FNV-1a hash computed by the firmware over the compiled `g_model[]` in flash (`MODEL_FNV1A:<hex>`), read by the bench before the first image or stimulus. In PIL: written from November 2026 on, absent in earlier files; `null` when the firmware is a build that reports no fingerprint. |
| `model.identity_verified` | PIL, HIL | `boolean` | `true` if the on-chip fingerprint and size match the host `.tflite` file. Confirms that the physical board ran the expected compiled model (a non-cryptographic check against stale or wrongly flashed firmware). `false` only in a run allowed with `--allow_model_mismatch`. In PIL: written from November 2026 on; the earlier PIL files carry no such check. |
| `model.arena_used_bytes` | PIL | `integer\|null` | Part of the tensor arena the model uses on the board, as the firmware reports it (`ARENA_USED:<int>`); `null` when the firmware is a build that does not report it to the bench. Written from November 2026 on; absent in earlier files. (HIL records the same figure in `conditions_*.json`, `firmware_model.arena_used_bytes`.) |
| `model.note` | PIL, HIL | `string` | Explanatory note detailing the verification method and scope of the checksum. |
| `model.kernels` | SIL | `string` | TFLite kernels that ran the model on the PC, as given with `--kernels`: `"default"` (the interpreter's own choice, which includes the XNNPACK delegate), `"builtin"` (built-in optimized kernels without that delegate) or `"reference"` (reference kernels). Written from November 2026 on; absent in earlier files, which were all measured with the default. |

### 3. Dataset & Sampling (`dataset`)
| Key | Level | Type | Description |
|---|---|---|---|
| `dataset.data_dir` | MIL, SIL | `string` | Path to the directory containing the evaluated images (`data/splits/test` or explicit folder). |
| `dataset.folder` | PIL | `string` | Path to the directory containing the evaluated images. |
| `dataset.splits_dir` | SIL | `string` | Root folder holding the reproducible dataset partitions (`data/splits`). |
| `dataset.stimulus_source` | HIL | `string` | Origin of the stimuli: `"manifest:<path>"` (default, test manifest) or `"folder:<path>"` (when `--folder` is used). |
| `dataset.manifest_subset` | HIL | `string\|null` | Filter applied to the manifest (default: `"hil_subset==1"`, selecting the 200 fixed test images; `"all-test"` with `--all_test`; `null` with `--folder`). |
| `dataset.class_names` | All | `array[string]` | List of class labels in canonical argmax index order matching firmware output. |
| `dataset.image_size` | MIL, SIL | `object` | Tensor input geometry. |
| `dataset.image_size.width` | MIL, SIL | `integer` | Input width in pixels. |
| `dataset.image_size.height` | MIL, SIL | `integer` | Input height in pixels. |
| `dataset.n_samples` | MIL, SIL | `integer` | Total number of test images evaluated in the session. |
| `dataset.n_images` | PIL | `integer` | Number of test images sent to the board and recorded in the session (one row of each CSV); the unanswered ones are among them. |
| `dataset.n_images_planned` | PIL | `integer` | Number of images the run set out to send; equal to `dataset.n_images` in a completed run. Written from November 2026 on; absent in earlier files. |
| `dataset.n_stimuli` | HIL | `integer` | Number of optical stimuli presented and recorded in the session (one row of the latency CSV each). |
| `dataset.n_stimuli_planned` | HIL | `integer` | Number of stimuli the session set out to present; equal to `dataset.n_stimuli` in a completed run. Written from November 2026 on; absent in earlier files. |
| `dataset.n_answered` | PIL, HIL | `integer` | Number of test items for which the microcontroller successfully returned a class prediction. |
| `dataset.n_unanswered` | PIL, HIL | `integer` | Number of test items where the board timed out or failed to return a prediction. Disentangles hardware/communication dropouts from classification errors. |

### 4. Run & Environmental Rig Configuration (`run`)
| Key | Level | Type | Description |
|---|---|---|---|
| `run.port` | PIL, HIL | `string` | Serial port used to communicate with the board (e.g., `"COM9"`). |
| `run.board` | PIL, HIL | `string\|null` | Board the run was measured on, as the firmware announces it in its `BOARD:<name>` line: `"PORTENTA_H7"` or `"NICLA_VISION"`. `null` when the firmware is a build that does not announce it. The files published before the firmware announced its board were all measured on a Portenta H7, and the value was added to them by hand. |
| `run.baud` | PIL, HIL | `integer` | Serial transmission speed in baud (default: `115200`). |
| `run.input_shape` | PIL | `object` | Geometry of the images sent over serial. |
| `run.input_shape.width` | PIL | `integer` | Image width in pixels. |
| `run.input_shape.height` | PIL | `integer` | Image height in pixels. |
| `run.input_shape.channels` | PIL | `integer` | Image channels: `1` for grayscale, `3` for RGB (`--color`). |
| `run.input_shape_verified` | PIL | `boolean` | `true` when the firmware announced the input shape of its model (`INPUT_SHAPE:<H>x<W>x<C>`) and it is the shape of the images sent; a different shape stops the run before it starts. `false` when the firmware is a build that does not announce it. Written from November 2026 on; absent in earlier files. |
| `run.pre_delay_s` | PIL | `float` | Warm-up pause after opening the serial port before transmitting data (seconds). |
| `run.n_samples_requested` | PIL | `integer\|null` | PIL only: value of `--count` (number of images requested); `null` means every image in the folder. |
| `run.n_reconnects` | PIL | `integer` | Times the serial link dropped during the run and the bench reopened the port and sent the image again; `0` in a run without drops. Written from November 2026 on; absent in earlier files. |
| `run.max_unanswered` | PIL | `integer` | Value of `--max_unanswered`: images in a row without an answer after which the run is given up (`0`: never). Written from November 2026 on; absent in earlier files. |
| `run.stopped_because` | PIL | `string\|null` | Why a run that is not `completed` ended (for example `"the board stopped answering: 5 images in a row without an answer (last one: image 37 of 940)"`); `null` in a completed run. Written from November 2026 on; absent in earlier files. (HIL records it in `conditions_*.json`.) |
| `run.settle_s` | HIL | `float` | Display stabilization time after rendering each stimulus before triggering optical camera capture (default: `1.5` s). |
| `run.gap_s` | PIL, HIL | `float` | Inter-stimulus delay between successive evaluations (default: `0.5` s). |
| `run.seed` | HIL | `integer` | Random seed used for reproducible stimulus ordering (default: `42`). |
| `run.frame_dump` | HIL | `boolean` | `true` if raw 160×120 sensor frames were read back over serial for cross-level validation and image logging. |
| `run.conditions_complete` | HIL | `boolean` | `true` if all mandatory environmental conditions (`illuminance_lux`, `camera_to_screen_distance_cm`, `ambient_temp_c`) were recorded. |
| `run.completed` | PIL, HIL | `boolean` | In PIL: `true` when every planned image was sent and recorded, `false` when the run was cut short (the serial link was lost for good, the board stopped answering, Ctrl+C, an error); the figures of the file then cover only the `dataset.n_images` images recorded. In HIL: `true` when every planned stimulus was presented and recorded. `false` when the session was cut short (a lost serial link, a board that stopped answering, Ctrl+C, an error, or a stimulus image that could not be displayed): the figures of the file then cover only the `dataset.n_stimuli` stimuli measured, and the reason is in `stopped_because` of the conditions file. Written from November 2026 on; absent in earlier files, which are all complete runs. |
| `run.illuminance_lux` | HIL | `float` | Illuminance measured at the screen plane in lux (`10.0` lux in the primary protocol). |
| `run.camera_to_screen_distance_cm` | HIL | `float` | Distance from the camera lens to the monitor screen in cm (`24.0` cm in the primary protocol). |
| `run.ambient_temp_c` | HIL | `float` | Ambient room temperature measured at session start in °C. |
| `run.correction_note` | HIL | `string` | Optional post-hoc documentation note (e.g., documenting distance measurement reconciliation). Added by hand to a published file; the bench does not write it. |

### 5. Evaluation Metrics (`global_metrics` & `per_class_metrics`)
| Key | Level | Type | Description |
|---|---|---|---|
| `global_metrics.accuracy` | All | `float` | Top-1 overall classification accuracy ($[0.0, 1.0]$). |
| `global_metrics.precision_weighted` | All | `float` | Support-weighted average precision across all classes. |
| `global_metrics.recall_weighted` | All | `float` | Support-weighted average recall across all classes. |
| `global_metrics.f1_weighted` | All | `float` | Support-weighted average F1-score across all classes. |
| `global_metrics.f1_macro` | All | `float` | Unweighted macro-average F1-score. On balanced splits it matches weighted F1; any divergence indicates minority-class performance drops. |
| `global_metrics.precision_macro` | HIL | `float` | Unweighted macro-average precision across all classes. |
| `global_metrics.recall_macro` | HIL | `float` | Unweighted macro-average recall across all classes. |
| `per_class_metrics.<CLASS>` | All | `object\|float` | Per-class breakdown dictionary, or the aggregate entries described under *Table convention* (`accuracy` is a number). |
| `per_class_metrics.<CLASS>.precision` | All | `float` | Precision of the class or aggregate. |
| `per_class_metrics.<CLASS>.recall` | All | `float` | Recall of the class or aggregate. |
| `per_class_metrics.<CLASS>.f1-score` | All | `float` | F1-score of the class or aggregate. |
| `per_class_metrics.<CLASS>.support` | All | `float` | Sample count of the class or aggregate. |

### 6. Confusion Matrix & Prediction Counts (`confusion_matrix`, `prediction_counts`)
| Key | Level | Type | Description |
|---|---|---|---|
| `confusion_matrix.labels` | All | `array[string]` | List of class names matching matrix rows and columns in argmax order. |
| `confusion_matrix.matrix` | All | `array[array[integer]]` | 2D contingency table of predictions. Row index represents true class; column index represents predicted class. |
| `confusion_matrix.order` | All | `string` | Clarification string confirming layout: `"rows = true label, columns = predicted label"`. |
| `prediction_counts.<CLASS>` | PIL | `integer` | Total count of predictions for that class index string. |

### 7. Hardware Latency Profiling (`latency`)
| Key | Level | Type | Description |
|---|---|---|---|
| `latency.source` | PIL, HIL | `string` | Hardware timer description: ARM Cortex-M7 DWT cycle counter (`DWT->CYCCNT`) running at 400.0 MHz. |
| `latency.n_images_measured` | PIL, HIL | `integer` | Number of processed images with valid hardware latency telemetry lines. |
| `latency.p95_method` | PIL, HIL | `string` | Method used for 95th percentile: nearest-rank over sorted samples without interpolation. |
| `latency.per_field` | PIL, HIL | `object` | Per-field timing summaries, one entry per `<FIELD>`. |
| `latency.per_field.<FIELD>` | PIL, HIL | `object` | Per-phase timing summary. |
| `latency.per_field.<FIELD>.n` | PIL, HIL | `integer` | Number of samples. |
| `latency.per_field.<FIELD>.mean` | PIL, HIL | `float` | Mean. |
| `latency.per_field.<FIELD>.median` | PIL, HIL | `float` | Median. |
| `latency.per_field.<FIELD>.std` | PIL, HIL | `float` | Sample standard deviation. |
| `latency.per_field.<FIELD>.min` | PIL, HIL | `float` | Minimum. |
| `latency.per_field.<FIELD>.max` | PIL, HIL | `float` | Maximum. |
| `latency.per_field.<FIELD>.p95` | PIL, HIL | `float` | 95th percentile (see `latency.p95_method`). |

Values of `<FIELD>` (`CYC_*` in core cycles, `US_*` in microseconds):

- `CYC_CAPTURE`, `US_CAPTURE` (HIL): HM01B0 DCMI DMA frame acquisition duration.
- `CYC_PRE`, `US_PRE` (PIL, HIL): Sensor readout buffer transfer and input quantization time.
- `CYC_INF`, `US_INF` (PIL, HIL): Core TFLite-Micro neural network inference time (`Invoke()`).
- `CYC_POST`, `US_POST` (PIL, HIL): Post-processing, argmax identification, and telemetry formatting time.
- `CYC_TOTAL`, `US_TOTAL` (PIL, HIL): End-to-end processing latency from start of capture/reception to final output.
- `TEMP_C` (PIL, HIL): On-chip junction temperature reading from STM32H7 internal sensor in °C.
- `TS_MS` (HIL): Microcontroller system timestamp in milliseconds (`millis()`).

### 8. Cross-Validation (`cross_validation`)
| Key | Level | Type | Description |
|---|---|---|---|
| `cross_validation.sil_on_pc` | HIL | `string\|null` | Path to the INT8 `.tflite` model evaluated on the PC over the exact captured camera frame. |
| `cross_validation.sil_kernels` | HIL | `string\|null` | TFLite kernels that ran that model on the PC, as given with `--sil_kernels` (`"default"`, `"builtin"` or `"reference"`; same meaning as `model.kernels` of a SIL file); `null` without `--sil_model`. Written from November 2026 on; absent in earlier files, which used the default. |
| `cross_validation.mil_on_pc` | HIL | `string\|null` | Path to the float `.keras` model evaluated on the PC over the exact captured camera frame. |
| `cross_validation.n_with_sil` | HIL | `integer` | Number of frames successfully cross-evaluated against the PC SIL engine. |
| `cross_validation.n_with_mil` | HIL | `integer` | Number of frames successfully cross-evaluated against the PC MIL engine. |

### 9. Output Artifacts (`artifacts`)
| Key | Level | Type | Description |
|---|---|---|---|
| `artifacts.confusion_matrix_png` | All | `string` | Relative path to the generated confusion matrix heatmap image (`confusion_matrix_*.png` or `confusion_matrix_*.png`). |
| `artifacts.predictions_csv` | MIL, SIL, PIL | `string` | Relative path to the per-image classification table (`predictions_*.csv`). |
| `artifacts.latency_csv` | PIL, HIL | `string` | Relative path to the per-stimulus telemetry and timing record (`latency_metrics_*.csv` or `latency_metrics_*.csv`). |
| `artifacts.conditions_json` | HIL | `string` | Relative path to the companion test rig environment record (`conditions_*.json`). |

### 10. Software Behind the Result (`software`)
The versions that can change a figure, read when the file is written by `artifacts.software_versions()`. Each level lists the packages it uses, with the key names of the training record (`{model}.training.json`, block `software`). A version is the one of the module that was running; for a package the run did not import, the one of the installed distribution. The block holds no path, user name or host name. On PIL and HIL it describes the PC only: the model runs on the board, whose firmware is identified by `run.board` and the fingerprint in `model`. Written from November 2026 on; absent in earlier files.

| Key | Level | Type | Description |
|---|---|---|---|
| `software.python` | All | `string` | Python version (`platform.python_version()`). |
| `software.platform` | All | `string` | Operating system and architecture, as `platform.platform()` gives them (e.g. `"Windows-11-10.0.26300-SP0"`). |
| `software.numpy` | All | `string\|null` | numpy version. `null` in any package key means the package is not installed. |
| `software.scikit_learn` | All | `string\|null` | scikit-learn version: it computes the metrics and the confusion matrix. |
| `software.tensorflow` | MIL, SIL, HIL | `string\|null` | TensorFlow version: it runs the float model (MIL) and the TFLite interpreter (SIL; the kernels are in `model.kernels`). In HIL it is loaded only to classify the captured frames on the PC (`--sil_model`, `--mil_model`): `null` in a run without them. Absent in PIL, which does not use TensorFlow. |
| `software.keras` | MIL, HIL | `string\|null` | Keras version: it loads and runs the `.keras` model. In HIL, `null` unless Keras was loaded for the PC classification of the frames. |
| `software.pillow` | MIL, SIL, PIL | `string\|null` | Pillow version: it decodes the image files (`src/images.py`). |
| `software.pyserial` | PIL, HIL | `string\|null` | pyserial version: the serial link to the board. |
| `software.opencv` | HIL | `string\|null` | OpenCV version: it decodes and shows the stimuli and resizes the captured frames. |
| `software.gpus` | MIL, HIL | `array[string]\|null` | GPUs TensorFlow sees (empty list: the model ran on the CPU), as in the training record: a float model can give other figures on a GPU. `null` in HIL when TensorFlow was not loaded. |

## Physical Rig Telemetry Schema (`conditions_*.json`)

Alongside `metrics_*.json`, the HIL benchmark generates a standalone `conditions_{model}_int8.json` file logging environmental variables and hardware-in-the-loop integrity:

| Key | Type | Description |
|---|---|---|
| `rig` | `string` | Rig configuration identifier (e.g., `"monitor-controlled-stimulus (checklist Phase 1C, option a)"`). |
| `port` | `string` | USB serial device address. |
| `board` | `string\|null` | Board announced by the firmware at boot (`BOARD:<name>`); same values and same remark as `run.board` in the metrics file. |
| `baud` | `integer` | Serial communication baud rate. |
| `settle_s` | `float` | Display stabilization pause in seconds. |
| `gap_s` | `float` | Inter-stimulus gap in seconds. |
| `seed` | `integer` | Random seed controlling stimulus display sequence. |
| `stimulus_source` | `string` | Origin of the stimuli: `"manifest:<path>"` (default, test manifest) or `"folder:<path>"` (when `--folder` is used). |
| `manifest` | `string\|null` | Path to the dataset manifest; `null` when `--folder` is used. |
| `manifest_subset` | `string\|null` | Filter applied to the manifest (`"hil_subset==1"`, or `"all-test"`); `null` when `--folder` is used. |
| `dataset_folder` | `string\|null` | Folder passed with `--folder`; `null` when the stimuli come from the manifest. |
| `class_names` | `array[string]` | Class names in label order (the list index is the integer label returned by the firmware). |
| `n_stimuli` | `integer` | Number of stimuli presented and recorded in the run. |
| `n_stimuli_planned` | `integer` | Number of stimuli the run set out to present; equal to `n_stimuli` in a completed run. Written from November 2026 on; absent in earlier files. |
| `completed` | `boolean` | `true` when every planned stimulus was presented and recorded; `false` when the run was cut short. This file is written either way. Written from November 2026 on; absent in earlier files, which are all complete runs. |
| `stopped_because` | `string\|null` | Why a run that is not `completed` ended (for example `"serial link lost at stimulus 199 of 200: ..."`); `null` in a completed run. Written from November 2026 on; absent in earlier files. |
| `stimulus_monitor` | `object\|null` | Monitor the stimulus window was sent to with `--monitor`; `null` without that option (the window opens where the system puts it). Written from November 2026 on; absent in earlier files. |
| `stimulus_monitor.index` | `integer` | Value of `--monitor`: `0` is the primary monitor, `1` the next one. |
| `stimulus_monitor.geometry` | `array[integer]` | `[x, y, width, height]` of that monitor in desktop coordinates. |
| `stimulus_monitor.monitors` | `array[array[integer]]` | The same four numbers for every monitor found, the primary one first. |
| `notes` | `string` | User-provided laboratory notes (monitor model, lighting fixtures, etc.). |
| `frame_dump` | `boolean` | `true` when the captured frames were sent back and saved (`--dump_frames`, also implied by `--sil_model` / `--mil_model`). |
| `frames_dir` | `string\|null` | Folder where the dumped frames were written; `null` when `frame_dump` is `false`. |
| `sil_model` | `string\|null` | Path to the companion INT8 model executed on the host for paired cross-validation. |
| `sil_kernels` | `string\|null` | TFLite kernels that ran `sil_model` on the host (`--sil_kernels`); `null` without `--sil_model`. Written from November 2026 on; absent in earlier files, which used the default. |
| `mil_model` | `string\|null` | Path to the companion float model executed on the host for paired cross-validation. |
| `model_tag` | `string` | Experiment stem used to index all output artifacts. |
| `illuminance_lux` | `float\|null` | Calibrated illuminance measured at the screen plane in lux. |
| `camera_to_screen_distance_cm` | `float\|null` | Physical lens-to-screen distance in cm. |
| `ambient_temp_c` | `float\|null` | Room temperature measured during the benchmarking session in °C. |
| `conditions_complete` | `boolean` | `true` when lux, distance, and temperature are all recorded. |
| `utc_start` | `string` | ISO 8601 UTC timestamp marking benchmark execution start. |
| `utc_end` | `string` | ISO 8601 UTC timestamp marking benchmark execution completion. |
| `firmware_model` | `object` | Model fingerprint reported by the MCU at boot. |
| `firmware_model.model_bytes` | `integer` | Size of `g_model[]` in flash reported by MCU over serial (`MODEL_BYTES:<int>`). |
| `firmware_model.model_fnv1a` | `string` | 8-character hex 32-bit FNV-1a hash emitted by MCU (`MODEL_FNV1A:<hex>`). |
| `firmware_model.arena_used_bytes` | `integer` | Peak tensor arena memory usage reported by TFLM (`ARENA_USED:<int>`). |
| `expected_model` | `object\|null` | Fingerprint of the host `.tflite` file; `null` when no model file was given. |
| `expected_model.path` | `string` | Path to the host `.tflite` model targeted for deployment. |
| `expected_model.bytes` | `integer` | Size in bytes of the host `.tflite` model file. |
| `expected_model.sha256` | `string` | 64-character SHA-256 hash of the host `.tflite` model file. |
| `expected_model.fnv1a` | `string` | 8-character hex 32-bit FNV-1a hash calculated from the host `.tflite` file. |
| `model_identity_verified` | `boolean` | `true` if `firmware_model.model_fnv1a == expected_model.fnv1a` and `firmware_model.model_bytes == expected_model.bytes`. Validates that the flashed firmware contains the exact intended model. |

Some CLI options are recorded indirectly or not at all: `--all_test` is recorded as `manifest_subset = "all-test"` (otherwise `"hil_subset==1"`); `--folder` as `stimulus_source = "folder:<path>"` plus `dataset_folder` (and `manifest` / `manifest_subset` become `null`); `--dump_frames` as `frame_dump`; `--monitor` as `stimulus_monitor`. `--windowed`, `--delay`, `--allow_model_mismatch` and `--allow_missing_conditions` are not recorded; the outcome of the two safety checks is recorded in `model_identity_verified` and `conditions_complete`.

## Inspecting Results Programmatically

Because all levels adhere to this unified schema, the full fidelity ladder can be compared across levels, architectures, and latency domains with a few lines of Python:

```python
import glob, json

for f in sorted(glob.glob("results/*/metrics_*.json")):
    with open(f, encoding="utf-8") as fh:
        d = json.load(fh)
    level = d["level"]
    m_name = d["model"]["name"]
    g = d["global_metrics"]
    sha = (d["model"]["sha256"] or "N/A")[:12]
    unresp = d.get("dataset", {}).get("n_unanswered", 0)

    # Hardware latency & on-chip verification (PIL / HIL)
    lat_str = ""
    if "latency" in d and "per_field" in d["latency"]:
        us_tot = d["latency"]["per_field"].get("US_TOTAL")
        if us_tot and us_tot.get("mean") is not None:
            lat_str = f" | {us_tot['mean']/1000.0:.2f} ms"

    verif_str = ""
    if level == "HIL":
        verified = d["model"].get("identity_verified")
        verif_str = f" [chip_verified={verified}]"

    print(f"{level:4} {m_name:40} acc={g['accuracy']:.4f} f1_macro={g['f1_macro']:.4f} "
          f"sha256={sha} drops={unresp}{lat_str}{verif_str}")
```
