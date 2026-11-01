# Results Schema

Single source of truth for the machine-readable result files in `results/`. Run
`python -m unittest discover -s tests -v` to check this document against every published JSON:
`tests/test_results_schema.py` fails if a key exists in a JSON but not here, or here but in no JSON.

**Table convention.** In every table of this document, the first column holds the full path of exactly
one key between backticks, with dots for nesting. The `metrics_*.json` sections give paths from the root
of the metrics JSON; the `hil_conditions_*.json` section gives paths from the root of the conditions JSON.

Dynamic keys use placeholders:

- `<FIELD>`: a telemetry field under `latencia.por_campo` (listed in section 7).
- `<CLASS>` under `metricas_por_clase`: the class names (`Class1`, `Class2`; `person`, `non_person` in
  VWW) and also the aggregate entries added by scikit-learn: `macro avg` and `weighted avg` (objects with
  `precision`, `recall`, `f1-score`, `support`) and `accuracy` (a number, not an object).
- `<CLASS>` under `conteo_predicciones`: a predicted class index as a string (`"0"`, `"1"`, ...).

## Metrics Schema Glossary (`metrics_*.json`)

Every evaluation script writes a `metrics_{model}[_int8].json` file capturing classification accuracy, execution metadata, and latency.

> **Note on Schema Keys:** JSON keys are maintained in Spanish (e.g., `nivel`, `n_sin_respuesta`, `identidad_verificada`, `conteo_predicciones`) across all 32 published files in `results/` to guarantee strict backward compatibility with existing analysis notebooks, data pipelines, and validation scripts. All field definitions, data types, and documentation below are in English.

### 1. Root & Metadata
| Key | Level | Type | Description |
|---|---|---|---|
| `nivel` | All | `string` | Evaluation level in the fidelity ladder: `"MIL"`, `"SIL"`, `"PIL"`, or `"HIL"`. |
| `generado_utc` | All | `string` | ISO 8601 UTC timestamp when the evaluation run was completed. |
| `modelo` | All | `object` | Model identification, checksums, and on-chip hardware verification (see Section 2). |
| `corrida` | PIL, HIL | `object` | Serial connection parameters and physical rig environment settings (see Section 4). |
| `dataset` | All | `object` | Dataset partition directories, sample counts, and class definitions (see Section 3). |
| `metricas_globales` | All | `object` | Global classification summary: accuracy, weighted/macro F1, precision, recall (see Section 5). |
| `metricas_por_clase` | All | `object` | Detailed classification breakdown per class (`precision`, `recall`, `f1-score`, `support`), plus macro/weighted averages (see Section 5). |
| `matriz_confusion` | All | `object` | Confusion matrix structure (`labels`, `matriz`, and layout orientation `orden`; see Section 6). |
| `latencia` | PIL, HIL | `object` | Hardware cycle and microsecond timer statistics per inference phase (see Section 7). |
| `conteo_predicciones` | PIL | `object` | Frequency dictionary mapping predicted class index strings (`"0"`, `"1"`, etc.) to total counts (see Section 6). |
| `validacion_cruzada` | HIL | `object` | Summary of paired PC MIL/SIL evaluation on the exact camera-captured frames (see Section 8). |
| `artefactos` | All | `object` | Relative filesystem paths to companion outputs (PNG heatmap, CSV tables, conditions JSON; see Section 9). |

### 2. Model Traceability (`modelo`)
| Key | Level | Type | Description |
|---|---|---|---|
| `modelo.nombre` | All | `string` | Model architecture tag / run identifier (e.g., `"MobileNetV2+32+20+0.0001+160+120_int8"`). |
| `modelo.path` | All | `string` | Relative path to the evaluated `.keras` or `.tflite` model artifact. |
| `modelo.sha256` | All | `string` | 64-character SHA-256 hash of the host model file: the `.keras` file in MIL and the `.tflite` file in SIL, PIL and HIL. In PIL it does not prove which model is flashed on the board (only HIL checks that, see `modelo.identidad_verificada`). |
| `modelo.bytes` | All | `integer` | Model file size on disk in bytes. |
| `modelo.fnv1a_tflite` | HIL | `string` | 8-character hex 32-bit FNV-1a hash computed directly from the host `.tflite` binary file. |
| `modelo.fnv1a_chip` | HIL | `string` | 8-character hex 32-bit FNV-1a hash computed by the Portenta H7 firmware over the compiled `g_model[]` in flash and emitted at boot (`MODEL_FNV1A:<hex>`). |
| `modelo.identidad_verificada` | HIL | `boolean` | `true` if the on-chip fingerprint and size match the host `.tflite` file. Confirms that the physical board ran the expected compiled model (a non-cryptographic check against stale or wrongly flashed firmware). |
| `modelo.nota` | PIL, HIL | `string` | Explanatory note detailing the verification method and scope of the checksum. |

### 3. Dataset & Sampling (`dataset`)
| Key | Level | Type | Description |
|---|---|---|---|
| `dataset.data_dir` | MIL, SIL | `string` | Path to the directory containing the evaluated images (`data/splits/test` or explicit folder). |
| `dataset.folder` | PIL | `string` | Path to the directory containing the evaluated images. |
| `dataset.splits_dir` | SIL | `string` | Root folder holding the reproducible dataset partitions (`data/splits`). |
| `dataset.stimulus_source` | HIL | `string` | Origin of the stimuli: `"manifest:<path>"` (default, test manifest) or `"folder:<path>"` (when `--folder` is used). |
| `dataset.manifest_subset` | HIL | `string\|null` | Filter applied to the manifest (default: `"hil_subset==1"`, selecting the 200 fixed test images; `"all-test"` with `--all-test`; `null` with `--folder`). |
| `dataset.class_names` | All | `array[string]` | List of class labels in canonical argmax index order matching firmware output. |
| `dataset.image_size` | MIL, SIL | `object` | Tensor input geometry. |
| `dataset.image_size.width` | MIL, SIL | `integer` | Input width in pixels. |
| `dataset.image_size.height` | MIL, SIL | `integer` | Input height in pixels. |
| `dataset.n_muestras` | MIL, SIL | `integer` | Total number of test images evaluated in the session. |
| `dataset.n_imagenes` | PIL | `integer` | Total number of test images sent to the board in the session. |
| `dataset.n_estimulos` | HIL | `integer` | Total number of optical stimuli presented in the session. |
| `dataset.n_respondidas` | PIL, HIL | `integer` | Number of test items for which the microcontroller successfully returned a class prediction. |
| `dataset.n_sin_respuesta` | PIL, HIL | `integer` | Number of test items where the board timed out or failed to return a prediction. Disentangles hardware/communication dropouts from classification errors. |

### 4. Run & Environmental Rig Configuration (`corrida`)
| Key | Level | Type | Description |
|---|---|---|---|
| `corrida.puerto` | PIL, HIL | `string` | Serial port used to communicate with the Arduino Portenta H7 (e.g., `"COM9"`). |
| `corrida.baud` | PIL, HIL | `integer` | Serial transmission speed in baud (default: `115200`). |
| `corrida.input_shape` | PIL | `object` | Geometry of the images sent over serial. |
| `corrida.input_shape.width` | PIL | `integer` | Image width in pixels. |
| `corrida.input_shape.height` | PIL | `integer` | Image height in pixels. |
| `corrida.input_shape.channels` | PIL | `integer` | Image channels: `1` for grayscale, `3` for RGB (`--color`). |
| `corrida.pre_delay_s` | PIL | `float` | Warm-up pause after opening the serial port before transmitting data (seconds). |
| `corrida.n_samples_pedidas` | PIL | `integer\|null` | PIL only: value of `--count` (number of images requested); `null` means every image in the folder. |
| `corrida.settle_s` | HIL | `float` | Display stabilization time after rendering each stimulus before triggering optical camera capture (default: `1.5` s). |
| `corrida.gap_s` | PIL, HIL | `float` | Inter-stimulus delay between successive evaluations (default: `0.5` s). |
| `corrida.seed` | HIL | `integer` | Random seed used for reproducible stimulus ordering (default: `42`). |
| `corrida.frame_dump` | HIL | `boolean` | `true` if raw 160×120 sensor frames were read back over serial for cross-level validation and image logging. |
| `corrida.condiciones_completas` | HIL | `boolean` | `true` if all mandatory environmental conditions (`illuminance_lux`, `camera_to_screen_distance_cm`, `ambient_temp_c`) were recorded. |
| `corrida.illuminance_lux` | HIL | `float` | Illuminance measured at the screen plane in lux (`10.0` lux in the primary protocol). |
| `corrida.camera_to_screen_distance_cm` | HIL | `float` | Distance from the camera lens to the monitor screen in cm (`24.0` cm in the primary protocol). |
| `corrida.ambient_temp_c` | HIL | `float` | Ambient room temperature measured at session start in °C. |
| `corrida.nota_correccion` | HIL | `string` | Optional post-hoc documentation note (e.g., documenting distance measurement reconciliation). Added by hand to a published file; the bench does not write it. |

### 5. Evaluation Metrics (`metricas_globales` & `metricas_por_clase`)
| Key | Level | Type | Description |
|---|---|---|---|
| `metricas_globales.accuracy` | All | `float` | Top-1 overall classification accuracy ($[0.0, 1.0]$). |
| `metricas_globales.precision_weighted` | All | `float` | Support-weighted average precision across all classes. |
| `metricas_globales.recall_weighted` | All | `float` | Support-weighted average recall across all classes. |
| `metricas_globales.f1_weighted` | All | `float` | Support-weighted average F1-score across all classes. |
| `metricas_globales.f1_macro` | All | `float` | Unweighted macro-average F1-score. On balanced splits it matches weighted F1; any divergence indicates minority-class performance drops. |
| `metricas_globales.precision_macro` | HIL | `float` | Unweighted macro-average precision across all classes. |
| `metricas_globales.recall_macro` | HIL | `float` | Unweighted macro-average recall across all classes. |
| `metricas_por_clase.<CLASS>` | All | `object\|float` | Per-class breakdown dictionary, or the aggregate entries described under *Table convention* (`accuracy` is a number). |
| `metricas_por_clase.<CLASS>.precision` | All | `float` | Precision of the class or aggregate. |
| `metricas_por_clase.<CLASS>.recall` | All | `float` | Recall of the class or aggregate. |
| `metricas_por_clase.<CLASS>.f1-score` | All | `float` | F1-score of the class or aggregate. |
| `metricas_por_clase.<CLASS>.support` | All | `float` | Sample count of the class or aggregate. |

### 6. Confusion Matrix & Prediction Counts (`matriz_confusion`, `conteo_predicciones`)
| Key | Level | Type | Description |
|---|---|---|---|
| `matriz_confusion.labels` | All | `array[string]` | List of class names matching matrix rows and columns in argmax order. |
| `matriz_confusion.matriz` | All | `array[array[integer]]` | 2D contingency table of predictions. Row index represents true class; column index represents predicted class. |
| `matriz_confusion.orden` | All | `string` | Clarification string confirming layout: `"filas = etiqueta real, columnas = etiqueta predicha"`. |
| `conteo_predicciones.<CLASS>` | PIL | `integer` | Total count of predictions for that class index string. |

### 7. Hardware Latency Profiling (`latencia`)
| Key | Level | Type | Description |
|---|---|---|---|
| `latencia.fuente` | PIL, HIL | `string` | Hardware timer description: ARM Cortex-M7 DWT cycle counter (`DWT->CYCCNT`) running at 400.0 MHz. |
| `latencia.n_imagenes_con_medida` | PIL, HIL | `integer` | Number of processed images with valid hardware latency telemetry lines. |
| `latencia.metodo_p95` | PIL, HIL | `string` | Method used for 95th percentile: nearest-rank over sorted samples without interpolation. |
| `latencia.por_campo` | PIL, HIL | `object` | Per-field timing summaries, one entry per `<FIELD>`. |
| `latencia.por_campo.<FIELD>` | PIL, HIL | `object` | Per-phase timing summary. |
| `latencia.por_campo.<FIELD>.n` | PIL, HIL | `integer` | Number of samples. |
| `latencia.por_campo.<FIELD>.media` | PIL, HIL | `float` | Mean. |
| `latencia.por_campo.<FIELD>.mediana` | PIL, HIL | `float` | Median. |
| `latencia.por_campo.<FIELD>.desv_std` | PIL, HIL | `float` | Sample standard deviation. |
| `latencia.por_campo.<FIELD>.min` | PIL, HIL | `float` | Minimum. |
| `latencia.por_campo.<FIELD>.max` | PIL, HIL | `float` | Maximum. |
| `latencia.por_campo.<FIELD>.p95` | PIL, HIL | `float` | 95th percentile (see `latencia.metodo_p95`). |

Values of `<FIELD>` (`CYC_*` in core cycles, `US_*` in microseconds):

- `CYC_CAPTURE`, `US_CAPTURE` (HIL): HM01B0 DCMI DMA frame acquisition duration.
- `CYC_PRE`, `US_PRE` (PIL, HIL): Sensor readout buffer transfer and input quantization time.
- `CYC_INF`, `US_INF` (PIL, HIL): Core TFLite-Micro neural network inference time (`Invoke()`).
- `CYC_POST`, `US_POST` (PIL, HIL): Post-processing, argmax identification, and telemetry formatting time.
- `CYC_TOTAL`, `US_TOTAL` (PIL, HIL): End-to-end processing latency from start of capture/reception to final output.
- `TEMP_C` (PIL, HIL): On-chip junction temperature reading from STM32H7 internal sensor in °C.
- `TS_MS` (HIL): Microcontroller system timestamp in milliseconds (`millis()`).

### 8. Cross-Validation (`validacion_cruzada`)
| Key | Level | Type | Description |
|---|---|---|---|
| `validacion_cruzada.sil_en_pc` | HIL | `string\|null` | Path to the INT8 `.tflite` model evaluated on the PC over the exact captured camera frame. |
| `validacion_cruzada.mil_en_pc` | HIL | `string\|null` | Path to the float `.keras` model evaluated on the PC over the exact captured camera frame. |
| `validacion_cruzada.n_con_sil` | HIL | `integer` | Number of frames successfully cross-evaluated against the PC SIL engine. |
| `validacion_cruzada.n_con_mil` | HIL | `integer` | Number of frames successfully cross-evaluated against the PC MIL engine. |

### 9. Output Artifacts (`artefactos`)
| Key | Level | Type | Description |
|---|---|---|---|
| `artefactos.matriz_png` | All | `string` | Relative path to the generated confusion matrix heatmap image (`Matriz_*.png` or `HIL_Confusion_Matrix_*.png`). |
| `artefactos.predicciones_csv` | MIL, SIL, PIL | `string` | Relative path to the per-image classification table (`predictions_*.csv`). |
| `artefactos.latencias_csv` | PIL, HIL | `string` | Relative path to the per-stimulus telemetry and timing record (`latency_metrics_*.csv` or `hil_latencies_*.csv`). |
| `artefactos.condiciones_json` | HIL | `string` | Relative path to the companion test rig environment record (`hil_conditions_*.json`). |

## Physical Rig Telemetry Schema (`hil_conditions_*.json`)

Alongside `metrics_*.json`, the HIL benchmark generates a standalone `hil_conditions_{model}_int8.json` file logging environmental variables and hardware-in-the-loop integrity:

| Key | Type | Description |
|---|---|---|
| `rig` | `string` | Rig configuration identifier (e.g., `"monitor-controlled-stimulus (checklist Fase 1C, opcion a)"`). |
| `port` | `string` | USB serial device address. |
| `baud` | `integer` | Serial communication baud rate. |
| `settle_s` | `float` | Display stabilization pause in seconds. |
| `gap_s` | `float` | Inter-stimulus gap in seconds. |
| `seed` | `integer` | Random seed controlling stimulus display sequence. |
| `stimulus_source` | `string` | Origin of the stimuli: `"manifest:<path>"` (default, test manifest) or `"folder:<path>"` (when `--folder` is used). |
| `manifest` | `string\|null` | Path to the dataset manifest; `null` when `--folder` is used. |
| `manifest_subset` | `string\|null` | Filter applied to the manifest (`"hil_subset==1"`, or `"all-test"`); `null` when `--folder` is used. |
| `dataset_folder` | `string\|null` | Folder passed with `--folder`; `null` when the stimuli come from the manifest. |
| `class_names` | `array[string]` | Class names in label order (the list index is the integer label returned by the firmware). |
| `n_stimuli` | `integer` | Number of stimuli presented in the run. |
| `notes` | `string` | User-provided laboratory notes (monitor model, lighting fixtures, etc.). |
| `frame_dump` | `boolean` | `true` when the captured frames were sent back and saved (`--dump-frames`, also implied by `--sil-model` / `--mil-model`). |
| `frames_dir` | `string\|null` | Folder where the dumped frames were written; `null` when `frame_dump` is `false`. |
| `sil_model` | `string\|null` | Path to the companion INT8 model executed on the host for paired cross-validation. |
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

Some CLI options are recorded indirectly or not at all: `--all-test` is recorded as `manifest_subset = "all-test"` (otherwise `"hil_subset==1"`); `--folder` as `stimulus_source = "folder:<path>"` plus `dataset_folder` (and `manifest` / `manifest_subset` become `null`); `--dump-frames` as `frame_dump`. `--windowed`, `--delay`, `--allow-model-mismatch` and `--allow-missing-conditions` are not recorded; the outcome of the two safety checks is recorded in `model_identity_verified` and `conditions_complete`.
