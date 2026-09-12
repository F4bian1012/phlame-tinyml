# Secciones nuevas para el README — listas para pegar

> Redactadas a partir del `hil_firmware.ino` real del branch (270 líneas, verificado).
> El contenido de las secciones está en inglés para mantener la coherencia con el resto
> del README. Colócalas donde se indica.

---

## 1) Serial Protocol (Hardware-in-the-Loop Bench)

*Va después del diagrama Mermaid, como sección propia `## Serial Protocol` (añádela también al TOC).*

```markdown
## Serial Protocol (HIL Bench)

The HIL bench streams a full image from the host to the Portenta H7 over USB
Serial, runs on-device inference, and reads the predicted class back. The wire
format is a framed raw-byte protocol:

| Element        | Value            | Meaning                                        |
|----------------|------------------|------------------------------------------------|
| Start marker   | `#` (`0x23`)     | Begin of image packet                          |
| End marker     | `@` (`0x40`)     | End of image packet → triggers inference       |
| Escape byte    | `ESC` (`0x1B`)   | Next byte is real data, recovered as `b ^ 0x20`|
| Baud rate      | `115200`         | `Serial.begin(115200)`                         |

**Encoding rule.** Image pixels are sent raw between the markers. If a pixel byte
equals `#`, `@`, or `ESC`, the sender escapes it as `ESC` followed by
`byte ^ 0x20`, so control bytes never appear inside the payload. The firmware
reverses this on reception (`hil_benchmark.py` performs the escaping on the host
side).

**Input handling.** Received bytes (uint8, 0–255) are written into the model
input tensor. For an INT8 model each pixel is mapped as `int8 = pixel - 128`;
for a FP32 model as `float = pixel / 255.0`. If fewer bytes than the tensor
expects arrive, the remainder is zero-padded.

**Response.** After `Invoke()`, the board prints the predicted class index as a
single integer line (argmax of the output tensor). Diagnostic banners and the
on-chip CPU temperature (`°C`, from STM32H7 factory calibration) are also printed
around each inference; `hil_benchmark.py` parses the integer class from the stream.
```

---

## 2) Hardware Requirements

*Va justo antes de `## Quickstart & Usage` (añádela al TOC).*

```markdown
## Hardware Requirements

| Component        | Specification                                                        |
|------------------|---------------------------------------------------------------------|
| Board            | Arduino Portenta H7 (STM32H747XI, dual-core Cortex-M7 @ 480 MHz)     |
| External RAM     | 8 MB SDRAM (required — tensor arena + image buffer live here)        |
| Camera (capture) | Portenta Vision Shield, HM01B0 monochrome sensor (160×120, 30 fps)   |
| Flash            | 2 MB internal (holds the quantized model via `model.h`)              |
| Host link        | USB-C, 115200 baud serial                                            |

**Memory configuration (firmware).** The tensor arena is allocated in **SDRAM**,
not internal SRAM:

- `kTensorArenaSize = 4 * 1024 * 1024` (**4 MB**), 16-byte aligned.
- Image receive buffer: `IMAGE_BUFFER_SIZE = 400 * 1024` (400 KB), sized for up
  to 320×320 inputs.
- Op resolution uses `AllOpsResolver` (all TFLite-Micro ops registered) to avoid
  `AllocateTensors()` failures from missing operators.

**Toolchain.** Arduino CLI + `arduino:mbed_portenta` core + `Chirale_TensorFlowLite`
library (see `requirements.txt`).
```

---

## 3) Results

*Va después del Quickstart o al final, como `## Results` (añádela al TOC). **Rellena las
celdas `TBD` con tus cifras reales** — los CSV del banco y las matrices de confusión INT8
del repo tienen los números.*

```markdown
## Results

Binary classification (Class1 vs Class2), grayscale input. INT8 models evaluated
on the Portenta H7 via the HIL bench.

| Model        | Input Res | Params | .tflite size | Acc FP32 | Acc INT8 | On-device latency | Arena used |
|--------------|-----------|--------|--------------|----------|----------|-------------------|------------|
| MobileNetV2  | 160×120   | TBD    | TBD          | TBD      | TBD      | TBD ms            | TBD        |
| MobileNetV2  | 320×240   | TBD    | TBD          | TBD      | TBD      | TBD ms            | TBD        |
| MobileNetV2  | 320×320   | TBD    | TBD          | TBD      | TBD      | TBD ms            | TBD        |

Confusion matrices for the INT8 models are in `models/tflite/` and training
curves in `tensorboard_logs/`.
```

---

## 4) Data Availability & Known Limitations

*Va al final, antes de `## Citation` (añádela al TOC).*

```markdown
## Data Availability

The dataset consists of grayscale images for a binary classification task
(Class1 vs Class2). Place raw images under `data/raw/<ClassName>/`; the
`data/` tree is git-ignored (only `.gitkeep` is tracked). [Add here: dataset
source / download link / license, or state "available on request".]

## Known Limitations

- **Binary task.** Current models are trained and evaluated on two classes;
  `models/class_names.txt` defines the label order.
- **Fixed quantization offset.** The firmware maps input pixels with a hardcoded
  `pixel - 128` (assumes input zero-point = -128). Models whose input
  quantization differs will need the real `scale`/`zero_point` applied instead.
- **Latency timing.** The inference loop currently includes a `delay(1000)`
  before each inference (temperature settling); subtract it — or measure with the
  Cortex-M7 DWT cycle counter (`DWT->CYCCNT`) — before reporting inference latency.
- **Class index only.** The board returns the argmax class, not per-class
  confidence; add score output to the firmware if confidence-level comparison is
  needed.
- **Op resolver size.** `AllOpsResolver` maximizes compatibility at the cost of
  flash/RAM; switching to a `MicroMutableOpResolver` with only the required ops
  reduces footprint for a final deployment build.
```
