# Hardware and Porting

Reference details that complement [Hardware Details](../README.md#hardware-details) in the README. Sizes use binary units (1 KiB = 1,024 bytes).

## Memory Configuration

The tensor arena is allocated in **SDRAM**,
not internal SRAM:

- `kTensorArenaSize = 4 * 1024 * 1024` (**4 MiB**), 16-byte aligned.
- Image receive buffer: `IMAGE_BUFFER_SIZE = 400 * 1024` (400 KiB), sized for up
  to 320×320 inputs.
- Op resolution uses `AllOpsResolver` (all TFLite-Micro ops registered) to avoid
  `AllocateTensors()` failures from missing operators, and to keep the firmware
  model-agnostic across the framework's supported architectures (MobileNetV2/V3,
  ResNet18, SqueezeNet) without per-model resolver maintenance — each architecture
  uses a different operator set (e.g. hard-swish/squeeze-excite in MobileNetV3,
  residual `Add` in ResNet18, `Concatenation` in SqueezeNet's fire modules), and
  `AllOpsResolver` avoids re-editing a fixed op list when switching models. This
  trades a larger firmware binary (all kernels linked, not just the ones the
  loaded model needs) for that flexibility — see the flash budget in the [README](../README.md#hardware-details).

## Upload Troubleshooting

**`Last page at 0x08... is not writeable`.** `arduino-cli upload`
failing with `exit status 64` and this message means the binary overflowed the
2 MiB flash — the address in the message is where the binary *would* have ended.
Subtract `0x08000000` to get its size. The usual cause is a stale `model.h` from a
larger model: regenerate it for the model you actually mean to deploy with
`python src/tflite_to_c.py models/tflite/{model_name}_int8.tflite`. The
`Invalid DFU suffix signature` warning that `dfu-util` prints alongside it is
unrelated and harmless.

## Portability and the SDRAM Constraint

Both host benches (`pil_benchmark.py` and `hil_camera_benchmark.py`) and the wire protocols (`#...@` byte-escaped framing for PIL; `READY_HIL` / `MODEL_FNV1A:<hex>` / `'T'` / `'F1'` for HIL) do not assume a specific MCU or camera peripheral: they communicate over standard USB CDC serial. A port must keep the Spanish end-of-inference lines that the benches search for (see [Serial Protocol](#serial-protocol-pil-bench)). In the firmware sketches (`pil_firmware.ino` and `hil_camera_firmware.ino`), the Cortex-M7 DWT cycle counter (`DWT->CYCCNT` / `SystemCoreClock`), the `FNV-1a` model hash, and the TFLite-Micro `Invoke()` loop are portable across any ARM Cortex-M4/M7 target; only two hardware-abstraction blocks (~15 lines total) are board-specific:
1. **Arena & buffer allocation (`SDRAM.begin()` / `SDRAM.malloc()`):** Porting to a board without external SDRAM (such as the **Arduino Nicla Vision**, which shares the same STM32H747XI MCU but has no external SDRAM chip) replaces `SDRAM.malloc()` with a static `alignas(16) uint8_t tensor_arena[...]` in internal SRAM. Compilation tests against `arduino:mbed_nicla:nicla_vision` (`700 KiB` PIL / `713 KiB` HIL flash) leave at most **~288 KiB** of contiguous internal SRAM for the tensor arena — sufficient for `MobileNet` (`177.2 KiB` arena) and `MobileNetV3Small` (`220.5 KiB` arena), but insufficient for `MobileNetV2` (`348.1 KiB`), `ResNet8` (`388.8 KiB`), or `SqueezeNet` (`404.7 KiB`). The Portenta H7's **8 MiB external SDRAM** is therefore required to evaluate all five deployable architectures under an identical 4 MiB arena configuration.
2. **Camera sensor driver (`#include "himax.h"` / `HM01B0`):** Porting `hil_camera_firmware.ino` to another camera shield or onboard sensor (e.g., the `GC2145` RGB sensor on the Nicla Vision) only replaces the sensor initialization and frame-grab call (`cam.begin(...)` and `cam.grabFrame(...)` + RGB565-to-grayscale luminance conversion) before `CYC_PRE` begins.

## Serial Protocol (PIL Bench)

> This section describes the **Processor-in-the-Loop (PIL)** protocol used by `pil_benchmark.py` / `pil_firmware.ino` (image injected over serial). The camera-in-the-loop **HIL** bench (`hil_camera_benchmark.py` / `hil_camera_firmware.ino`, see [`docs/scripts.md`](scripts.md)) uses a lighter single-byte trigger protocol (`READY_HIL` handshake and `MODEL_FNV1A:<hex>` model fingerprint at boot, then `'T'` to capture+infer and `'F1'`/`'F0'` to toggle frame-dump) — see that firmware's header comments for details.

The PIL bench streams a full image from the host to the Portenta H7 over USB
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
reverses this on reception (`pil_benchmark.py` performs the escaping on the host
side).

**Input handling.** Received bytes (uint8, 0–255) are written into the model
input tensor. For an INT8 model the firmware reads the tensor's own
`params.scale` and `params.zero_point` — set by the TFLite converter — and
applies `q = round(pixel / scale) + zero_point`, saturated to `[-128, 127]`.
Nothing is hardcoded, so the same firmware is correct for any quantized model;
`scale ≈ 1.0, zero_point = -128` reduces to the familiar `pixel - 128`. For a
FP32 model the firmware maps each pixel as `float = pixel / 255.0`; that branch is
not validated (see [Known Limitations](../README.md#known-limitations)). If fewer bytes than
the tensor expects arrive, the remainder is padded with `zero_point` (the
quantized representation of a real 0), not with a raw 0.

> The value fed in is the **raw** pixel, because the shipped models are
> converted without a `Rescaling` layer. This must match the preprocessing of
> the representative dataset used at conversion time — if you retrain with
> `Rescaling(1./255)` or MobileNet's `preprocess_input`, adjust `real_value` in
> `loadImageToInputTensor()` accordingly.

**Response.** After `Invoke()`, the board prints the predicted class index as a
single integer line (argmax of the output tensor). Diagnostic banners and the
on-chip CPU temperature (`°C`, from STM32H7 factory calibration) are also printed
around each inference; `pil_benchmark.py` parses the integer class from the stream.

**End-of-inference markers (porting note).** Both benches detect the end of each
inference by searching the firmware's text output for Spanish strings: `"listo"` or
`"siguiente imagen"` in PIL (case-insensitive), and `"Listo"`, `"siguiente imagen"` or
`"siguiente trigger"` in HIL (`END_MARKERS`). The firmware prints them in lines such as
`--- Listo. Esperando siguiente imagen... ---`. A port that does not print them gets a
timeout on every image, with no other error, so keep those lines when porting.
