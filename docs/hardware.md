# Hardware and Porting

Reference details that complement [Hardware Details](../README.md#hardware-details) in the README. Sizes use binary units (1 KiB = 1,024 bytes).

## Boards and Memory

One sketch per level serves both boards. What depends on the board is decided in the `BOARD SUPPORT` block at
the top of each sketch, selected by the build target (`--board` in `compile_upload_arduino.py`):

| | Arduino Portenta H7 (`--board portenta`) | Arduino Nicla Vision (`--board nicla`) |
|---|---|---|
| Build target | `arduino:mbed_portenta:envie_m7:split=100_0` | `arduino:mbed_nicla:nicla_vision` |
| Tensor arena | 4 MiB in external SDRAM, 16-byte aligned | Static array in internal RAM: 368 KiB by default |
| Image buffer | 400 KiB in external SDRAM (up to 320×320 RGB) | Static array: 19,200 bytes by default (one 160×120 grayscale image) |
| Camera (HIL) | HM01B0 of the Vision Shield, monochrome, 160×120 | Built-in GC2145, colour: read as RGB565 at 160×120 and converted to luminance |
| Announces | `BOARD:PORTENTA_H7` | `BOARD:NICLA_VISION` |
| Status | Board behind every published result | Builds; not run on a board yet |

The Portenta H7 FQBN needs `:split=100_0` (2 MiB of flash for the M7 core): with the default split the firmware
does not fit.

**Sizes set at build time.** `--arena_kb` and `--image_buffer_bytes` of `compile_upload_arduino.py` (the macros
`TENSOR_ARENA_KB` and `IMAGE_BUFFER_BYTES` of the sketch) change the two sizes without editing the sketch. On the
Nicla Vision both are static arrays, so the linker accounts for them: sizes that do not fit fail at build time
(`region RAM overflowed`), and an arena too small for the model fails at boot, in `AllocateTensors()`, with a
message that says so. A model input that does not fit the image buffer also stops at boot with a message.

**RAM budget on the Nicla Vision.** The board has no external SDRAM: 523,624 bytes of internal RAM hold the arena,
the buffers, the camera frame and everything else. Reported by the compiler for the default sizes, with the
`model.h` in the repository (`MobileNetV2`, 630.5 KiB):

| Sketch | Flash (of 1,966,080 bytes) | Global RAM (of 523,624 bytes) | Left for the heap and the rest |
|---|---|---|---|
| `pil_firmware` | 1,027,440 (52%) | 455,632 (87%) | 67,992 |
| `hil_firmware` | 1,040,880 (52%) | 475,752 (90%) | 47,872 |
| `pil_firmware`, `--arena_kb 288` | 1,027,440 (52%) | 373,712 (71%) | 149,912 |

The arena each deployable model needs was measured on the Portenta H7 (`ARENA_USED`): `MobileNet` 177.2 KiB,
`MobileNetV3Small` 220.5 KiB, `MobileNetV2` 348.1 KiB, `ResNet8` 388.8 KiB, `SqueezeNet` 404.7 KiB. The default
368 KiB therefore covers the first three, with 20 KiB to spare for `MobileNetV2`. `ResNet8` and `SqueezeNet` need
`--arena_kb 392` and `--arena_kb 408`, which takes another 24 KiB and 40 KiB from the last column. These are
build-time figures: whether what is left is enough at run time is only known by running it.

**Operators.** Op resolution uses `AllOpsResolver` (all TFLite-Micro ops registered) to avoid
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

## Portability

Both host benches (`pil_benchmark.py` and `hil_benchmark.py`) and the wire protocols (`#...@` byte-escaped framing for PIL; `READY_HIL` / `MODEL_FNV1A:<hex>` / `'T'` / `'F1'` for HIL) do not assume a specific MCU or camera peripheral: they communicate over standard USB CDC serial. A port must keep the end-of-inference lines that the benches search for (see [Serial Protocol](#serial-protocol-pil-bench)). In the firmware sketches (`pil_firmware.ino` and `hil_firmware.ino`), the Cortex-M7 DWT cycle counter (`DWT->CYCCNT` / `SystemCoreClock`), the `FNV-1a` model hash, and the TFLite-Micro `Invoke()` loop are portable across ARM Cortex-M4/M7 targets. What is board-specific sits in the `BOARD SUPPORT` block of each sketch, and the Nicla Vision is the worked example of a second board:
1. **Arena and buffers.** On the Portenta H7 they are allocated in external SDRAM (`SDRAM.begin()` / `ea_malloc()`); on the Nicla Vision, which shares the STM32H747 MCU but has no SDRAM chip, they are static arrays in internal RAM (see [Boards and Memory](#boards-and-memory)).
2. **Camera.** The Portenta H7 reads its monochrome `HM01B0` as 8-bit grayscale. The Nicla Vision's `GC2145` is a colour sensor, and in the `arduino:mbed_nicla` 4.6.0 core its grayscale mode delivers raw Bayer data rather than luminance, so the sketch reads RGB565 and converts it in the MCU with the BT.601 weights in integer arithmetic. The conversion gives the same value as Pillow's `convert('L')` for every RGB565 input, so "grayscale" means the same on the board and on the host. Its time is counted in the `PRE` phase, and the frame-dump returns the luminance frame.

Adding a third board takes one more block in each sketch and one entry in `src/boards.py`.

**Not yet verified on a Nicla Vision.** The sketches build for it and the host side is tested against a simulated board, but nothing has run on the physical board. Until it does, these remain open: the byte order of the RGB565 frame (taken from the core's own viewers; a wrong order would show as a visibly corrupted frame-dump), the image the GC2145 gives at 160×120 (according to the driver, a central 640×480 window subsampled by four, so a narrower field of view than the full sensor), whether the RAM left is enough at run time, whether `AllocateTensors()` succeeds for each model, the temperature reading, and the clock (`SystemCoreClock` is printed at boot).

## Serial Protocol (PIL Bench)

> This section describes the **Processor-in-the-Loop (PIL)** protocol used by `pil_benchmark.py` / `pil_firmware.ino` (image injected over serial). The camera-in-the-loop **HIL** bench (`hil_benchmark.py` / `hil_firmware.ino`, see [`docs/scripts.md`](scripts.md)) uses a lighter single-byte trigger protocol (`READY_HIL` handshake and `MODEL_FNV1A:<hex>` model fingerprint at boot, then `'T'` to capture+infer and `'F1'`/`'F0'` to toggle frame-dump) — see that firmware's header comments for details.

The PIL bench streams a full image from the host to the board over USB
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

**Board announcement.** Each sketch prints `BOARD:<name>` (`PORTENTA_H7` or
`NICLA_VISION`): at boot in both, and again with every image in PIL, because the PIL
bench opens the port after the boot lines are gone. The benches record the name in
`run.board` and stop if it is not the board given with `--board`. A firmware that
prints no such line is accepted and recorded as `null`.

**End-of-inference markers (porting note).** Both benches detect the end of each
inference by searching the firmware's text output, case-insensitively, for
`"next image"` (PIL) or `"next trigger"` (HIL), which the sketches print in the lines
`--- Done. Waiting for the next image... ---` and
`--- Done. Waiting for the next trigger... ---`. A port that does not print them gets a
timeout on every image, with no other error, so keep those lines when porting, and do
not use those words in any other line. The markers live in `src/boards.py`
(`END_MARKERS`); the Spanish ones the sketches printed until November 2026
(`"siguiente imagen"`, `"siguiente trigger"`, `"listo"`) are still accepted, so a board
flashed with an earlier build works without being reflashed.
