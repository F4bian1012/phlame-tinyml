# Hardware and Porting

The hardware reference of PHLAME: the rig of the published results, the flash budget, the two boards and their memory, upload troubleshooting, porting, the HIL rig and the serial protocol. Sizes use binary units (1 KiB = 1,024 bytes; 1 MiB = 1,024 KiB), like `results/tables/`.

## Hardware Details

The Arduino Portenta H7 set-up; the Arduino Nicla Vision is described under [Boards and Memory](#boards-and-memory).

| Component        | Specification                                                        |
|------------------|---------------------------------------------------------------------|
| Board            | Arduino Portenta H7 (STM32H747XI, dual-core Cortex-M7 nominal up to 480 MHz; runs at `SystemCoreClock = 400.0 MHz`, 2.50 ns/cycle under `arduino:mbed_portenta`, DWT 32-bit overflow ~10.74 s) |
| External RAM     | 8 MiB SDRAM (required — tensor arena + image buffer live here)        |
| Camera (capture) | Portenta Vision Shield, HM01B0 monochrome sensor (160×120, 30 fps)   |
| Thermal control  | Active forced-air cooling (constant-speed fan over Portenta H7 + Vision Shield) |
| Flash            | 2 MiB internal, shared by firmware and `model.h`                      |
| Host link        | USB-C, 115200 baud serial                                            |

## Flash Budget

**This is a hard limit.** The model is linked into the firmware as
a C array, so the quantized model and the firmware code share the same 2 MiB of
internal flash. Measured on a build of the PIL firmware with `AllOpsResolver`
([`flash_budget_build.txt`](flash_budget_build.txt) records the build):

| | Size |
|---|---|
| Internal flash (STM32H747, `0x08000000`–`0x081FFFFF`) | 2048 KiB |
| Bootloader region (the application starts at `0x08040000`) | 256 KiB |
| Firmware code (TFLite-Micro + all kernels, no model) | ~418 KiB |
| **Headroom left for `model.h`** | **~1374 KiB** |

The HIL firmware carries 14 KiB more code, which leaves ~1360 KiB.

Which of the INT8 models of the COCO benchmark fit, at 160×120. "Flash used" counts the bootloader region,
the PIL firmware code and the model. The build measured carries the `MobileNetV2` of the repository's `model.h`
(1,335,488 bytes in all, 645,640 of them the model); each row is that budget plus the `.tflite` of the model:

| Model | `.tflite` | Flash used | Fits |
|---|---|---|---|
| `MobileNet` | 311 KiB | 0.96 MiB | yes |
| `MobileNetV2` | 630 KiB | 1.27 MiB | yes |
| `SqueezeNet` | 809 KiB | 1.45 MiB | yes |
| `MobileNetV3Small` | 816 KiB | 1.45 MiB | yes |
| `ResNet8` | 1249 KiB | 1.88 MiB | yes, 125 KiB to spare |
| `MobileNetV3Large` | 2180 KiB | 2.79 MiB | **no** — 0.79 MiB over |
| `ResNet18` | 11057 KiB | 11.46 MiB | **no** — 9.46 MiB over |

`MobileNetV3Large` and `ResNet18` train and quantize fine, and are valid MIL/SIL
targets on the PC, but they cannot be deployed from internal flash: the `.tflite`
alone exceeds it (`MobileNetV3Large`) or exceeds it several times over (`ResNet18`).
Options are running the model from external memory (QSPI flash or SDRAM, not
implemented in PHLAME) or a narrower variant (lower width multiplier), which is a
different model. Unstructured pruning (`prune_model.py`) does not help because tensor
shapes are kept, and a lower input resolution does not either because these networks
end in global average pooling: it shrinks the arena, not the model. For a borderline
model whose `.tflite` does fit, swapping `AllOpsResolver` for a
`MicroMutableOpResolver` listing only the operators that model uses would free flash
(not implemented in PHLAME).

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
| Status | PIL and HIL measured on the board | PIL measured on the board; HIL builds, not run yet |

The Portenta H7 FQBN needs `:split=100_0` (2 MiB of flash for the M7 core): with the default split the firmware
does not fit. Of those 2 MiB the first 256 KiB are the bootloader region; the sizes of the two sketches, read
from a build, are in [`flash_budget_build.txt`](flash_budget_build.txt).

**What has run on a board.** The results published before November 2026 (the Visual Wake Words run and the vehicle
application case) were measured with the single-board sketches that preceded these. The two-board PIL sketch was then checked on a Portenta H7 against its predecessor, following steps 5 and 6
of the README with the same model and the same 300 test images: both gave the same prediction for all 300 images,
and the bench recorded `BOARD:PORTENTA_H7`. The two-board HIL sketch first ran on a
Portenta H7 on 2026-11-15 (see [HIL Rig and Session](#hil-rig-and-session)), and has not been run on a Nicla Vision.

**First runs on a Nicla Vision** (2026-11-14, `results/nicla/pil/`). The PIL sketch boots on the board, allocates
the default 368 KiB arena (`MobileNetV2` uses 356,404 bytes of it, the same as on the Portenta H7) and reports the
same 400 MHz clock. With the `MobileNetV2` of the vehicle application case ([`vehicle_benchmark.md`](vehicle_benchmark.md)) it answered the 940 test images
without a dropped link, and gave the same prediction as a Portenta H7 measured the same day for all 940 (937 of
940 against the Portenta H7 file published in September, for the reason given under
[Serial Protocol](#serial-protocol-pil-bench): the pixels the host sent were no longer the same for three files).
One difference stands out: the inference takes 128.8 million cycles on the Nicla Vision and 160.5 million on the
Portenta H7, 19.7% fewer, with the same MCU, clock and model. The arena is in internal RAM on one board and in
external SDRAM on the other, which is the likely cause; it has not been isolated.

**The COCO benchmark on both boards** (2026-11-14 and 15, `results/pil/` and `results/nicla/pil/`). The five
models that fit were measured on each board over the 5,000 test images: ten runs, every image answered, no
dropped link, and the two boards give the same prediction for every image of every model. Both boards carry the same
STM32H747, so that agreement is a consistency check between two boards, not evidence of portability to other
microcontroller families. Arena, flash and RAM
are those of each flashed build (`ARENA_USED` in the boot lines, sizes reported by the compiler):

| Model | Arena used (bytes) | `--arena_kb` on the Nicla Vision | Nicla Vision: flash / RAM left (bytes) | `INF` Portenta H7 (ms) | `INF` Nicla Vision (ms) |
|---|---:|---:|---:|---:|---:|
| `MobileNet` | 181,492 | default (368) | 700,144 / 67,992 | 241.40 | 193.45 |
| `MobileNetV2` | 356,404 | default (368) | 1,026,512 / 67,992 | 401.15 | 322.06 |
| `MobileNetV3Small` | 225,812 | default (368) | 1,217,632 / 67,992 | 489.59 | 414.81 |
| `SqueezeNet` | 414,404 | 408 | 1,210,632 / 27,032 | 1153.63 | 987.00 |
| `ResNet8` | 398,164 | 392 | 1,661,040 / 43,416 | 2892.09 | 2561.31 |

`SqueezeNet` and `ResNet8` ran the 5,000 images on the Nicla Vision with 27 KB and 43 KB of RAM left. The Nicla
Vision needs 11% to 20% fewer cycles per inference than the Portenta H7 for the same file.

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
build-time figures; with them both models ran the 5,000 images of the COCO benchmark on the board (see above).

**Operators.** Op resolution uses `AllOpsResolver` (all TFLite-Micro ops registered) to avoid
`AllocateTensors()` failures from missing operators, and to keep the firmware
model-agnostic across the framework's supported architectures (MobileNetV2/V3,
ResNet18, SqueezeNet) without per-model resolver maintenance — each architecture
uses a different operator set (e.g. hard-swish/squeeze-excite in MobileNetV3,
residual `Add` in ResNet18, `Concatenation` in SqueezeNet's fire modules), and
`AllOpsResolver` avoids re-editing a fixed op list when switching models. This
trades a larger firmware binary (all kernels linked, not just the ones the
loaded model needs) for that flexibility — see the [flash budget](#flash-budget).

**Kernels on the chip.** The TFLite-Micro library the sketches build with (`Chirale_TensorFlowLite` 2.0.0) carries
the CMSIS-NN versions (CMSIS-NN V.11.0.0) of `ADD`, `CONV_2D`, `DEPTHWISE_CONV_2D`, `FULLY_CONNECTED`, `MUL`, the
pooling operators, `SOFTMAX` and `SVDF`, and no reference version of them, so those are the kernels both boards run
for these operators. The sketches are built for `-mcpu=cortex-m7`, for which CMSIS-NN enables its DSP code paths
(`ARM_MATH_DSP`), and the linker map of the firmware holds `arm_convolve_wrapper_s8`,
`arm_depthwise_conv_wrapper_s8`, `arm_fully_connected_s8`, `arm_elementwise_add_s8`, `arm_elementwise_mul_s8`,
`arm_avgpool_s8`, `arm_max_pool_s8` and `arm_softmax_s8`. Every other operator (hard-swish, mean, pad,
concatenation and the rest) runs the TFLite-Micro reference kernel. Which CMSIS-NN variant each layer takes at run
time was not traced. The CMSIS-NN headers describe these kernels as bit-accurate with TensorFlow Lite Micro. The
PIL runs of the COCO benchmark put that to the test on hardware, against the reference kernels of TFLite on the PC:
the predictions are the same for 4,975 to 4,999 of 5,000 images in four models and for 4,754 in `MobileNetV3Small`
(see [Results](../README.md#results), section 1).

## Upload Troubleshooting

**`Last page at 0x08... is not writeable`.** `arduino-cli upload`
failing with `exit status 64` and this message means the binary overflowed the
2 MiB flash — the address in the message is where the binary *would* have ended.
Subtract `0x08000000` to get its size. The usual cause is a stale `model.h` from a
larger model: regenerate it for the model you actually mean to deploy with
`python src/tflite_to_c.py models/tflite/{model_name}_int8.tflite`. The
`Invalid DFU suffix signature` warning that `dfu-util` prints alongside it is
unrelated and harmless.

**`No DFU capable USB device available`.** `arduino-cli upload` failing with
`exit status 74` and this message means the board did not enter its bootloader
when the upload asked it to. Put it there by hand: press the reset button twice
in quick succession (the green LED fades in and out), then run the same command
again. While in the bootloader the board appears on a different serial port; it
returns to its usual port once the firmware is running, and that one is the port
to give the benches.

## Portability

Both host benches (`pil_benchmark.py` and `hil_benchmark.py`) and the wire protocols (`#...@` byte-escaped framing, `READY_PIL` and `MODEL_FNV1A:<hex>` for PIL; `READY_HIL` / `MODEL_FNV1A:<hex>` / `'T'` / `'F1'` for HIL) do not assume a specific MCU or camera peripheral: they communicate over standard USB CDC serial. A port must keep the end-of-inference lines that the benches search for (see [Serial Protocol](#serial-protocol-pil-bench)). In the firmware sketches (`pil_firmware.ino` and `hil_firmware.ino`), the Cortex-M7 DWT cycle counter (`DWT->CYCCNT` / `SystemCoreClock`), the `FNV-1a` model hash, and the TFLite-Micro `Invoke()` loop are portable across ARM Cortex-M4/M7 targets. What is board-specific sits in the `BOARD SUPPORT` block of each sketch, and the Nicla Vision is the worked example of a second board:
1. **Arena and buffers.** On the Portenta H7 they are allocated in external SDRAM (`SDRAM.begin()` / `ea_malloc()`); on the Nicla Vision, which shares the STM32H747 MCU but has no SDRAM chip, they are static arrays in internal RAM (see [Boards and Memory](#boards-and-memory)).
2. **Camera.** The Portenta H7 reads its monochrome `HM01B0` as 8-bit grayscale. The Nicla Vision's `GC2145` is a colour sensor, and in the `arduino:mbed_nicla` 4.6.0 core its grayscale mode delivers raw Bayer data rather than luminance, so the sketch reads RGB565 and converts it in the MCU with the BT.601 weights in integer arithmetic. The conversion gives the same value as Pillow's `convert('L')` for every RGB565 input, so "grayscale" means the same on the board and on the host. Its time is counted in the `PRE` phase, and the frame-dump returns the luminance frame.

Adding a third board takes one more block in each sketch and one entry in `src/boards.py`; [`extending.md`](extending.md#adding-a-board) lists what each of them holds and which tests notice.

**Not yet verified on a Nicla Vision.** The PIL sketch has run on the board (see [Boards and Memory](#boards-and-memory)); the HIL sketch has not. Until it does, these remain open: the byte order of the RGB565 frame (taken from the core's own viewers; a wrong order would show as a visibly corrupted frame-dump), the image the GC2145 gives at 160×120 (according to the driver, a central 640×480 window subsampled by four, so a narrower field of view than the full sensor), and whether the RAM left is enough for the camera frame at run time. The five models that fit have run on the board at PIL.

## HIL Rig and Session

The HIL bench shows each stimulus full-screen on a monitor that the camera of the board looks at. Three things about that set-up are handled on the host:

- **Which monitor.** The stimulus window opens where the system puts it. `--monitor N` (bench and calibrator; Windows only) sends it to the N-th monitor, the primary one being 0, and the conditions file records the choice (`stimulus_monitor`). The option makes the process DPI-aware, so that the monitor rectangles are in physical pixels. On a display scaled above 100% that can change how large and how sharp the stimulus is drawn compared with a session without the option; this has not been measured. Calibrate and measure with the same value.
- **Alignment.** `calibrate_hil_rig.py` shows its patterns through the same window and reads the captured frames with the same protocol (`'T'`, `'F1'`), so it measures what the bench will see. Its acceptance criteria, at the 160×120 of the frame: centre shift up to 2 px, scale within 3%, rotation up to 1°, fit residual up to 1.5 px; eight gray bands in increasing order with a range of at least 150 levels; corner-to-centre ratio of a flat field of at least 0.60, with its mean varying by at most 2 levels (standard deviation) between captures and at most 2% of saturated pixels; and a fine-to-coarse checkerboard contrast of at least 0.50. They are reported as PASS or WARN in `results/hil/calibration/calibration_{UTC time}.json`, never enforced: keep the report with the results of the session.
- **A session that ends early.** Every stimulus is written to the latency CSV as soon as it is measured. If the serial link drops, the board stops answering, the operator presses Ctrl+C or anything else fails, the bench still writes the conditions file and, when at least one stimulus was answered, the metrics and the confusion matrix, all marked `completed: false` with the number of stimuli planned; it then ends with exit code 3. Unlike the PIL bench it does not reopen the port and repeat the stimulus. A key pressed in the stimulus window during the settle time is ignored: before November 2026 it ended the wait (that is what `cv2.waitKey` does) and the capture was triggered early.

**What has run on the physical rig** (2026-11-15, Portenta H7). The calibrator, in both modes, and the bench (a 20-image session of the COCO `MobileNetV2`, with the model fingerprint verified and every frame also classified on the PC) ran with `--monitor 1`: the stimulus window opened on the second monitor and the camera captured it. Not exercised on the rig: a session that ends early, a key pressed during the settle time, and the effect of DPI awareness on the size of the stimulus (no session was run without `--monitor` to compare). The tests cover the first two against a simulated board and window (`tests/test_hil_session.py`).

**When the board does not answer.** The HIL sketch prints its boot lines once, after a restart, and stops for good when the camera does not initialize (`CAM_INIT:FAIL`: on the Portenta H7, a Vision Shield that is not fully seated on its two connectors does this, and moving the camera can unseat it). The bench must therefore be the first to open the port after a restart, which flashing provides. The calibrator does not need that: when the firmware is silent it asks for RESET, stays on the port while it disappears, opens it again and reads the boot lines (`tests/test_calibration_restart.py`). A firmware left in the middle of a frame by Ctrl+C was seen to stop answering until the next restart; the cause was not traced.

## Serial Protocol (PIL Bench)

> This section describes the **Processor-in-the-Loop (PIL)** protocol used by `pil_benchmark.py` / `pil_firmware.ino` (image injected over serial). The camera-in-the-loop **HIL** bench (`hil_benchmark.py` / `hil_firmware.ino`, see [`docs/scripts.md`](scripts.md)) uses a lighter single-byte trigger protocol (`READY_HIL` handshake and `MODEL_FNV1A:<hex>` model fingerprint at boot, then `'T'` to capture+infer and `'F1'`/`'F0'` to toggle frame-dump) — see that firmware's header comments for details.

The PIL bench streams a full image from the host to the board over USB
Serial, runs on-device inference, and reads the predicted class back. The wire
format is a framed raw-byte protocol:

| Element        | Value            | Meaning                                        |
|----------------|------------------|------------------------------------------------|
| Start marker   | `#` (`0x23`)     | Begin of image packet                          |
| End marker     | `@` (`0x40`)     | End of image packet → triggers inference       |
| Empty packet   | `#@`             | Asks for the identity lines; runs no inference |
| Escape byte    | `ESC` (`0x1B`)   | Next byte is real data, recovered as `b ^ 0x20`|
| Baud rate      | `115200`         | `Serial.begin(115200)`                         |

**Encoding rule.** Image pixels are sent raw between the markers. If a pixel byte
equals `#`, `@`, or `ESC`, the sender escapes it as `ESC` followed by
`byte ^ 0x20`, so control bytes never appear inside the payload. The firmware
reverses this on reception (`pil_benchmark.py` performs the escaping on the host
side).

**Identity lines and handshake.** The sketch says what it is running in six parseable lines:
`BOARD:<name>`, `INPUT_SHAPE:<H>x<W>x<C>` (the input tensor of the model), `MODEL_BYTES:<int>` and
`MODEL_FNV1A:<hex>` (size and 32-bit FNV-1a fingerprint of the model array in flash, which is the
`.tflite` byte for byte), `ARENA_USED:<bytes>` and, last, `READY_PIL`. It prints them at the end of
its boot lines, which appear once, when the port is opened for the first time after a reset, and
again whenever the host sends an empty packet (`#@`). Because `#` restarts a packet, that request is
understood whatever the firmware was waiting for. The bench reads the boot lines if they come and
asks with `#@` otherwise, and then stops before sending any image unless the board is the one given
with `--board`, the fingerprint and the size are those of the `.tflite` given with `--model_path`,
and the images have the announced shape. A build earlier than November 2026 has no identity lines:
it takes the empty packet for an image and answers with an inference, which is how the bench
recognises it (it asks twice, since a board left in the middle of a packet by an interrupted run
answers the first request the same way). Such a build is refused unless the run is allowed with
`--allow_model_mismatch`; a board that answers nothing ends the run with exit code 1, and the HIL
sketch is recognised by its `READY_HIL`.

> **Checked on both boards** (2026-11-15, COCO `MobileNetV2`). The identity lines, the empty packet and
> the checks of the bench were added after the COCO runs were measured, so those runs do not carry
> them. On a Portenta H7 and on a Nicla Vision: the boot lines end with the six identity lines; the
> fingerprint on the chip equals the one the host computes for the flashed `.tflite`; the bench took
> the identity from the boot lines when it was the first to open the port (Portenta H7) and through
> `#@` otherwise (Nicla Vision); a run of 300 images completed with `model.identity_verified = true`
> and gave, on each board, the same prediction as the earlier firmware for all 300. On the Nicla
> Vision a board carrying another model and a run with another image size were refused before
> anything was sent. Not exercised on a board: `#@` sent by hand in the middle of a run, a link lost
> for longer than the reconnect window, and a board held in reset.

**Dropped link.** A USB serial link can disappear for a moment (cable, hub, the host's
power management) while the board keeps running. When that happens the bench reopens the
port and sends the same image again, up to five times, and counts it in `run.n_reconnects`;
if the port does not come back the run ends as described under "A run that ends early". On the
firmware side a raw `#` in the middle of a packet restarts the packet, so the image sent
again is not appended to what arrived before the drop. The latency is measured on the
chip, so a repeated exchange is a valid measurement like any other.

**A run that ends early.** Each image is written to the two CSV files as soon as it is measured. A
board that stops answering makes every image wait 30 s, so after 5 images in a row without an
answer (`--max_unanswered`; 0 never gives up) the run is given up. In that case, and when the link
is lost for good, the operator presses Ctrl+C or an image cannot be read, the bench still writes
`metrics_*.json`, marked `run.completed = false` with the reason in `run.stopped_because` and the
number of images planned in `dataset.n_images_planned`, and ends with exit code 3. Its figures
cover only the images recorded.

**What the host sends depends on its decoder.** The bench decodes each image file on the host and sends the
pixels. With JPEG files the pixels are not fixed by the file alone: the same Portenta H7, model and JPEG test
images gave 3 different predictions out of 940 between the run published in September and a run in November,
and both boards agreed with each other on all 940 in November. A partition stored as lossless PNG
(`reshape_images.py --format png`) removes that dependence: every decoder returns the same pixels.

**Input handling.** Received bytes (uint8, 0–255) are written into the model
input tensor. For an INT8 model the firmware reads the tensor's own
`params.scale` and `params.zero_point` — set by the TFLite converter — and
applies `q = round(pixel / scale) + zero_point`, saturated to `[-128, 127]`.
Nothing is hardcoded, so the same firmware is correct for any quantized model;
`scale ≈ 1.0, zero_point = -128` reduces to the familiar `pixel - 128`. For a
FP32 model the firmware maps each pixel as `float = pixel / 255.0`; that branch is
not validated (see [Known Limitations](../README.md#known-limitations)). If fewer bytes than
the tensor expects arrive, the remainder is padded with `zero_point` (the
quantized representation of a real 0), not with a raw 0. The bench does not let that happen:
it refuses to send images whose shape is not the one the firmware announces.

> The value fed in is the **raw** pixel, because the shipped models are
> converted without a `Rescaling` layer. This must match the preprocessing of
> the representative dataset used at conversion time — if you retrain with
> `Rescaling(1./255)` or MobileNet's `preprocess_input`, adjust `real_value` in
> `loadImageToInputTensor()` accordingly.

**Response.** After `Invoke()`, the board prints the predicted class index as a
single integer line (argmax of the output tensor). Diagnostic banners and the
on-chip CPU temperature (`°C`, from STM32H7 factory calibration) are also printed
around each inference; `pil_benchmark.py` parses the integer class from the stream.
Reporting the class index rather than the full per-class logit vector keeps the
post-processing short and compatible with arbitrary class counts: its median is 0.5 to 0.7 µs in each of the 16 PIL and HIL runs measured before the COCO benchmark (the Visual Wake Words run and the vehicle application case), and 117 of their 7,000 measurements take 1 µs or more (7.5 µs at most).

**Board announcement.** Each sketch prints `BOARD:<name>` (`PORTENTA_H7` or
`NICLA_VISION`): at boot in both, and in PIL also among the identity lines and with every
image, which is where a bench reads it when it opens the port after the boot lines are
gone. The benches record the name in `run.board` and stop if it is not the board given
with `--board`. A firmware that prints no such line is recorded as `null`.

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
