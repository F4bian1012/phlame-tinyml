# Benchmark `COCO`: MIL, SIL and PIL

Built by `src/build_benchmark_table.py --tag COCO` from the result files. Accuracy / macro-F1; `=` columns count the test images with the same prediction at both levels.

| Architecture | `.tflite` (INT8) | n | MIL | SIL | MIL = SIL | PIL Portenta H7 | SIL = PIL | `INF` (ms) | `TOTAL` (ms) | PIL Nicla Vision | SIL = PIL | `INF` (ms) | `TOTAL` (ms) | Portenta H7 = Nicla Vision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `MobileNet` | 310.8 KiB | 5000 | 0.8478 / 0.8462 | 0.8438 / 0.8431 | 4800 / 5000 | 0.8428 / 0.8421 | 4975 / 5000 | 241.40 | 244.64 | 0.8428 / 0.8421 | 4975 / 5000 | 193.45 | 196.57 | 5000 / 5000 |
| `MobileNetV2` | 629.5 KiB | 5000 | 0.8592 / 0.8575 | 0.8590 / 0.8573 | 4849 / 5000 | 0.8592 / 0.8575 | 4999 / 5000 | 401.15 | 404.40 | 0.8592 / 0.8575 | 4999 / 5000 | 322.06 | 325.18 | 5000 / 5000 |
| `SqueezeNet` | 809.3 KiB | 5000 | 0.8374 / 0.8359 | 0.8356 / 0.8340 | 4959 / 5000 | 0.8360 / 0.8344 | 4996 / 5000 | 1153.63 | 1156.88 | 0.8360 / 0.8344 | 4996 / 5000 | 987.00 | 990.12 | 5000 / 5000 |
| `MobileNetV3Small` | 816.2 KiB | 5000 | 0.8388 / 0.8366 | 0.7856 / 0.7842 | 4238 / 5000 | 0.7800 / 0.7788 | 4754 / 5000 | 489.59 | 492.84 | 0.7800 / 0.7788 | 4754 / 5000 | 414.81 | 417.93 | 5000 / 5000 |
| `ResNet8` | 1249.2 KiB | 5000 | 0.8362 / 0.8340 | 0.8366 / 0.8344 | 4966 / 5000 | 0.8360 / 0.8337 | 4993 / 5000 | 2892.09 | 2895.34 | 0.8360 / 0.8337 | 4993 / 5000 | 2561.31 | 2564.43 | 5000 / 5000 |
| `MobileNetV3Large` | 2179.5 KiB | 5000 | 0.8816 / 0.8806 | 0.8658 / 0.8644 | 4709 / 5000 | — | — | — | — | — | — | — | — | — |
| `ResNet18` | 11057.3 KiB | 5000 | 0.8524 / 0.8511 | 0.8548 / 0.8534 | 4950 / 5000 | — | — | — | — | — | — | — | — | — |

## The same INT8 file with other TFLite kernels on the PC

The SIL column above uses the kernels named under "SIL of record". Each other kind is compared with it and with the board (Portenta H7 / Nicla Vision).

| Architecture | SIL of record | SIL `default` | = SIL of record | = PIL | SIL `builtin` | = SIL of record | = PIL |
|---|---|---:|---:|---:|---:|---:|---:|
| `MobileNet` | `reference` | 0.8448 / 0.8440 | 4929 / 5000 | 4920 / 5000 · 4920 / 5000 | 0.8432 / 0.8424 | 4971 / 5000 | 4960 / 5000 · 4960 / 5000 |
| `MobileNetV2` | `reference` | 0.8568 / 0.8550 | 4927 / 5000 | 4926 / 5000 · 4926 / 5000 | 0.8578 / 0.8561 | 4948 / 5000 | 4947 / 5000 · 4947 / 5000 |
| `SqueezeNet` | `reference` | 0.8350 / 0.8334 | 4959 / 5000 | 4961 / 5000 · 4961 / 5000 | 0.8364 / 0.8347 | 4984 / 5000 | 4986 / 5000 · 4986 / 5000 |
| `MobileNetV3Small` | `reference` | — | — | — | 0.7828 / 0.7817 | 4676 / 5000 | 4670 / 5000 · 4670 / 5000 |
| `ResNet8` | `reference` | 0.8354 / 0.8332 | 4976 / 5000 | 4973 / 5000 · 4973 / 5000 | 0.8374 / 0.8352 | 4994 / 5000 | 4991 / 5000 · 4991 / 5000 |
| `MobileNetV3Large` | `reference` | 0.8662 / 0.8647 | 4828 / 5000 | — · — | 0.8618 / 0.8602 | 4822 / 5000 | — · — |
| `ResNet18` | `reference` | 0.8532 / 0.8518 | 4956 / 5000 | — · — | 0.8550 / 0.8537 | 4987 / 5000 | — · — |
