# PIL latency of `MobileNet_COCO+32+50+0.0003+160+120_int8`, run by run

Built by `src/compare_pil_cycles.py` from the per-image latency files of 4 PIL runs of the same model file. Every figure is a mean over the 200 images that all the runs measured.

## Core cycles per phase

| Run | Board | Arena memory | Images of the run | Images in common | `PRE` | `INF` | `POST` | `TOTAL` |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `results/pil` | Portenta H7 | SDRAM | 5000 | 200 | 1,298,128 | 94,912,686 | 208 | 96,211,023 |
| `results/nicla/pil` | Nicla Vision | INTERNAL | 5000 | 200 | 1,250,124 | 79,338,465 | 131 | 80,588,720 |
| `results/experiments/arena_internal` | Portenta H7 | INTERNAL | 200 | 200 | 1,251,099 | 77,538,767 | 125 | 78,789,991 |
| `results/experiments/arena_sdram` | Portenta H7 | SDRAM | 200 | 200 | 1,298,051 | 94,917,048 | 212 | 96,215,310 |

## Microseconds per phase

| Run | `PRE` | `INF` | `POST` | `TOTAL` |
|---|---:|---:|---:|---:|
| `results/pil` | 3,245.321 | 237,281.716 | 0.521 | 240,527.557 |
| `results/nicla/pil` | 3,125.311 | 198,346.163 | 0.328 | 201,471.801 |
| `results/experiments/arena_internal` | 3,127.748 | 193,846.917 | 0.312 | 196,974.977 |
| `results/experiments/arena_sdram` | 3,245.127 | 237,292.621 | 0.529 | 240,538.276 |

## Difference to the first run

Cycles of each run relative to those of `results/pil`, in per cent; negative is fewer cycles.

| Run | `PRE` | `INF` | `POST` | `TOTAL` |
|---|---:|---:|---:|---:|
| `results/nicla/pil` | -3.70% | -16.41% | -36.92% | -16.24% |
| `results/experiments/arena_internal` | -3.62% | -18.31% | -40.00% | -18.11% |
| `results/experiments/arena_sdram` | -0.01% | +0.00% | +1.58% | +0.00% |
