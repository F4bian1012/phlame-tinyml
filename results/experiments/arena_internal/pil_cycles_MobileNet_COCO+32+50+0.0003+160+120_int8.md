# PIL latency of `MobileNet_COCO+32+50+0.0003+160+120_int8`, run by run

Built by `src/compare_pil_cycles.py` from the per-image latency files of 4 PIL runs of the same model file. Every figure is a mean over the 200 images that all the runs measured.

## Core cycles per phase

| Run | Board | Arena memory | Images of the run | Images in common | `PRE` | `INF` | `POST` | `TOTAL` |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `results/pil` | Portenta H7 | SDRAM (board default) | 5000 | 200 | 1,298,594 | 96,558,554 | 213 | 97,857,360 |
| `results/nicla/pil` | Nicla Vision | INTERNAL (board default) | 5000 | 200 | 1,247,415 | 77,384,145 | 115 | 78,631,675 |
| `results/experiments/arena_internal` | Portenta H7 | INTERNAL | 200 | 200 | 1,251,099 | 77,538,767 | 125 | 78,789,991 |
| `results/experiments/arena_sdram` | Portenta H7 | SDRAM | 200 | 200 | 1,298,051 | 94,917,048 | 212 | 96,215,310 |

"(board default)": the files of the run do not record where the arena was. It was measured with a firmware that did not announce it, in which the arena is where the default build of the board has it.

## Microseconds per phase

| Run | `PRE` | `INF` | `POST` | `TOTAL` |
|---|---:|---:|---:|---:|
| `results/pil` | 3,246.485 | 241,396.385 | 0.532 | 244,643.401 |
| `results/nicla/pil` | 3,118.537 | 193,460.362 | 0.288 | 196,579.187 |
| `results/experiments/arena_internal` | 3,127.748 | 193,846.917 | 0.312 | 196,974.977 |
| `results/experiments/arena_sdram` | 3,245.127 | 237,292.621 | 0.529 | 240,538.276 |

## Difference to the first run

Cycles of each run relative to those of `results/pil`, in per cent; negative is fewer cycles.

| Run | `PRE` | `INF` | `POST` | `TOTAL` |
|---|---:|---:|---:|---:|
| `results/nicla/pil` | -3.94% | -19.86% | -45.89% | -19.65% |
| `results/experiments/arena_internal` | -3.66% | -19.70% | -41.32% | -19.48% |
| `results/experiments/arena_sdram` | -0.04% | -1.70% | -0.65% | -1.68% |
