# COCO HIL sessions measured before the rig was adjusted

The five HIL sessions of the COCO benchmark measured on 2026-11-15 between 16:00 and 17:15 UTC, on the
Portenta H7, with the rig as it was first set up: 23 cm between camera and screen, the image filling
88 % of the frame. The calibration report taken right before them,
[`../calibration/calibration_20261115T155936Z.json`](../calibration/calibration_20261115T155936Z.json),
did not pass: geometry, gray scale and flat field at WARN, focus at PASS.

The camera was then moved closer until the geometry check passed, and the sessions were measured again.
Those are the sessions of record, in [`results/hil/`](..); the ones here are kept because the same
models, images and conditions with two states of the rig show how much its alignment weighs.

| Model | PIL, same 200 images | HIL, these sessions |
|---|---:|---:|
| `MobileNet` | 0.830 | 0.735 |
| `MobileNetV2` | 0.840 | 0.680 |
| `MobileNetV3Small` | 0.725 | 0.670 |
| `SqueezeNet` | 0.825 | 0.755 |
| `ResNet8` | 0.805 | 0.720 |

Accuracy. The ladder tables built from these sessions are in
[`results/tables/coco_before_rig_adjustment/`](../../tables/coco_before_rig_adjustment/); the paths they
name (`results/hil`) are those the files had when the tables were built.

The frames of these sessions were written to one folder under names that do not carry the model, so
each session overwrote those of the previous one; the predictions of the PC on each frame are in the
`latency_metrics_*.csv` files (`sil_label`, `mil_label`).
