# Related Work: PHLAME Next to Other TinyML Benchmarks and Tools

> **Status of the sources.** The sources cited on this page were read on 15 November 2026 in their HTML versions (arXiv HTML, web documentation, repositories), through a tool that summarises each page, and the citations still have to be checked against the PDFs; where a source was read only in part (an abstract, a README), the text says so.

This page covers the works listed under [References](#references); it is not a systematic survey.

## What the Other Benchmarks and Tools Measure

**MLPerf Tiny** [1] is the reference benchmark for inference on microcontrollers: it compares hardware and software stacks under fixed run rules on four tasks and, since v1.3, a streaming wake-word test [2]. Its harness [3, 5] sends pre-recorded inputs from a host to the device, with no physical sensor in the loop, and reports accuracy, throughput and, optionally, energy per inference. The accuracy log of the one public submission we examined holds the output of the device for every input file [4], but the benchmark reports no comparison with the same model on the host.

**Benchmarking tools** compare frameworks, compilers and boards on latency and memory. MLonMCU [6] does it for deployment flows, on a simulator and on boards. EdgeMark [7] also measures the mean difference between the output on the device and on the PC, over ten inputs. MLino Bench [8] reports on-target accuracy from test vectors sent over serial; we read its README and abstract only and do not know whether it stores each prediction.

**Energy.** CREST [10] measures latency and energy on boards, the Portenta H7 among them, while it selects models, and computes task quality on the host. Bartoli et al. [11] measure energy and latency per phase on the MLPerf Tiny models; accuracy is part of their method but is not reported.

**Vendor tools** check the deployed model against the original one. ST Edge AI Core [12–14] runs the generated C model on the host and on the target, saves every input and output and profiles each layer. Edge Impulse [15–17] tests the float32 or int8 variant sample by sample in its Studio and runs the deployed model on live sensor data, without labelled stimuli. MATLAB runs generated deep-learning code on Cortex-M boards in processor-in-the-loop mode [18]: the terms MIL, SIL and PIL come from model-based design, and PIL for neural networks did not start with PHLAME.

The table adds the functional simulators [19–22] and PICO [9], which targets Linux boards, not microcontrollers. U-TOE [27] and Kenning [28] were read only from an abstract and from documentation, and are not compared.

## Comparison Table

Levels: **M**, float model on the host; **S**, quantized model on the host; **P**, quantized model on the chip, input injected from the host; **H**, physical sensor of the device in the loop. "Not checked" and "not found" describe what we read, not the system.

| System | Purpose | Levels | Per-sample outputs | Latency | Energy | Hardware | Openness and version checked |
|---|---|---|---|---|---|---|---|
| **PHLAME** | Fidelity ladder for INT8 image classifiers (binary classification, COCO person / non-person) | M, S, P, H (those measured on the public benchmark: [README](../README.md#results)) | Every level, every image evaluated, in the repository; paired tests (exact McNemar, Holm, paired bootstrap) | On the chip, per phase (DWT cycle counter) | No | Portenta H7, Nicla Vision (both STM32H747, Cortex-M7) | MIT; no release yet |
| MLPerf Tiny [1–5] | Standard benchmark of hardware and software inference stacks | P (UART injection); streaming test: stored audio over I2S | Device output per input in the runner log of the one v1.1 submission examined; no host counterpart | Median throughput; timer of at least 1 kHz, or GPIO edge | Yes, external monitor | Submitters' devices; reference NUCLEO-L4R5ZI | Apache-2.0 repository; EnergyRunner binary by click-through; datasets under an agreement with MLCommons; v1.3 results (Sep 2025) |
| MLonMCU [6] | Benchmarking of deployment flows on the MLPerf Tiny models | P and instruction-set simulation; check against golden outputs described, no results reported | Not reported | Instruction counts (simulator); run time (boards) | No | ETISS simulator (RISC-V), ESP32, ESP32-C3, STM32F4, STM32F7 | Apache-2.0; tag v0.10.0 (Oct 2026) |
| EdgeMark [7] | Automation and comparison of embedded AI tools | Host reference + P | Mean device-versus-PC output error over 10 inputs | Mean of 10 runs | No | NUCLEO-L4R5ZI, Renesas RX65N | Open source (licence not stated in the text read); J. Syst. Archit. 2025 |
| MLino Bench [8] | On-target benchmarking of classical ML models and neural networks | P (test vectors over serial) | On-target accuracy; per-sample output not documented in what we read | Classification time | No | Arduino Mega, Arduino Due, Teensy 4.0 | Apache-2.0; J. Syst. Archit. 2024 (README and abstract only) |
| CREST [10] | Model selection with the board in the loop | S (quality on the host) + P (cost, on recorded inputs) | Not reported | Yes | Yes (INA228, GPIO-marked windows) | Nano 33 BLE Sense, Portenta H7, NUCLEO-N657X0-Q | Open source (licence not checked); arXiv v2 (Nov 2026) |
| PICO [9] | Latency, CPU and memory of TFLite models | On the device only (embedded Linux, not a microcontroller) | Confidence scores; no accuracy | 100 iterations | No | BeagleBone AI-64, Raspberry Pi 4 | No code link found; arXiv v1 (Sep 2025) |
| Bartoli et al. [11] | Method for per-phase energy and latency | P (input loaded from external flash) | Accuracy planned, not reported | Per phase (GPIO + oscilloscope) | Per phase (shunt + oscilloscope) | STM32N6570-DK (Cortex-M55 + NPU) | No code link found; IJCNN 2025 |
| ST Edge AI Core `validate` [12–14] | Check of the generated C model against the source model | Source model on the host + C model on the host + P (serial) | All inputs and outputs saved (npz) | Cycles and ms, per layer | Not in `validate`; other ST tools not checked | STM32 and other ST targets | Proprietary (licence not read); v4.1.0 |
| Edge Impulse [15–17] | End-to-end TinyML platform | M, S per sample in Studio; run on the device with the live sensor, no labelled stimuli | In Studio; printed per window on the device | Estimated (Renode and device benchmarks); signal-processing and classification time measured on the device | Not checked | Many supported boards | Commercial service; MLSys 2023 paper |
| MATLAB PIL [18] | Model-based design | M (MATLAB) + P, with an output tolerance check | Programmatic | Execution profiling | — | Cortex-M boards (example: STM32F769I-Discovery) | Commercial; R2026b documentation |
| Functional simulators (Renode, Arm FVPs) [19–22] | Running firmware without hardware | Firmware on a simulated microcontroller; sensors can be fed recorded data | — | Not cycle-accurate | — | Simulated Cortex-M, RISC-V | Renode open; FVP licence not checked |

## What PHLAME Does That We Did Not Find in Them

1. **The same test images at every level, every prediction published, paired statistics.** MIL, SIL and PIL are evaluated on the complete test set and the camera level (HIL) on a fixed subset of it. Every level stores the prediction of each image, in the repository, and the levels are compared image by image: agreement counts and, with the four levels measured, exact McNemar tests with Holm correction and paired bootstrap intervals. Among the systems reviewed we did not find one that publishes host and on-chip predictions for a complete test set and compares them with paired statistics: the MLPerf Tiny log we examined holds the outputs of the device only, ST saves both but expects a small subset and does no statistical inference, and EdgeMark uses ten inputs.
2. **The camera in front of labelled stimuli.** At the HIL level the camera of the board looks at a monitor that shows each test image, so every captured frame has a label, and classifying the same frames on the PC separates the effect of the camera from that of the chip. We did not find a benchmark that puts a physical sensor in front of labelled stimuli; the search was not exhaustive (no access to Scopus or IEEE Xplore), and its closest precedent photographs printed images with a phone [26] (abstract only).
3. **The measured difference between the PC interpreter and the chip.** On the 5,000 COCO test images, for four of the five models that fit the board, the default TFLite interpreter (XNNPACK delegate) gives another prediction than the chip for 27 to 80 images, and the reference kernels of TFLite for 1 to 25 (for the fifth model see [`coco_benchmark.md`](coco_benchmark.md#notes-on-the-results)). That integer kernels on a PC and on a microcontroller can disagree is known to TFLM developers [23–25]; in what we reviewed we did not find it measured at the level of predictions, on a public benchmark and a real chip. The chip runs CMSIS-NN kernels for most operators of these models ([`hardware.md`](hardware.md#boards-and-memory)), so the figures check on hardware the bit-exactness that the CMSIS-NN README states [23].

## What PHLAME Does Not Do

- **No energy.** Neither energy nor power is measured. MLPerf Tiny, CREST and Bartoli et al. measure it.
- **One microcontroller, one runtime.** Both boards carry the same STM32H747 and run TFLM through the same library, so their identical predictions are a consistency check, not a portability result.
- **One kind of task.** Binary classification of 160×120 grayscale images. Its COCO figures are not comparable with those of the visual wake words task of MLPerf Tiny: 96×96 colour input, another reference model, other rules.
- **No standardised submission process.** No run rules, divisions, review of submitted results or community of submitters, and one maintainer. PHLAME is not a ranking of hardware.
- **No release yet.** No tagged release and no DOI.
- **Less than the vendor tools in two respects.** The firmware returns only the predicted class, so metrics over the output vector, which ST and MLPerf Tiny compute, cannot be obtained from a board run. And its SIL runs TFLite on the PC, not the kernels of the firmware, which the generated C model of ST and the TFLM Python interpreter (a prototype, by its own documentation) [29] run on the host.
- **Per-phase latency is not specific to it.** Bartoli et al., ST and Edge Impulse also time phases, layers or stages.
- **Also:** post-training quantization only, published PIL files that predate the on-chip check of the model, and PIL runs of 65 minutes to almost 5 hours for 5,000 images at 115,200 baud.

The levels measured on the public benchmark are stated in the README ([Results](../README.md#results), section 1), and the limitations of the framework under [Known Limitations](../README.md#known-limitations).

## References

Identifiers as recorded when the sources were read (15 November 2026). Three details to settle when the citations are checked: the arXiv comment of [6] names the 2022 edition of the workshop and the ACM publication is dated 2023; [10] changed its title between its two arXiv versions; and the arXiv version of [15] lists the authors in another order than the proceedings. The exact name of the proceedings of [1] was not confirmed on the proceedings page.

1. C. Banbury, V. Janapa Reddi, P. Torelli, et al. "MLPerf Tiny Benchmark". NeurIPS 2021 Datasets and Benchmarks Track (Round 1). arXiv:2106.07597. <https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/da4fb5c6e93e74d3df8527599fa62642-Abstract-round1.html>
2. MLCommons. "MLCommons New MLPerf Tiny v1.3 Benchmark Results Released", 17 September 2025. <https://mlcommons.org/2025/09/mlperf-tiny-v1-3-results/>
3. MLCommons. `mlcommons/tiny`, `benchmark/runner` (README). Apache-2.0. <https://github.com/mlcommons/tiny/tree/master/benchmark/runner>
4. MLCommons. `tiny_results_v1.1`, `closed/STMicroelectronics/results/NUCLEO_H7A3ZI_Q/vww/accuracy` (`log.txt`, `results.txt`). <https://github.com/mlcommons/tiny_results_v1.1>
5. EEMBC. "EnergyRunner" (user guide; version 3.0.10, 2021-11-15). <https://github.com/eembc/energyrunner>
6. P. van Kempen, R. Stahl, D. Mueller-Gritschneder, U. Schlichtmann. "MLonMCU: TinyML Benchmarking with Fast Retargeting". Proc. 2023 Workshop on Compilers, Deployment, and Tooling for Edge AI (CODAI), ACM, pp. 32–36. DOI 10.1145/3615338.3618128. arXiv:2306.08951. Code: <https://github.com/tum-ei-eda/mlonmcu>
7. M. A. Hasanpour, M. Kirkegaard, X. Fafoutis. "EdgeMark: An automation and benchmarking system for embedded artificial intelligence tools". Journal of Systems Architecture 167:103488, 2025. DOI 10.1016/j.sysarc.2025.103488. arXiv:2502.01700. Code: <https://github.com/Black3rror/EdgeMark>
8. V.-E. Baciu, J. Stiens, B. da Silva. "MLino bench: A comprehensive benchmarking tool for evaluating ML models on edge devices". Journal of Systems Architecture 155:103262, 2024. DOI 10.1016/j.sysarc.2024.103262. Code: <https://github.com/vladBaciu/MLino-Bench>
9. A. Dey, S. Srivastava, G. Singh, R. G. Pettit. "Real-Time Performance Benchmarking of TinyML Models in Embedded Systems (PICO: Performance of Inference, CPU, and Operations)". arXiv:2509.04721, 2025.
10. J. Q. Zales, P. Sharma, M. Srivastava. "Measure the Deployment, Not Just the Model: Rethinking Hardware-Aware Model Selection for Microcontrollers" (version 2, 9 November 2026; version 1, 12 June 2026, titled "CREST: Deployment-Realistic Hardware-in-the-Loop NAS for Embedded Sensing Systems"). arXiv:2606.15004. Code: <https://github.com/nesl/crest>
11. P. Bartoli, C. Veronesi, A. Giudici, D. Siorpaes, D. Trojaniello, F. Zappa. "Benchmarking Energy and Latency in TinyML: A Novel Method for Resource-Constrained AI". 2025 International Joint Conference on Neural Networks (IJCNN), IEEE, pp. 1–8. DOI 10.1109/IJCNN64981.2025.11228997. arXiv:2505.15622.
12. STMicroelectronics. "Evaluation report and metrics", ST Edge AI Core Technology 4.1.0 (r2.8). <https://stm32ai-cs.st.com/assets/embedded-docs/evaluation_metrics.html>
13. STMicroelectronics. "How to use the AiRunner package", ST Edge AI Core Technology 4.1.0 (r1.0). <https://stm32ai-cs.st.com/assets/embedded-docs/ai_runner_python_module.html>
14. STMicroelectronics. "AI:Validation and Performance Measurement" (wiki, version approved on 9 February 2026). <https://wiki.st.com/stm32mcu/wiki/AI:Validation_and_Performance_Measurement>
15. C. Banbury, V. Janapa Reddi, A. Elium, et al. "Edge Impulse: An MLOps Platform for Tiny Machine Learning". Proceedings of Machine Learning and Systems 5 (MLSys 2023). <https://proceedings.mlsys.org/paper_files/paper/2023/hash/49fe55f5e9574714dda575bfb2177662-Abstract-mlsys2023.html>. arXiv:2212.03332.
16. Edge Impulse Docs. "Model testing". <https://docs.edgeimpulse.com/docs/edge-impulse-studio/model-testing>
17. Edge Impulse Docs. "Edge Impulse CLI: run impulse". <https://docs.edgeimpulse.com/docs/tools/edge-impulse-cli/cli-run-impulse>
18. MathWorks. "Code Generation and Deployment for Acoustic-Based Machine Fault Detection Using Deep Learning on ARM Cortex-M Hardware" (documentation R2026b). <https://www.mathworks.com/help/ecoder/armcortexm/ref/code-generation-and-deployment-for-acoustic-based-machine-fault-detection-using-deep-learning-on-armcortex-m-hardware.html>
19. Arm. "Arm FVP models: Overview", Arm Virtual Hardware 2.2.0. <https://arm-software.github.io/AVH/main/simulation/html/index.html>
20. Arm. Fast Models Reference Guide, version 11.22 (superseded), "Model capabilities". <https://developer.arm.com/documentation/100964/1122/About-the-models/Model-capabilities>
21. Antmicro. "RISC-V co-design using trace-based simulation with Renode and TBM", 18 July 2023. <https://antmicro.com/blog/2023/07/risc-v-co-design-using-trace-based-simulation-with-renode-and-tbm/>
22. M. Gielda. "Running and Testing TF Lite on Microcontrollers without hardware in Renode", TensorFlow Blog, 9 June 2020. <https://blog.tensorflow.org/2020/06/running-and-testing-tf-lite-on-microcontrollers.html>
23. Arm. "CMSIS-NN" (README). Apache-2.0. <https://github.com/ARM-software/CMSIS-NN>
24. TensorFlow Lite Micro, issue #3252. "Standardize TFLM Reference Kernels to Single Rounding Requantization", 14 November 2025. <https://github.com/tensorflow/tflite-micro/issues/3252>
25. SIG Micro (TensorFlow). "TFLM model predictions differ to TFLite model predictions", March 2022. <https://groups.google.com/a/tensorflow.org/g/micro/c/w38Q81rOR-w>
26. A. Kurakin, I. Goodfellow, S. Bengio. "Adversarial examples in the physical world". arXiv:1607.02533, 2017.
27. Z. Huang, K. Zandberg, K. Schleiser, E. Baccelli. "U-TOE: Universal TinyML On-board Evaluation Toolkit for Low-Power IoT". arXiv:2306.14574 (IFIP/IEEE PEMWN 2023).
28. Antmicro. "Kenning" (README). Apache-2.0. <https://github.com/antmicro/kenning>
29. TensorFlow. `tflite-micro/python/tflite_micro` (README). <https://github.com/tensorflow/tflite-micro/tree/main/python/tflite_micro>
