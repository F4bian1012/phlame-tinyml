import os
import sys

import artifacts
import cli

# --kernels: which implementation of the operators runs the model on the PC.
KERNELS = {
    "default": "AUTO",
    "builtin": "BUILTIN_WITHOUT_DEFAULT_DELEGATES",
    "reference": "BUILTIN_REF",
}


def parse_args():
    parser = cli.ArgumentParser(description="SIL level: evaluate an INT8 .tflite model on the test split, with the same metrics as mil_benchmark.py")
    parser.add_argument('--width', type=int, default=160, help="Image width")
    parser.add_argument('--height', type=int, default=120, help="Image height")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the .tflite model to evaluate (required)")
    parser.add_argument('--splits_dir', type=str, default="data/splits",
                        help="Directory holding the splits produced by split_dataset.py "
                             "(evaluation uses ONLY <splits_dir>/test)")
    parser.add_argument('--data_dir', type=str, default=None, help="Explicit path to the test split; defaults to <splits_dir>/test")
    parser.add_argument('--class_names_path', type=str, default=None, help="Path to the txt file with the class names (default: the .classes.txt next to the model, else models/class_names.txt)")
    parser.add_argument('--output_dir', type=str, default=os.path.join("results", "sil"), help="Output directory for the matrix, JSON and CSV (default: results/sil)")
    parser.add_argument('--overwrite', action='store_true', help="Replace the matrix, JSON and CSV if they already exist")
    parser.add_argument('--kernels', choices=sorted(KERNELS), default="default",
                        help="TFLite kernels that run the model on the PC. default: what the interpreter picks, "
                             "which includes the XNNPACK delegate. builtin: the built-in optimized kernels without "
                             "that delegate. reference: the reference kernels, the ones TFLite-Micro is derived "
                             "from. The choice is recorded in metrics_*.json (model.kernels)")

    args = parser.parse_args()

    # SIL evaluates EXCLUSIVELY the test split, the exact same split consumed
    # by MIL, PIL, and HIL, so that jumps across ladder levels are attributable
    # to the deployment level rather than to a different image sample. Never
    # train (which calibrated INT8 quantization in quantize_int8_basic.py) and
    # never val (already consumed by early stopping and model selection).
    if args.data_dir is None:
        args.data_dir = os.path.join(args.splits_dir, "test")

    return args


if __name__ == "__main__":
    # The command line is read before TensorFlow is imported, which takes
    # several seconds: --help and a wrong option answer at once.
    ARGS = parse_args()

import tensorflow as tf  # noqa: E402
import numpy as np  # noqa: E402

import evaluation  # noqa: E402  (the part shared with mil_benchmark.py)
import images  # noqa: E402


def quantize_input(data, input_details):
    """
    If the TFLite model expects integer inputs instead of float32 (e.g. a
    full-INT8 model), scale and offset the input data manually according to
    its quantization parameters before invoking inference.
    """
    if input_details['dtype'] == np.int8:
        scale, zero_point = input_details['quantization']

        # TFLite returns iterable lists or numpy arrays
        s = scale[0] if isinstance(scale, (list, np.ndarray)) and len(scale) > 0 else scale
        z = zero_point[0] if isinstance(zero_point, (list, np.ndarray)) and len(zero_point) > 0 else zero_point

        if s > 0:
            # Map floating-point values into the quantized INT8 domain
            q_data = (data / s) + z
            q_data = np.clip(q_data, -128, 127).astype(np.int8)
            return q_data
        return data.astype(np.int8)
    # Return as float32 for non-INT8 input tensors
    return data.astype(np.float32)


def make_interpreter(model_path, kernels="default"):
    """TFLite interpreter for model_path with the kernels asked for, tensors allocated.

    The default interpreter hands most of the graph to the XNNPACK delegate, which
    refuses some valid INT8 files (seen with a tensor that is constant over the
    calibration images and so gets a degenerate scale). The run stops with what
    to try instead of a traceback: the other kernels accept such a file.
    """
    resolver = getattr(tf.lite.experimental.OpResolverType, KERNELS[kernels])
    interpreter = tf.lite.Interpreter(model_path=model_path, experimental_op_resolver_type=resolver)
    try:
        interpreter.allocate_tensors()
    except RuntimeError as exc:
        print(f"ERROR: the TFLite interpreter could not prepare {model_path} with --kernels {kernels}:")
        print(f"  {str(exc).strip()}")
        others = [name for name in sorted(KERNELS) if name != kernels]
        print(f"Try --kernels {others[0]} (or {others[1]}); the kernels used are recorded in the metrics file.")
        sys.exit(1)
    return interpreter


def dequantize_output(data, output_details):
    """
    If the output tensor is quantized in INT8, restore it to floating-point
    scores by subtracting zero_point and multiplying by scale.
    """
    if output_details['dtype'] == np.int8:
        scale, zero_point = output_details['quantization']

        s = scale[0] if isinstance(scale, (list, np.ndarray)) and len(scale) > 0 else scale
        z = zero_point[0] if isinstance(zero_point, (list, np.ndarray)) and len(zero_point) > 0 else zero_point

        if s > 0:
            dq_data = (data.astype(np.float32) - z) * s
            return dq_data
    return data.astype(np.float32)


def main(args=None):
    if args is None:
        args = parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: You must provide a valid TFLite model. Not found: {args.model_path}")
        sys.exit(1)

    evaluation.refuse_to_replace_results(args.output_dir, args.model_path, args.overwrite)

    class_names, class_names_path = evaluation.read_class_names(args.model_path, args.class_names_path)

    print(f"\nInitializing TFLite Interpreter with {args.model_path}...")
    interpreter = make_interpreter(args.model_path, args.kernels)
    print(f"TFLite kernels: {args.kernels}")

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    _, expected_height, expected_width, expected_channels = input_details['shape']

    # Strip surrounding whitespace or stray quotes from the path string
    if args.data_dir:
        args.data_dir = args.data_dir.strip().strip("'").strip('"')

    # If the path encodes dimensions in its name (legacy paths such as
    # data/processed/160x120), update them automatically to match the model
    if f"{args.width}x{args.height}" in args.data_dir and (args.width != expected_width or args.height != expected_height):
        args.data_dir = args.data_dir.replace(f"{args.width}x{args.height}", f"{expected_width}x{expected_height}")
        print(f"[INFO] Updating path resolution to {expected_width}x{expected_height} (required by the model).")

    # Input geometry is always dictated by the model's input tensor: the split
    # path no longer encodes resolution, so updating the path alone is not enough.
    if args.width != expected_width or args.height != expected_height:
        print(f"[INFO] Resizing inputs to {expected_width}x{expected_height} (required by the model).")
        args.width = expected_width
        args.height = expected_height

    # Fall back to <cwd>/<splits_dir>/test if the given path is missing
    if not os.path.exists(args.data_dir):
        fallback_path = os.path.join(os.getcwd(), args.splits_dir, "test")
        if os.path.exists(fallback_path):
            args.data_dir = fallback_path

    if not os.path.exists(args.data_dir):
        evaluation.stop_without_test_split(args.data_dir, expected_width, expected_height, args.splits_dir)

    print(f"Loading the test dataset from {args.data_dir}...")
    class_names = evaluation.class_names_of_this_split(class_names, class_names_path, args.data_dir)
    # Read with the loader shared by MIL and the PIL bench (images.py), so the
    # three levels receive the same bytes for the same file. The order is fixed
    # (classes, then file names), which keeps inputs aligned with file names and labels.
    inferred = not class_names
    test_images, y_true, rel_paths, class_names = images.load_folder(
        args.data_dir, args.width, args.height, class_names if class_names else None,
        grayscale=(expected_channels == 1))
    evaluation.describe_test_split(rel_paths, class_names, inferred)

    predictions = []

    print("Generating frame-by-frame predictions with the TFLite model (sequential inference)...")

    # One image per call, because the TFLite interpreter runs single-frame inference.
    for image in test_images:
        # Raw pixels in [0..255] as float32; converted below as the input tensor requires.
        input_data = images.as_model_input(image[np.newaxis])

        # Reshape if necessary to match the exact expected input tensor shape
        if tuple(input_data.shape) != tuple(input_details['shape']):
            input_data = np.reshape(input_data, input_details['shape'])

        input_data = quantize_input(input_data, input_details)
        interpreter.set_tensor(input_details['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details['index'])
        output_data = dequantize_output(output_data, output_details)
        predictions.append(output_data[0])

    predictions = np.array(predictions)
    y_pred = evaluation.predicted_labels(predictions, len(class_names))

    evaluation.write_level_results(
        level="SIL", model_path=args.model_path, output_dir=args.output_dir,
        y_true=y_true, y_pred=y_pred, rel_paths=rel_paths, class_names=class_names,
        dataset={"splits_dir": args.splits_dir, "data_dir": args.data_dir},
        image_size=(args.width, args.height),
        # What ran the INT8 model and computed the figures (see artifacts.software_versions).
        # The TFLite kernels are chosen with --kernels and recorded in model.kernels.
        software=artifacts.software_versions(["tensorflow", "numpy", "scikit_learn", "pillow"]),
        report_title="              METRICS REPORT (TFLite)",
        plot_title=f'SIL Confusion Matrix - {evaluation.model_name(args.model_path)}',
        model_extra={"kernels": args.kernels})


if __name__ == "__main__":
    main(ARGS)
