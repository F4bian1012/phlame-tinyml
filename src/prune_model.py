"""Optional step: unstructured magnitude pruning of a trained .keras model.

Zeroes the lowest-magnitude kernel weights of the Conv2D and Dense layers of the
model, those inside a nested model included (the MobileNets keep their backbone
as one nested model). Depthwise kernels, batch-normalisation parameters and
biases are left as they are. The tensor shapes do not change, so the .tflite of
a pruned model is not smaller. No published result uses this script.

Output: {model}_pruned_{technique}_{sparsity}.keras next to the input.
"""
import os
import sys

import artifacts
import cli


def parse_args(argv=None):
    parser = cli.ArgumentParser(description="Manual Model Pruning for TensorFlow/Keras Models")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the original model (e.g., models/checkpoints/model.keras)")
    parser.add_argument("--sparsity", type=float, required=True,
                        help="Target fraction of kernel weights to prune in [0.0, 1.0) (e.g., 0.2 for 20%%)")
    parser.add_argument("--technique", type=str, choices=["global", "layer"], required=True,
                        help="Type of manual unstructured pruning technique to apply")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace the pruned model if it already exists")
    return parser.parse_args(argv)


def output_path_of(model_path, technique, sparsity):
    base, ext = os.path.splitext(model_path)
    return f"{base}_pruned_{technique}_{sparsity}{ext}"


def prunable_layers(model):
    """The Conv2D and Dense layers of the model, in order, those of nested models included."""
    import tensorflow as tf
    for layer in model.layers:
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)):
            yield layer
        elif hasattr(layer, "layers"):
            yield from prunable_layers(layer)


def kernels(model):
    """(layer, kernel, rest of its weights) for every prunable layer that has weights."""
    for layer in prunable_layers(model):
        weights = layer.get_weights()
        if weights:
            yield layer, weights[0], weights[1:]


def calculate_sparsity(model):
    """Fraction [0.0, 1.0] of zero kernel weights across the Conv2D and Dense layers of the model."""
    import numpy as np
    total_params = 0
    zero_params = 0
    for _, w, _ in kernels(model):
        total_params += w.size
        zero_params += int(np.sum(w == 0))
    if total_params == 0:
        return 0.0
    return zero_params / total_params


def apply_global_unstructured_pruning(model, sparsity):
    """
    Applies global unstructured magnitude-based pruning.
    Finds one magnitude threshold across the kernel weights of every Conv2D and
    Dense layer (leaving biases intact) and zeroes the weights below it.
    """
    import numpy as np
    print(f"Applying Global Unstructured Pruning (target sparsity: {sparsity*100:.1f}%)")

    found = list(kernels(model))
    if not found:
        print("No Conv2D or Dense layers found to prune.")
        return model

    abs_weights = np.abs(np.concatenate([w.flatten() for _, w, _ in found]))
    threshold = np.percentile(abs_weights, sparsity * 100)
    print(f"Global magnitude threshold calculated: {threshold:.6f}")

    for layer, w, rest in found:
        layer.set_weights([w * (np.abs(w) >= threshold)] + list(rest))  # biases are not pruned
    return model


def apply_layer_unstructured_pruning(model, sparsity):
    """
    Applies layer-wise unstructured magnitude-based pruning.
    For each Conv2D and Dense layer, zeroes the lowest-magnitude fraction
    `sparsity` of its kernel weights (leaving biases intact).
    """
    import numpy as np
    print(f"Applying Layer-wise Unstructured Pruning (target sparsity per layer: {sparsity*100:.1f}%)")

    for layer, w, rest in list(kernels(model)):
        abs_w = np.abs(w)
        threshold = np.percentile(abs_w, sparsity * 100)
        layer.set_weights([w * (abs_w >= threshold)] + list(rest))
    return model


def main(argv=None):
    args = parse_args(argv)

    # A step that reports an error ends with a non-zero exit code, like the other
    # scripts, so a chained pipeline does not carry on with an earlier file.
    if not os.path.exists(args.model_path):
        print(f"Error: Model path {args.model_path} does not exist.")
        sys.exit(1)

    if not (0.0 <= args.sparsity < 1.0):
        print("Error: Sparsity must be in the range [0.0, 1.0) (e.g., 0.2 for 20%).")
        sys.exit(1)

    output_path = output_path_of(args.model_path, args.technique, args.sparsity)
    artifacts.refuse_to_overwrite([output_path], args.overwrite)

    # TensorFlow is loaded only now, so --help, a wrong option and the three checks
    # above answer at once.
    import tensorflow as tf

    print(f"Loading model from {args.model_path}...")
    try:
        model = tf.keras.models.load_model(args.model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    initial_sparsity = calculate_sparsity(model)
    print(f"Initial model sparsity: {initial_sparsity*100:.2f}%")

    if args.technique == "global":
        model = apply_global_unstructured_pruning(model, args.sparsity)
    elif args.technique == "layer":
        model = apply_layer_unstructured_pruning(model, args.sparsity)

    final_sparsity = calculate_sparsity(model)
    print(f"Final model sparsity achieved: {final_sparsity*100:.2f}%")

    print(f"Saving pruned model to {output_path}...")
    try:
        model.save(output_path)
        print("Model saved successfully.")
    except Exception as e:
        print(f"Error saving pruned model: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
