import tensorflow as tf
import numpy as np
import argparse
import os

def calculate_sparsity(model):
    """Calculates the overall sparsity of the model (percentage of zero weights)."""
    total_params = 0
    zero_params = 0
    for layer in model.layers:
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)):
            weights = layer.get_weights()
            if len(weights) > 0:
                w = weights[0] # The weight matrix
                total_params += w.size
                zero_params += np.sum(w == 0)
    
    if total_params == 0:
        return 0.0
    return zero_params / total_params

def apply_global_unstructured_pruning(model, sparsity):
    """
    Applies global unstructured magnitude-based pruning.
    Finds the global threshold across all trainable weights (Conv2D and Dense)
    and sets weights below this threshold to zero.
    """
    print(f"Applying Global Unstructured Pruning (target sparsity: {sparsity*100:.1f}%)")
    
    # 1. Collect all weights
    all_weights = []
    for layer in model.layers:
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)):
            weights = layer.get_weights()
            if len(weights) > 0:
                all_weights.append(weights[0].flatten())
                
    if not all_weights:
        print("No Conv2D or Dense layers found to prune.")
        return model
        
    # 2. Find global threshold
    concat_weights = np.concatenate(all_weights)
    abs_weights = np.abs(concat_weights)
    threshold = np.percentile(abs_weights, sparsity * 100)
    print(f"Global magnitude threshold calculated: {threshold:.6f}")
    
    # 3. Apply threshold mask
    for layer in model.layers:
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)):
            weights = layer.get_weights()
            if len(weights) > 0:
                w = weights[0]
                b = weights[1] if len(weights) > 1 else None
                
                # Create mask and apply to weights
                mask = np.abs(w) >= threshold
                pruned_w = w * mask
                
                new_weights = [pruned_w]
                if b is not None:
                    new_weights.append(b) # We typically don't prune biases
                    
                layer.set_weights(new_weights)
                
    return model

def apply_layer_unstructured_pruning(model, sparsity):
    """
    Applies layer-wise unstructured magnitude-based pruning.
    For each trainable layer, sets the lowest X% of weights to zero.
    """
    print(f"Applying Layer-wise Unstructured Pruning (target sparsity per layer: {sparsity*100:.1f}%)")
    
    for layer in model.layers:
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)):
            weights = layer.get_weights()
            if len(weights) > 0:
                w = weights[0]
                b = weights[1] if len(weights) > 1 else None
                
                abs_w = np.abs(w)
                threshold = np.percentile(abs_w, sparsity * 100)
                
                mask = abs_w >= threshold
                pruned_w = w * mask
                
                new_weights = [pruned_w]
                if b is not None:
                    new_weights.append(b)
                    
                layer.set_weights(new_weights)
                
    return model

def main():
    parser = argparse.ArgumentParser(description="Manual Model Pruning for TensorFlow/Keras Models")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the original model (e.g., models/checkpoints/model.keras)")
    parser.add_argument("--sparsity", type=float, required=True, 
                        help="Percentage of weights to prune (e.g., 0.2 for 20%)")
    parser.add_argument("--technique", type=str, choices=["global", "layer"], required=True,
                        help="Type of manual unstructured pruning technique to apply")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model path {args.model_path} does not exist.")
        return
        
    if args.sparsity < 0.0 or args.sparsity >= 1.0:
        print("Error: Sparsity must be between 0.0 and 0.99.")
        return

    print(f"Loading model from {args.model_path}...")
    try:
        model = tf.keras.models.load_model(args.model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    initial_sparsity = calculate_sparsity(model)
    print(f"Initial model sparsity: {initial_sparsity*100:.2f}%")
    
    if args.technique == "global":
        model = apply_global_unstructured_pruning(model, args.sparsity)
    elif args.technique == "layer":
        model = apply_layer_unstructured_pruning(model, args.sparsity)
        
    final_sparsity = calculate_sparsity(model)
    print(f"Final model sparsity achieved: {final_sparsity*100:.2f}%")
    
    # Save the model
    base, ext = os.path.splitext(args.model_path)
    output_path = f"{base}_pruned_{args.technique}_{args.sparsity}{ext}"
    
    print(f"Saving pruned model to {output_path}...")
    try:
        model.save(output_path)
        print("Model saved successfully.")
    except Exception as e:
        print(f"Error saving pruned model: {e}")

if __name__ == "__main__":
    main()
