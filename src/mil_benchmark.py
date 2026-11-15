import os
import sys

import artifacts
import cli

BATCH_SIZE = 32


def parse_args():
    parser = cli.ArgumentParser(description="MIL level: evaluate a .keras model on the test split and compute its metrics")
    parser.add_argument('--width', type=int, default=160, help="Image width")
    parser.add_argument('--height', type=int, default=120, help="Image height")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the .keras model to evaluate (required)")
    parser.add_argument('--splits_dir', type=str, default="data/splits",
                        help="Directory holding the splits produced by split_dataset.py "
                             "(evaluation uses ONLY <splits_dir>/test)")
    parser.add_argument('--data_dir', type=str, default=None, help="Explicit path to the test split; defaults to <splits_dir>/test")
    parser.add_argument('--class_names_path', type=str, default=None, help="Path to the txt file with the class names (default: the .classes.txt next to the model, else models/class_names.txt)")
    parser.add_argument('--output_dir', type=str, default=os.path.join("results", "mil"), help="Output directory for the matrix, JSON and CSV (default: results/mil)")
    parser.add_argument('--overwrite', action='store_true', help="Replace the matrix, JSON and CSV if they already exist")

    args = parser.parse_args()

    # MIL evaluates the test split held out by split_dataset.py, not the full
    # directory: otherwise the reported accuracy would include the images the
    # model was trained on.
    if args.data_dir is None:
        args.data_dir = os.path.join(args.splits_dir, "test")

    return args


if __name__ == "__main__":
    # The command line is read before TensorFlow is imported, which takes
    # several seconds: --help and a wrong option answer at once.
    ARGS = parse_args()

from tensorflow import keras  # noqa: E402

import evaluation  # noqa: E402  (the part shared with sil_benchmark.py)
import images  # noqa: E402


def main(args=None):
    if args is None:
        args = parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: model not found at {args.model_path}")
        sys.exit(1)

    if not os.path.exists(args.data_dir):
        evaluation.stop_without_test_split(args.data_dir, args.width, args.height, args.splits_dir)

    evaluation.refuse_to_replace_results(args.output_dir, args.model_path, args.overwrite)

    class_names, class_names_path = evaluation.read_class_names(args.model_path, args.class_names_path)
    class_names = evaluation.class_names_of_this_split(class_names, class_names_path, args.data_dir)

    print(f"\nLoading the model from {args.model_path}...")
    model = keras.models.load_model(args.model_path, safe_mode=False)

    print(f"Loading the test dataset from {args.data_dir}...")
    # Read with the loader shared by SIL and the PIL bench (images.py), so the
    # three levels receive the same bytes for the same file. The order is fixed
    # (classes, then file names), which keeps y_true aligned with y_pred.
    inferred = not class_names
    test_images, y_true, rel_paths, class_names = images.load_folder(
        args.data_dir, args.width, args.height, class_names if class_names else None)
    evaluation.describe_test_split(rel_paths, class_names, inferred)

    print("Generating predictions with the model (this may take a few seconds)...")
    predictions = model.predict(images.as_model_input(test_images), batch_size=BATCH_SIZE)
    y_pred = evaluation.predicted_labels(predictions, len(class_names))

    evaluation.write_level_results(
        level="MIL", model_path=args.model_path, output_dir=args.output_dir,
        y_true=y_true, y_pred=y_pred, rel_paths=rel_paths, class_names=class_names,
        dataset={"data_dir": args.data_dir},
        image_size=(args.width, args.height),
        # What ran the float model and computed the figures (see artifacts.software_versions).
        software=artifacts.software_versions(
            ["tensorflow", "keras", "numpy", "scikit_learn", "pillow"], gpus=True),
        report_title="                 METRICS REPORT",
        plot_title='Confusion Matrix')


if __name__ == "__main__":
    main(ARGS)
