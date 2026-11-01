import cv2
import os
import argparse

def process_images(raw_dir, processed_dir):
    """
    Converts images to grayscale while controlling the compression
    so that the file size does not grow.
    """
    os.makedirs(processed_dir, exist_ok=True)

    # Accepted extensions, compared in lowercase: .JPG, .Png, etc. also
    # count on Linux/macOS (glob skipped them there, being case-sensitive).
    extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    files = []

    print(f"Looking for images in {raw_dir}...")
    for root, _, names in os.walk(raw_dir):
        for name in sorted(names):
            if os.path.splitext(name)[1].lower() in extensions:
                files.append(os.path.join(root, name))

    if not files:
        print("No images found.")
        return

    print(f"Found {len(files)} images. Starting processing...\n")

    processed_count = 0

    for file_path in files:
        try:

            img = cv2.imread(file_path)
            if img is None:
                print(f"Error loading: {file_path}")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            rel_path = os.path.relpath(file_path, raw_dir)
            output_path = os.path.join(processed_dir, rel_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            filename = os.path.basename(file_path)
            ext_lower = os.path.splitext(filename)[1].lower()

            encode_params = []

            if ext_lower in ['.jpg', '.jpeg']:
                encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 70]

            elif ext_lower == '.png':
                encode_params = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]

            cv2.imwrite(output_path, gray, encode_params)

            processed_count += 1

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print("\n" + "="*40)
    print("Processing Completed")
    print(f"Total: {processed_count} images")
    print(f"Folder: {processed_dir}")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converts images to grayscale.")
    parser.add_argument("--raw_path", type=str, default="data/raw", help="Path to the original images")
    parser.add_argument("--path_processed", type=str, default="data/processed/grayscale", help="Destination path for the processed images")
    args = parser.parse_args()

    process_images(args.raw_path, args.path_processed)
