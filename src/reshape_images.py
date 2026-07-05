import cv2
import os
import glob
import argparse

PROCESSED_BASE_DIR = "data/processed"

def reshape_images(width, height, input_dir):
    """
    Resizes images recursively from input_dir and saves them to data/processed/{width}x{height}.
    """
    output_folder_name = f"{width}x{height}"
    output_dir = os.path.join(PROCESSED_BASE_DIR, output_folder_name)
    
    os.makedirs(output_dir, exist_ok=True)
    
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    files = []
    
    print(f"Searching for images recursively in {input_dir}...")
    for ext in extensions:
        files.extend(glob.glob(os.path.join(input_dir, '**', ext), recursive=True))
        
    if not files:
        print(f"No images found in {input_dir} or its subdirectories")
        return

    print(f"Found {len(files)} images. Resizing to {width}x{height}...")
    
    processed_count = 0
    
    for file_path in files:
        try:
            img = cv2.imread(file_path)
            if img is None:
                print(f"Failed to load: {file_path}")
                continue
                
            resized = cv2.resize(img, (width, height))
            
            rel_path = os.path.relpath(file_path, input_dir)
            output_path = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            cv2.imwrite(output_path, resized)
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print("\n========================================")
    print(f"Reshape Complete!")
    print(f"Target Size: {width}x{height}")
    print(f"Output Folder: {output_dir}")
    print(f"Processed: {processed_count}/{len(files)}")
    print("========================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reshape images for TinyML")
    parser.add_argument("--width", type=int, default=28, help="Target width")
    parser.add_argument("--height", type=int, default=28, help="Target height")
    parser.add_argument("--input_dir", type=str, default="data/processed/grayscale", help="Input directory")
    
    args = parser.parse_args()
    
    reshape_images(args.width, args.height, args.input_dir)