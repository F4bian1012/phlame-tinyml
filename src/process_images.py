import cv2
import os
import glob
import argparse

def process_images(raw_dir, processed_dir):
    """
    Convierte imágenes a escala de grises controlando la compresión
    para evitar que aumente el tamaño del archivo.
    """
    os.makedirs(processed_dir, exist_ok=True)
    
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    files = []
    
    print(f"Buscando imágenes en {raw_dir}...")
    for ext in extensions:
        files.extend(glob.glob(os.path.join(raw_dir, '**', ext), recursive=True))
        
    if not files:
        print("No se encontraron imágenes.")
        return

    print(f"Encontradas {len(files)} imágenes. Iniciando procesamiento...\n")
    
    processed_count = 0
    
    for file_path in files:
        try:
            
            img = cv2.imread(file_path)
            if img is None:
                print(f"Error al cargar: {file_path}")
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
            print(f"Error procesando {file_path}: {e}")

    print("\n" + "="*40)
    print(f"Procesamiento Completado")
    print(f"Total: {processed_count} imágenes")
    print(f"Carpeta: {processed_dir}")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Procesa imágenes a escala de grises.")
    parser.add_argument("--raw_path", type=str, default="data/raw", help="Ruta a las imágenes originales")
    parser.add_argument("--path_processed", type=str, default="data/processed/grayscale", help="Ruta de destino para imágenes procesadas")
    args = parser.parse_args()
    
    process_images(args.raw_path, args.path_processed)