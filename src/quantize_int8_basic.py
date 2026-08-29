import os
import glob
import argparse
import tensorflow as tf

def representative_data_gen(dataset, n_samples=100):
    """Generador que toma muestras del dataset para calibrar las activaciones a INT8.

    IMPORTANTE: el dataset debe provenir SIEMPRE de la particion de entrenamiento
    (data/splits/train). Calibrar con imagenes de test fijaria los rangos de
    activacion (scale/zero_point) a la distribucion del conjunto de evaluacion,
    sesgando al alza la accuracy de SIL, PIL y HIL.
    """
    def gen():
        # n_samples imagenes, de 1 en 1 como espera TFLiteConverter
        for images, _ in dataset.unbatch().batch(1).take(n_samples):
            yield [images]
    return gen

def main():
    parser = argparse.ArgumentParser(description="Cuantizar modelo a INT8.")
    parser.add_argument("--model_path", type=str, help="Ruta al modelo .keras a cuantizar")
    parser.add_argument("--splits_dir", type=str, default="data/splits",
                        help="Directorio con las particiones de split_dataset.py "
                             "(la calibracion usa SOLO <splits_dir>/train)")
    parser.add_argument("--calib_samples", type=int, default=100,
                        help="Imagenes de calibracion tomadas de la particion train")
    parser.add_argument("--calib_seed", type=int, default=42,
                        help="Semilla del mezclado de calibracion; fija que .tflite se produce")
    args = parser.parse_args()

    if args.model_path:
        model_path = args.model_path
        if not os.path.exists(model_path):
            print(f"Error: No se encontró el modelo en {model_path}.")
            return
    else:
        checkpoints = glob.glob("models/checkpoints/*.keras")
        if not checkpoints:
            print("Error: No se encontraron modelos (.keras) en la carpeta models/checkpoints.")
            print("Asegúrate de haber entrenado un modelo primero.")
            return
            
        # Usaremos el primer modelo que se encuentre (puedes cambiar esta lógica si tienes varios)
        model_path = checkpoints[0]
        
    print(f"Modelo seleccionado para cuantizar: {model_path}")

    # Extraer el basename para deducir dimensiones y nombre de salida
    # Ejemplo de formato esperado: MobileNet+32...val+width+height.keras
    basename = os.path.basename(model_path).replace(".keras", "")
    
    # Remove any pruning suffix before splitting by '+' to correctly parse dimensions
    if "_pruned" in basename:
        basename = basename.split("_pruned")[0]
        
    parts = basename.split('+')
    
    # Formato actual (sin validation_split): {modelo}+{batch}+{epochs}+{lr}+{W}+{H} -> 6 partes
    # Formato antiguo (con validation_split):                                      -> 7 partes
    if len(parts) >= 6:
        width = int(parts[-2])
        height = int(parts[-1])
    else:
        print("No se pudieron inferir las dimensiones del nombre. Usando 160x120 por defecto.")
        width, height = 160, 120
        
    # La calibracion usa EXCLUSIVAMENTE la particion de entrenamiento.
    # Nunca test (sesgaria la evaluacion) y tampoco val (ya lo consumen el
    # early stopping y ReduceLROnPlateau para seleccionar el modelo).
    data_dir = os.path.join(args.splits_dir, "train")
    
    if not os.path.exists(data_dir):
        print(f"Error: no se encontro la particion de entrenamiento en {data_dir}.")
        print("Genera las particiones primero:")
        print(f"  python src/split_dataset.py --input_dir data/processed/{width}x{height}"
              f" --output_dir {args.splits_dir}")
        return

    print(f"\nCalibrando con {args.calib_samples} imagenes de {data_dir} "
          f"(semilla {args.calib_seed})...")
    calib_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        # sin validation_split ni subset: la carpeta YA es la particion train
        shuffle=True,            # necesario: los archivos vienen agrupados por clase
        seed=args.calib_seed,    # con semilla fija el mezclado es reproducible
        color_mode='grayscale',
        image_size=(height, width),
        batch_size=32
    )

    print("\nCargando el modelo...")
    model = tf.keras.models.load_model(model_path)

    print("\nConfigurando la conversión de Full Integer Quantization (INT8)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Pasar el dataset representativo (es mandatorio para conversiones full-INT8)
    converter.representative_dataset = representative_data_gen(calib_ds, args.calib_samples)
    
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    # Definir entradas y salidas en formato int8 para máxima compatibilidad Edge/MCU
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    print("Convirtiendo modelo. Esto tomará unos segundos...")
    try:
        tflite_model = converter.convert()
        
        tflite_dir = "models/tflite"
        os.makedirs(tflite_dir, exist_ok=True)
        
        out_path = os.path.join(tflite_dir, f"{basename}_int8.tflite")
        with open(out_path, "wb") as f:
            f.write(tflite_model)
            
        print(f"\n ¡Conversión exitosa!")
        print(f"Modelo guardado en: {out_path}")
        print(f"Tamaño final: {len(tflite_model) / 1024:.2f} KB")
        
    except Exception as e:
        print(f"\n Ocurrió un error en la conversión (posibles operaciones incompatibles): {e}")

if __name__ == "__main__":
    main()
