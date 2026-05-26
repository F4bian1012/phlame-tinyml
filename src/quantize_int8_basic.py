import os
import glob
import argparse
import tensorflow as tf

def representative_data_gen(dataset):
    """Generador que toma muestras del dataset para calibrar las activaciones a INT8."""
    def gen():
        # Tomar 100 imágenes del dataset (1 a la vez como espera TFLiteConverter)
        for images, _ in dataset.unbatch().batch(1).take(100):
            yield [images]
    return gen

def main():
    parser = argparse.ArgumentParser(description="Cuantizar modelo a INT8.")
    parser.add_argument("--model_path", type=str, help="Ruta al modelo .keras a cuantizar")
    args = parser.parse_args()

    if args.model_path:
        model_path = args.model_path
        if not os.path.exists(model_path):
            print(f"Error: No se encontró el modelo en {model_path}.")
            return
    else:
        # 1. Buscar automáticamente el modelo .keras en models/checkpoints
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
    
    if len(parts) >= 7:
        width = int(parts[-2])
        height = int(parts[-1])
    else:
        print("No se pudieron inferir las dimensiones del nombre. Usando 96x96 por defecto.")
        width, height = 96, 96
        
    data_dir = f"data/processed/{width}x{height}"
    
    if not os.path.exists(data_dir):
        print(f"Error: Directorio de datos no encontrado en {data_dir}.")
        print("Se requieren las imágenes del proyecto para calibrar con precisión los rangos INT8.")
        return

    # 2. Cargar un dataset representativo para realizar la calibración
    print(f"\nCargando datos desde {data_dir} para el dataset representativo...")
    calib_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.1, # Solo un pequeño margen para calibración
        subset="training",
        seed=123,
        color_mode='grayscale',
        image_size=(height, width),
        batch_size=32
    )

    # 3. Cargar el modelo base
    print("\nCargando el modelo...")
    model = tf.keras.models.load_model(model_path)

    # 4. Configurar el convertidor TFLite
    print("\nConfigurando la conversión de Full Integer Quantization (INT8)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Habilitar opciones de cuantización predeterminadas
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Pasar el dataset representativo (es mandatorio para conversiones full-INT8)
    converter.representative_dataset = representative_data_gen(calib_ds)
    
    # Forzar que todas las operaciones soportadas sean estrictamente INT8
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    # Definir entradas y salidas en formato int8 para máxima compatibilidad Edge/MCU
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    # 5. Ejecutar la conversión
    print("Convirtiendo modelo. Esto tomará unos segundos...")
    try:
        tflite_model = converter.convert()
        
        # 6. Guardar en disco
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
