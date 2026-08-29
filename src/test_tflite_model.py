import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

try:
    import seaborn as sns
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
except ImportError:
    print("Por favor instala scikit-learn y seaborn para calcular las métricas.")
    print("Ejecuta: pip install scikit-learn seaborn")
    exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Test TFLite Model and Calculate Metrics exactly like test_model.py")
    parser.add_argument('--width', type=int, default=160, help="Image width")
    parser.add_argument('--height', type=int, default=120, help="Image height")
    parser.add_argument('--model_path', default="models/tflite/MobileNetV3Large+32+20+1e-06+320+320_int8.tflite", type=str, required=False, help="Ruta al modelo TFLite entrenado (ej. models/tflite/modelo_int8.tflite)")
    parser.add_argument('--splits_dir', type=str, default="data/splits",
                        help="Directorio con las particiones de split_dataset.py "
                             "(la evaluacion usa SOLO <splits_dir>/test)")
    parser.add_argument('--data_dir', type=str, default=None, help="Ruta explicita a la particion de prueba; por defecto se deduce como <splits_dir>/test")
    parser.add_argument('--class_names_path', type=str, default="models/class_names.txt", help="Ruta al txt con nombres de clases")

    args = parser.parse_args()

    # SIL evalua EXCLUSIVAMENTE la particion de test, la misma que consumen MIL,
    # PIL y HIL, para que los saltos entre niveles de la escalera sean
    # atribuibles al nivel y no a un muestreo distinto de imagenes. Nunca train
    # (es la que calibro la cuantizacion en quantize_int8_basic.py) ni val (la
    # consumieron el early stopping y la seleccion del modelo).
    if args.data_dir is None:
        args.data_dir = os.path.join(args.splits_dir, "test")

    return args

def quantize_input(data, input_details):
    """
    Si el modelo TFLite espera enteros en lugar de float32 (ej. un modelo full-INT8),
    necesitamos escalar la data manualmente según su cuantización antes de predecir.
    """
    if input_details['dtype'] == np.int8:
        scale, zero_point = input_details['quantization']
        
        # TFLite retorna listas o numpy arrays iterables
        s = scale[0] if isinstance(scale, (list, np.ndarray)) and len(scale) > 0 else scale
        z = zero_point[0] if isinstance(zero_point, (list, np.ndarray)) and len(zero_point) > 0 else zero_point
        
        if s > 0:
            # Revertir el valor al espacio cuantizado
            q_data = (data / s) + z
            q_data = np.clip(q_data, -128, 127).astype(np.int8)
            return q_data
        return data.astype(np.int8)
    # Devolver tal cual en float32 si es un modelo TFLite en float16 o fallback a float
    return data.astype(np.float32)

def dequantize_output(data, output_details):
    """
    Si el output viene cuantizado en INT8, lo debemos restaurar a probabilidad (punto flotante)
    invirtiendo su zero_point y multiplicando por escala.
    """
    if output_details['dtype'] == np.int8:
        scale, zero_point = output_details['quantization']
        
        s = scale[0] if isinstance(scale, (list, np.ndarray)) and len(scale) > 0 else scale
        z = zero_point[0] if isinstance(zero_point, (list, np.ndarray)) and len(zero_point) > 0 else zero_point
        
        if s > 0:
            dq_data = (data.astype(np.float32) - z) * s
            return dq_data
    return data.astype(np.float32)

def main():
    args = parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Debes proveer un modelo TFLite válido. No se encontró: {args.model_path}")
        return

    class_names = []
    if os.path.exists(args.class_names_path):
        with open(args.class_names_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        print(f"Nombres de clases cargados: {class_names}")

    print(f"\nInicializando TFLite Interpreter con {args.model_path}...")
    interpreter = tf.lite.Interpreter(model_path=args.model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    _, expected_height, expected_width, expected_channels = input_details['shape']

    # Limpiar espacios en blanco o comillas invisibles en la cadena
    if args.data_dir:
        args.data_dir = args.data_dir.strip().strip("'").strip('"')

    # Si la ruta lleva las dimensiones en la cadena (rutas antiguas del tipo
    # data/processed/160x120), corregirla automáticamente
    if f"{args.width}x{args.height}" in args.data_dir and (args.width != expected_width or args.height != expected_height):
        args.data_dir = args.data_dir.replace(f"{args.width}x{args.height}", f"{expected_width}x{expected_height}")
        print(f"ℹ️ Corrigiendo la ruta a {expected_width}x{expected_height} (exigido por el modelo).")

    # La geometría siempre la manda el tensor de entrada del modelo: la ruta de
    # la partición ya no codifica la resolución, así que no basta con corregirla.
    if args.width != expected_width or args.height != expected_height:
        print(f"ℹ️ Redimensionando a {expected_width}x{expected_height} (exigido por el modelo).")
        args.width = expected_width
        args.height = expected_height

    # Intentar buscar la ruta relativa estándar si falla la absoluta (común al pegar rutas)
    if not os.path.exists(args.data_dir):
        fallback_path = os.path.join(os.getcwd(), args.splits_dir, "test")
        if os.path.exists(fallback_path):
            args.data_dir = fallback_path

    if not os.path.exists(args.data_dir):
        print(f"ERROR: no se encontró la partición de prueba en {args.data_dir}")
        print("Ejecuta primero la partición del dataset:")
        print(f"  python src/split_dataset.py"
              f" --input_dir data/processed/{expected_width}x{expected_height}"
              f" --output_dir {args.splits_dir}")
        exit(1)

    print(f"Cargando dataset de prueba desde {args.data_dir}...")
    # Usamos batch_size=1 porque el intérprete TFLite generalemente infiere en ráfagas de a 1,
    # y shuffle=False para mantener correspondencia entre entrada e imagen/etiqueta.
    test_ds = tf.keras.utils.image_dataset_from_directory(
        args.data_dir,
        labels='inferred',
        label_mode='int',
        class_names=class_names if class_names else None,
        color_mode='grayscale',
        batch_size=1, 
        image_size=(args.height, args.width),
        shuffle=False
    )

    if not class_names:
        class_names = test_ds.class_names
        print(f"Nombres de clases inferidos del directorio: {class_names}")

    num_classes = len(class_names)
    
    y_true = []
    predictions = []

    print("Generando predicciones frame-by-frame con el modelo TFLite (inferencia secuencial)...")
    
    for x, y in test_ds:
        y_true.append(y.numpy()[0])
        
        # x esta escalado original entre [0..255] float32. Lo necesitamos pasar
        # dependiendo de lo que TFLite pida.
        input_data = x.numpy() 
        
        # Opcional: ajustar dimensionalidad al tensor exacto si varía
        if tuple(input_data.shape) != tuple(input_details['shape']):
            input_data = np.reshape(input_data, input_details['shape'])
        
        input_data = quantize_input(input_data, input_details)
        interpreter.set_tensor(input_details['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details['index'])
        output_data = dequantize_output(output_data, output_details)
        predictions.append(output_data[0])

    y_true = np.array(y_true)
    predictions = np.array(predictions)

    if num_classes == 1:
        # Clasificación binaria pura
        y_pred = (predictions > 0.5).astype(int).reshape(-1)
    elif num_classes == 2 and predictions.shape[1] == 1:
        # Clasificación binaria estructurada sigmoidal
        y_pred = (predictions > 0.5).astype(int).reshape(-1)
    else:
        # Multiclase Argmax (Softmax normalizado)
        y_pred = np.argmax(predictions, axis=1)

    print("\n" + "="*50)
    print("                 REPORTE DE MÉTRICAS (TFLite)")
    print("="*50)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    
    print(f"Accuracy (Exactitud): {accuracy:.4f}")
    print(f"Precision:            {precision:.4f}")
    print(f"Recall (Exhaustividad):{recall:.4f}")
    print(f"F1-Score:             {f1:.4f}")
    
    print("\nReporte de Clasificación Detallado:")
    print(classification_report(y_true, y_pred, target_names=class_names, labels=range(len(class_names)), zero_division=0))

    print("\nGenerando Matriz de Confusión...")
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'SIL Confusion Matrix - {os.path.splitext(os.path.basename(args.model_path))[0]}')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    plt.tight_layout()
    
    # Las matrices del nivel SIL viven con el resto de resultados de la escalera
    # (results/mil, results/pil, results/hil), no junto al .tflite. El nombre se
    # deriva del modelo, igual que en test_model.py y pil_benchmark.py, para que
    # dos corridas con modelos distintos no se pisen la matriz.
    out_dir = os.path.join("results", "sil")
    os.makedirs(out_dir, exist_ok=True)

    model_basename = os.path.basename(args.model_path)
    model_name_without_ext = os.path.splitext(model_basename)[0]
    cm_plot_name = f"Matriz_{model_name_without_ext}.png"
    cm_plot_path = os.path.join(out_dir, cm_plot_name)
    
    plt.savefig(cm_plot_path)
    print(f"\n Gráfico de la matriz de confusión guardado en {cm_plot_path}")

if __name__ == "__main__":
    main()
