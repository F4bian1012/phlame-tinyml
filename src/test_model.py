import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# Dependencias adicionales necesarias para las métricas y la matriz de confusión:
# pip install scikit-learn seaborn
try:
    import seaborn as sns
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
except ImportError:
    print("Por favor instala scikit-learn y seaborn para calcular las métricas.")
    print("Ejecuta: pip install scikit-learn seaborn")
    exit(1)

BATCH_SIZE = 32

def parse_args():
    parser = argparse.ArgumentParser(description="Test Model and Calculate Metrics")
    parser.add_argument('--width', type=int, default=96, help="Image width")
    parser.add_argument('--height', type=int, default=96, help="Image height")
    parser.add_argument('--learning_rate', type=float, default=0.0001, help="Learning rate")
    parser.add_argument('--model_path', type=str, default=None, help="Ruta al modelo entrenado")
    parser.add_argument('--data_dir', type=str, default=None, help="Ruta al directorio con imágenes de prueba (debe contener subcarpetas por clase)")
    parser.add_argument('--class_names_path', type=str, default="models/class_names.txt", help="Ruta al archivo txt con los nombres de las clases")
    
    args = parser.parse_args()
    
    if args.model_path is None:
        args.model_path = f"models/checkpoints/MobileNet+32+20+{args.learning_rate}+0.2+{args.width}+{args.height}.keras"
        
    if args.data_dir is None:
        args.data_dir = f"data/processed/{args.width}x{args.height}"
        
    return args

def main():
    args = parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: No se encontró el modelo en {args.model_path}")
        return

    if not os.path.exists(args.data_dir):
        print(f"Error: No se encontró el directorio de datos en {args.data_dir}")
        return  

    class_names = []
    if os.path.exists(args.class_names_path):
        with open(args.class_names_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        print(f"Nombres de clases cargados: {class_names}")

    print(f"\nCargando el modelo desde {args.model_path}...")
    model = keras.models.load_model(args.model_path, safe_mode=False)

    print(f"Cargando dataset de prueba desde {args.data_dir}...")
    # Usamos shuffle=False para mantener el orden de las imágenes y alinear y_true con y_pred
    test_ds = tf.keras.utils.image_dataset_from_directory(
        args.data_dir,
        labels='inferred',
        label_mode='int',
        class_names=class_names if class_names else None,
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        image_size=(args.height, args.width),
        shuffle=False
    )

    if not class_names:
        class_names = test_ds.class_names
        print(f"Nombres de clases inferidos del directorio: {class_names}")

    num_classes = len(class_names)
    
    print("Extrayendo etiquetas reales...")
    y_true = np.concatenate([y.numpy() for x, y in test_ds], axis=0)
    
    print("Generando predicciones con el modelo (esto puede tardar unos segundos)...")
    predictions = model.predict(test_ds)
    
    if num_classes == 1:
        # Clasificación binaria con 1 sola clase definida (salida sigmoid u otra)
        y_pred = (predictions > 0.5).astype(int).reshape(-1)
    elif num_classes == 2 and predictions.shape[1] == 1:
        # Clasificación binaria (salida sigmoid)
        y_pred = (predictions > 0.5).astype(int).reshape(-1)
    else:
        # Clasificación multiclase (salida softmax)
        y_pred = np.argmax(predictions, axis=1)

    print("\n" + "="*50)
    print("                 REPORTE DE MÉTRICAS")
    print("="*50)
    
    # Calcular métricas globales (weighted sirve bien si hay clases desbalanceadas)
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    
    print(f"Accuracy (Exactitud): {accuracy:.4f}")
    print(f"Precision:            {precision:.4f}")
    print(f"Recall (Exhaustividad):{recall:.4f}")
    print(f"F1-Score:             {f1:.4f}")
    
    # Reporte detallado por clase
    print("\nReporte de Clasificación Detallado:")
    print(classification_report(y_true, y_pred, target_names=class_names, labels=range(len(class_names)), zero_division=0))

    print("\nGenerando Matriz de Confusión...")
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusión')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    plt.tight_layout()
    
    out_dir = os.path.dirname(args.model_path)
    if not out_dir:
        out_dir = "."
        
    model_basename = os.path.basename(args.model_path)
    model_name_without_ext = os.path.splitext(model_basename)[0]
    cm_plot_name = f"Matriz_{model_name_without_ext}.png"
    cm_plot_path = os.path.join(out_dir, cm_plot_name)
    
    plt.savefig(cm_plot_path)
    print(f"Gráfico de la matriz de confusión guardado en {cm_plot_path}")

if __name__ == "__main__":
    main()
