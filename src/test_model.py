import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import os
from datetime import datetime, timezone

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

def _json_safe(o):
    """Convierte tipos de numpy a tipos nativos de Python.

    classification_report(output_dict=True) devuelve el campo "support" como
    np.int64 y las metricas como np.float64, y json.dump no sabe serializarlos:
    sin esto el volcado falla con TypeError despues de haber corrido toda la
    inferencia.
    """
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError("No se puede serializar a JSON: %r (%s)" % (o, type(o).__name__))


def parse_args():
    parser = argparse.ArgumentParser(description="Test Model and Calculate Metrics")
    parser.add_argument('--width', type=int, default=160, help="Image width")
    parser.add_argument('--height', type=int, default=120, help="Image height")
    parser.add_argument('--model_path', type=str, required=True, help="Ruta al modelo .keras a evaluar (obligatorio)")
    parser.add_argument('--data_dir', type=str, default=None, help="Ruta a la partición de prueba (default: data/splits/test, generada por split_dataset.py)")
    parser.add_argument('--class_names_path', type=str, default="models/class_names.txt", help="Ruta al archivo txt con los nombres de las clases")
    
    args = parser.parse_args()

    # MIL evalua la particion de test reservada por split_dataset.py, no el
    # directorio completo: de lo contrario la accuracy reportada incluiria las
    # imagenes con las que se entreno el modelo.
    if args.data_dir is None:
        args.data_dir = os.path.join("data", "splits", "test")

    return args

def main():
    args = parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: No se encontró el modelo en {args.model_path}")
        return

    if not os.path.exists(args.data_dir):
        print(f"ERROR: no se encontró la partición de prueba en {args.data_dir}")
        print("Ejecuta primero la partición del dataset:")
        print(f"  python src/split_dataset.py"
              f" --input_dir data/processed/{args.width}x{args.height}"
              f" --output_dir data/splits")
        exit(1)

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
    # Se pide dos veces: el texto para la consola y el dict para el JSON. Son la
    # misma llamada con output_dict, asi que las cifras no pueden divergir.
    print(classification_report(y_true, y_pred, target_names=class_names, labels=range(len(class_names)), zero_division=0))
    report_dict = classification_report(y_true, y_pred, target_names=class_names, labels=range(len(class_names)), zero_division=0, output_dict=True)

    print("\nGenerando Matriz de Confusión...")
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusión')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    plt.tight_layout()
    
    # Las matrices del nivel MIL viven junto al resto de resultados de la
    # escalera (results/pil, results/hil), no mezcladas con los checkpoints.
    out_dir = os.path.join("results", "mil")
    os.makedirs(out_dir, exist_ok=True)

    model_basename = os.path.basename(args.model_path)
    model_name_without_ext = os.path.splitext(model_basename)[0]
    cm_plot_name = f"Matriz_{model_name_without_ext}.png"
    cm_plot_path = os.path.join(out_dir, cm_plot_name)
    
    plt.savefig(cm_plot_path)
    print(f"Gráfico de la matriz de confusión guardado en {cm_plot_path}")

    # Ademas del PNG se guarda un JSON con las mismas cifras. La imagen sirve
    # para mirar, pero no para comparar niveles de la escalera ni para que otro
    # script lea los numeros; el JSON si. Mismo criterio de nombre que
    # Matriz_*.png: derivado del modelo, para que dos corridas no se pisen.
    metrics_path = os.path.join(out_dir, f"metrics_{model_name_without_ext}.json")

    resultados = {
        "nivel": "MIL",
        "generado_utc": datetime.now(timezone.utc).isoformat(),
        "modelo": {
            "path": args.model_path,
            "nombre": model_name_without_ext,
        },
        "dataset": {
            "data_dir": args.data_dir,
            "n_muestras": int(len(y_true)),
            "class_names": list(class_names),
            "image_size": {"width": args.width, "height": args.height},
        },
        "metricas_globales": {
            "accuracy": float(accuracy),
            "precision_weighted": float(precision),
            "recall_weighted": float(recall),
            "f1_weighted": float(f1),
        },
        "metricas_por_clase": report_dict,
        "matriz_confusion": {
            "labels": list(class_names),
            "matriz": cm.tolist(),
            "orden": "filas = etiqueta real, columnas = etiqueta predicha",
        },
        "artefactos": {
            "matriz_png": cm_plot_path,
        },
    }

    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(resultados, fh, indent=2, ensure_ascii=False, default=_json_safe)
    print(f"Métricas en JSON guardadas en {metrics_path}")


if __name__ == "__main__":
    main()
