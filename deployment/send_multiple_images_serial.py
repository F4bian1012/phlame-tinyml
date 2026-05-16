"""
send_multiple_images_serial.py
================================
Envía imágenes preprocesadas al Arduino Portenta H7 por puerto serial.

Protocolo:
  - Marcador de inicio : '#' (0x23)
  - Marcador de fin    : '@' (0x40)
  - Los bytes del payload que coincidan con '#' (0x23), '@' (0x40) o ESC (0x1B)
    se escapan enviando: ESC (0x1B) seguido del byte XOR 0x20.

Uso básico (una imagen):
  python send_multiple_images_serial.py --port COM3 --image ruta/imagen.png

Uso básico (carpeta completa):
  python send_multiple_images_serial.py --port COM3 --folder data/processed/160x120/Class1

Opciones:
  --port    Puerto serial (ej. COM3, /dev/ttyUSB0)  [requerido]
  --image   Ruta a una imagen concreta a enviar
  --folder  Carpeta con imágenes a enviar en secuencia
            (default: data\processed\160x120\Class1)
  --baud    Tasa de baudios (default: 115200)
  --width   Ancho objetivo del tensor (default: 120)
  --height  Alto objetivo del tensor (default: 160)
  --color   Enviar en color RGB (default: escala de grises)
  --delay   Segundos de espera antes de la primera transmisión (default: 1.0)
  --gap     Segundos de espera entre imágenes consecutivas (default: 0.5)
"""

import argparse
from collections import Counter
import glob
import os
import random
import time
import sys

try:
    import serial
except ImportError:
    print("ERROR: pyserial no está instalado. Ejecuta: pip install pyserial")
    sys.exit(1)

try:
    from PIL import Image
    import numpy as np
except ImportError:
    print("ERROR: Pillow y/o numpy no están instalados.")
    print("Ejecuta: pip install Pillow numpy")
    sys.exit(1)


# ──────────────────────────────────────────────
# Constantes del protocolo
# ──────────────────────────────────────────────
MARKER_START = b'#'   # 0x23
MARKER_END   = b'@'   # 0x40
ESCAPE_BYTE  = 0x1B   # ESC
BYTES_TO_ESCAPE = {0x23, 0x40, 0x1B}  # '#', '@', ESC


def escape_payload(raw_bytes: bytes) -> bytes:
    """
    Aplica el esquema de escape al payload antes de enviarlo.
    Si un byte es '#', '@' o ESC, se envía como: ESC, (byte XOR 0x20).
    """
    escaped = bytearray()
    for b in raw_bytes:
        if b in BYTES_TO_ESCAPE:
            escaped.append(ESCAPE_BYTE)
            escaped.append(b ^ 0x20)
        else:
            escaped.append(b)
    return bytes(escaped)


def preprocess_image(image_path: str, width: int, height: int, grayscale: bool) -> bytes:
    """
    Carga, redimensiona y convierte la imagen al formato esperado por el tensor.
    Retorna los bytes RAW (uint8, 0-255) en orden HWC.
    """
    img = Image.open(image_path)

    if grayscale:
        img = img.convert('L')          # escala de grises (1 canal)
    else:
        img = img.convert('RGB')        # 3 canales

    img = img.resize((width, height), Image.LANCZOS)

    arr = np.array(img, dtype=np.uint8)  # shape: (H, W) o (H, W, 3)

    # Aplanar en orden C (row-major) → bytes del tensor
    raw = arr.flatten().tobytes()

    print(f"[INFO] Imagen preprocesada: {arr.shape}  -> {len(raw)} bytes de payload")
    return raw


def build_packet(payload_raw: bytes) -> bytes:
    """
    Construye el paquete completo: MARKER_START + payload_escapado + MARKER_END
    """
    payload_escaped = escape_payload(payload_raw)
    packet = MARKER_START + payload_escaped + MARKER_END
    return packet


def send_image(port: str, baud: int, image_path: str,
               width: int, height: int, grayscale: bool,
               pre_delay: float):
    """
    Abre el puerto serial y envía la imagen con el protocolo #...@
    """
    # Preprocesar imagen
    raw_payload = preprocess_image(image_path, width, height, grayscale)
    packet = build_packet(raw_payload)

    print(f"[INFO] Conectando a {port} @ {baud} baud ...")
    try:
        ser = serial.Serial(port, baud, timeout=10)
    except serial.SerialException as e:
        print(f"ERROR al abrir el puerto: {e}")
        sys.exit(1)

    time.sleep(pre_delay)  # esperar a que el Arduino termine su setup
    print(f"[INFO] Puerto abierto. Esperando {pre_delay}s antes de enviar...")
    time.sleep(pre_delay)

    # Limpiar buffer de entrada
    ser.reset_input_buffer()

    print(f"[INFO] Enviando paquete: {len(packet)} bytes totales "
          f"(payload raw={len(raw_payload)}, escapado={len(packet)-2})")

    ser.write(packet)
    ser.flush()
    print(f"[OK]   Paquete enviado. Esperando respuesta del Arduino...\n")

    # Leer la respuesta hasta que se reciba la línea de "Listo"
    timeout = time.time() + 30  # máximo 30 segundos esperando respuesta
    while time.time() < timeout:
        if ser.in_waiting:
            line = ser.readline().decode('utf-8', errors='replace').rstrip()
            print(f"  Arduino >>> {line}")
            if "siguiente imagen" in line or "Listo" in line:
                break
        else:
            time.sleep(0.05)

    ser.close()
    print("\n[INFO] Puerto cerrado. Fin de transmisión.")


def send_folder(port: str, baud: int, folder: str,
                width: int, height: int, grayscale: bool,
                pre_delay: float, gap: float,
                n_samples: int = None):
    """
    Envía imágenes de `folder` al Arduino en secuencia.

    Si `n_samples` es None envía todas las imágenes encontradas.
    Si `n_samples` es un entero, selecciona ese número de imágenes
    aleatoriamente (sin reemplazo) antes de enviarlas.

    El Arduino retorna únicamente el número entero de la clase predicha.
    Al finalizar, imprime un resumen con el conteo y porcentaje por clase.

    Parámetros
    ----------
    port      : Puerto serial (ej. 'COM9' o '/dev/ttyUSB0')
    baud      : Baudrate
    folder    : Carpeta que contiene las imágenes
    width     : Ancho del tensor de entrada
    height    : Alto del tensor de entrada
    grayscale : True → escala de grises, False → RGB
    pre_delay : Segundos de espera antes de enviar la primera imagen
    gap       : Segundos de espera entre imágenes consecutivas
    n_samples : Número de imágenes a seleccionar aleatoriamente.
                Si es None (o >= total disponibles), se envían todas.
    """
    extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif')
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(folder, ext)))
        image_paths.extend(glob.glob(os.path.join(folder, ext.upper())))

    # Eliminar duplicados y ordenar para reproducibilidad
    image_paths = sorted(set(image_paths))

    if not image_paths:
        print(f"[ERROR] No se encontraron imágenes en: {folder}")
        sys.exit(1)

    # Muestreo aleatorio si el usuario especificó --count
    available = len(image_paths)
    if n_samples is not None and n_samples < available:
        image_paths = random.sample(image_paths, n_samples)
        print(f"[INFO] Seleccionadas {n_samples} imágenes aleatoriamente "
              f"de {available} disponibles en '{folder}'")
    else:
        print(f"[INFO] {available} imágenes encontradas en '{folder}' (enviando todas)")

    # Abrir el puerto una sola vez
    print(f"[INFO] Conectando a {port} @ {baud} baud ...")
    try:
        ser = serial.Serial(port, baud, timeout=10)
    except serial.SerialException as e:
        print(f"ERROR al abrir el puerto: {e}")
        sys.exit(1)

    print(f"[INFO] Puerto abierto. Esperando {pre_delay}s antes de enviar...")
    time.sleep(pre_delay)

    total = len(image_paths)  # puede ser < available si se muestreo

    # Contador de clases predichas
    class_counts: Counter = Counter()
    errors = 0  # imágenes sin respuesta válida

    for idx, image_path in enumerate(image_paths, start=1):
        print(f"\n[INFO] ({idx}/{total}) Procesando: {os.path.basename(image_path)}")

        raw_payload = preprocess_image(image_path, width, height, grayscale)
        packet = build_packet(raw_payload)

        ser.reset_input_buffer()
        print(f"[INFO] Enviando paquete: {len(packet)} bytes totales "
              f"(payload raw={len(raw_payload)}, escapado={len(packet)-2})")

        ser.write(packet)
        ser.flush()
        print(f"[OK]   Paquete enviado. Esperando clase predicha...")

        # Leer la respuesta: el Arduino envía solo el número de clase
        predicted_class = None
        timeout = time.time() + 30
        while time.time() < timeout:
            if ser.in_waiting:
                line = ser.readline().decode('utf-8', errors='replace').rstrip()
                print(f"  Arduino >>> {line}")
                # Intentar parsear como entero (clase predicha)
                try:
                    predicted_class = int(line.strip())
                    break
                except ValueError:
                    pass  # ignorar líneas que no sean un entero
            else:
                time.sleep(0.05)

        if predicted_class is not None:
            class_counts[predicted_class] += 1
            print(f"  → Clase predicha: {predicted_class}")
        else:
            errors += 1
            print(f"  [WARN] No se recibió clase predicha para esta imagen.")

        if idx < total:
            print(f"[INFO] Esperando {gap}s antes de la siguiente imagen...")
            time.sleep(gap)

    ser.close()
    print("\n[INFO] Puerto cerrado. Todas las imágenes enviadas.")

    # ── Resumen de predicciones ──────────────────────────────────────
    respondidas = total - errors
    print("\n" + "=" * 45)
    print(f"  RESUMEN DE PREDICCIONES")
    print(f"  Carpeta : {folder}")
    print(f"  Total   : {total} imágenes  |  Respondidas: {respondidas}  |  Sin respuesta: {errors}")
    print("=" * 45)
    if class_counts:
        print(f"  {'Clase':>6}  {'Cantidad':>9}  {'Porcentaje':>11}")
        print("  " + "-" * 32)
        for cls in sorted(class_counts):
            cnt = class_counts[cls]
            pct = cnt / respondidas * 100 if respondidas else 0
            print(f"  {cls:>6}  {cnt:>9}  {pct:>10.1f}%")
    print("=" * 45)


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Envía imagen(es) al Arduino por serial (protocolo #...@)"
    )
    parser.add_argument('--port',   default="COM9", required=False,
                        help="Puerto serial (ej. COM9 o /dev/ttyUSB0)")

    # Origen de la imagen: una sola imagen O una carpeta (mutuamente excluyentes)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--image',  default=None,
                        help="Ruta a una imagen concreta a enviar")
    source.add_argument('--folder', default=None,
                        help="Carpeta con imágenes a enviar en secuencia "
                             "(default: data\\processed\\160x120\\Class1)")

    parser.add_argument('--baud',   type=int, default=115200,
                        help="Baudrate (default: 115200)")
    parser.add_argument('--width',  type=int, default=320,
                        help="Ancho del tensor de entrada (default: 120)")
    parser.add_argument('--height', type=int, default=320,
                        help="Alto del tensor de entrada (default: 160)")
    parser.add_argument('--color',  action='store_true',
                        help="Enviar en color RGB (default: escala de grises)")
    parser.add_argument('--delay',  type=float, default=1.0,
                        help="Segundos de espera antes de enviar (default: 1.0)")
    parser.add_argument('--gap',    type=float, default=0.5,
                        help="Segundos entre imágenes consecutivas (default: 0.5)")
    parser.add_argument('--count',  type=int, default=None, metavar='N',
                        help="Número de imágenes aleatorias a enviar de la carpeta. "
                             "Si no se especifica, se envían todas.")
    return parser.parse_args()


def normalize_port(port: str) -> str:
    """
    Normaliza el nombre del puerto según la plataforma:
      - Windows nativo : COM9+ → \\\\.\\COMx
      - WSL (Linux)    : COMx  → /dev/ttySx  (COM1→ttyS1, COM7→ttyS7, etc.)
      - Linux nativo   : se usa tal cual (/dev/ttyUSB0, etc.)
    """
    import re
    m = re.match(r'^COM(\d+)$', port.strip().upper())
    if m:
        n = int(m.group(1))
        # Detectar WSL: sys.platform es 'linux' pero existe /proc/version con 'microsoft'
        is_wsl = False
        if sys.platform != 'win32':
            try:
                with open('/proc/version', 'r') as f:
                    is_wsl = 'microsoft' in f.read().lower()
            except OSError:
                pass

        if is_wsl:
            port = f'/dev/ttyS{n}'
            print(f"[INFO] Puerto normalizado para WSL: {port}")
        elif sys.platform == 'win32' and n >= 9:
            port = f'\\\\.\\COM{n}'
            print(f"[INFO] Puerto normalizado para Windows: {port}")
    return port


DEFAULT_FOLDER = os.path.join('data', 'processed', '160x120', 'Class1')

if __name__ == '__main__':
    args = parse_args()
    port = normalize_port(args.port)

    if args.image:
        # Enviar una única imagen
        send_image(
            port=port,
            baud=args.baud,
            image_path=args.image,
            width=args.width,
            height=args.height,
            grayscale=not args.color,
            pre_delay=args.delay,
        )
    else:
        # Enviar imágenes de la carpeta (todas o N aleatorias)
        folder = args.folder if args.folder is not None else DEFAULT_FOLDER
        send_folder(
            port=port,
            baud=args.baud,
            folder=folder,
            width=args.width,
            height=args.height,
            grayscale=not args.color,
            pre_delay=args.delay,
            gap=args.gap,
            n_samples=args.count,
        )
