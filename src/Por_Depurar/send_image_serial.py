"""
send_image_serial.py
=====================
Envía una imagen preprocesada al Arduino Portenta H7 por puerto serial.

Protocolo:
  - Marcador de inicio : '#' (0x23)
  - Marcador de fin    : '@' (0x40)
  - Los bytes del payload que coincidan con '#' (0x23), '@' (0x40) o ESC (0x1B)
    se escapan enviando: ESC (0x1B) seguido del byte XOR 0x20.

Uso básico:
  python send_image_serial.py --port COM3 --image ruta/imagen.png

Opciones:
  --port    Puerto serial (ej. COM3, /dev/ttyUSB0)  [requerido]
  --image   Ruta a la imagen a enviar               [requerido]
  --baud    Tasa de baudios (default: 115200)
  --width   Ancho objetivo del tensor (default: 48)
  --height  Alto objetivo del tensor (default: 48)
  --gray    Convertir a escala de grises (flag, default: True)
  --delay   Segundos de espera antes de enviar (default: 3)
"""

import argparse
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


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Envía una imagen al Arduino por serial (protocolo #...@)"
    )
    parser.add_argument('--port', default="COM7", required=False,
                        help="Puerto serial (ej. COM9 o /dev/ttyUSB0)")
    parser.add_argument('--image',  required=True,
                        help="Ruta a la imagen de entrada")
    parser.add_argument('--baud',   type=int, default=115200,
                        help="Baudrate (default: 115200)")
    parser.add_argument('--width',  type=int, default=160,
                        help="Ancho del tensor de entrada (default: 48)")
    parser.add_argument('--height', type=int, default=120,
                        help="Alto del tensor de entrada (default: 48)")
    parser.add_argument('--color',  action='store_true',
                        help="Enviar en color RGB (default: escala de grises)")
    parser.add_argument('--delay',  type=float, default=1.0,
                        help="Segundos de espera antes de enviar (default: 3)")
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


if __name__ == '__main__':
    args = parse_args()
    port = normalize_port(args.port)
    send_image(
        port=port,
        baud=args.baud,
        image_path=args.image,
        width=args.width,
        height=args.height,
        grayscale=not args.color,
        pre_delay=args.delay,
    )
