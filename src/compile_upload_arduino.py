import argparse
import subprocess
import json
import sys
import os

try:
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass
# FQBN configurado para Portenta M7
FQBN_BASE = "arduino:mbed_portenta:envie_m7"
FQBN_FULL = f"{FQBN_BASE}:split=100_0"

def ejecutar(comando):
    """Ejecuta un comando en la terminal y retorna éxito y su salida."""
    res = subprocess.run(comando, shell=True, capture_output=True, text=True)
    if res.returncode == 0:
        return True, res.stdout
    else:
        print(f" Error al ejecutar:\n{comando}\nDetalle:\n{res.stderr}")
        return False, res.stderr

def check_core_installed():
    """Verifica si el core de portenta está instalado, y si no, lo instala."""
    print("-> Verificando cores instalados...")
    success, output = ejecutar("arduino-cli core list --format json")
    if not success:
        print(" No se pudo verificar la lista de cores. ¿Está arduino-cli instalado y en el PATH?")
        sys.exit(1)
        
    try:
        data = json.loads(output)
        cores = data.get("platforms", []) if isinstance(data, dict) else data
        for core in cores:
            if isinstance(core, dict) and core.get("id") == "arduino:mbed_portenta":
                print("Core arduino:mbed_portenta ya está instalado.")
                return
    except json.JSONDecodeError:
        print(" Error al procesar la salida JSON de arduino-cli.")
        sys.exit(1)
        
    print("-> Core no encontrado. Instalando arduino:mbed_portenta (esto puede tardar unos minutos)...")
    success, _ = ejecutar("arduino-cli core install arduino:mbed_portenta")
    if not success:
        print(" Fallo al instalar el core. Por favor instálalo manualmente.")
        sys.exit(1)
    print(" Core instalado exitosamente.")

def find_portenta_port():
    """Busca una placa Portenta conectada usando arduino-cli y retorna su puerto COM."""
    print("-> Buscando placa Portenta conectada...")
    success, output = ejecutar("arduino-cli board list --format json")
    if not success:
        print(" No se pudo obtener la lista de placas conectadas.")
        sys.exit(1)
        
    try:
        data = json.loads(output)
        boards = data.get("detected_ports", []) if isinstance(data, dict) else data
        for board in boards:
            if not isinstance(board, dict): continue
            matching = board.get("matching_boards", [])
            for match in matching:
                fqbn = match.get("fqbn", "")
                name = match.get("name", "").lower()
                
                if "portenta" in name or FQBN_BASE in fqbn:
                    port = board.get("port", {}).get("address")
                    if port:
                        print(f" Encontrada placa: {match.get('name')} en el puerto {port}")
                        return port
    except json.JSONDecodeError:
        pass
        
    print(" No se detectó ninguna placa Portenta automáticamente conectada al equipo.")
    return None

def main():
    parser = argparse.ArgumentParser(description="Compilar y subir proyecto a Arduino Portenta M7.")
    parser.add_argument("--path_proyecto", 
                        default="../../deployment/arduino_project_test/",
                        help="Ruta al proyecto de Arduino a compilar y subir.")
    
    args = parser.parse_args()
    path_proyecto = os.path.abspath(args.path_proyecto)

    if not os.path.exists(path_proyecto):
        print(f" La ruta especificada para el proyecto no existe: {path_proyecto}")
        sys.exit(1)

    check_core_installed()

    puerto = find_portenta_port()
    if not puerto:
        print("-> Por favor, conecta la placa e inténtalo de nuevo.")
        sys.exit(1)

    print(f"\n=== 1. Compilando proyecto en: {path_proyecto} ===")
    cmd_compile = f"arduino-cli compile --fqbn {FQBN_FULL} --warnings none \"{path_proyecto}\""
    print(f"> {cmd_compile}")
    success, _ = ejecutar(cmd_compile)
    if not success:
        sys.exit(1)
    print(" Compilación exitosa.")

    print("\n=== 2. Subiendo a la placa ===")
    print("NOTA: Asegúrate de que la placa esté en modo Bootloader (parpadeando en verde) si tienes problemas de subida.")
    cmd_upload = f"arduino-cli upload -p {puerto} --fqbn {FQBN_FULL} \"{path_proyecto}\""
    print(f"> {cmd_upload}")
    success, _ = ejecutar(cmd_upload)
    if not success:
        sys.exit(1)
    print("✅ Proyecto subido exitosamente.")

if __name__ == "__main__":
    main()
