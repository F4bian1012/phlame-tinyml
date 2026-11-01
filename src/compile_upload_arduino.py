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
# FQBN configured for the Portenta M7
FQBN_BASE = "arduino:mbed_portenta:envie_m7"
FQBN_FULL = f"{FQBN_BASE}:split=100_0"

# Framework firmware sketches. The board runs ONE at a time, so --target
# picks which one gets uploaded (there is no "both" option).
FIRMWARE_DIRS = {
    'pil': os.path.join('deployment', 'pil_firmware'),
    'hil': os.path.join('deployment', 'hil_camera_firmware'),
}

# Repo root, resolved from this file so as not to depend on the cwd.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def ejecutar(comando):
    """Runs a command in the terminal and returns success and its output."""
    res = subprocess.run(comando, shell=True, capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    if res.returncode == 0:
        return True, res.stdout
    else:
        print(f" Error while running:\n{comando}\nDetails:\n{res.stderr}")
        return False, res.stderr

def check_core_installed():
    """Checks whether the Portenta core is installed and, if not, installs it."""
    print("-> Checking installed cores...")
    success, output = ejecutar("arduino-cli core list --format json")
    if not success:
        print(" Could not check the list of cores. Is arduino-cli installed and on the PATH?")
        sys.exit(1)

    try:
        data = json.loads(output)
        cores = (data.get("platforms") or []) if isinstance(data, dict) else (data or [])
        for core in cores:
            if isinstance(core, dict) and core.get("id") == "arduino:mbed_portenta":
                print("Core arduino:mbed_portenta is already installed.")
                return
    except json.JSONDecodeError:
        print(" Error while parsing the JSON output of arduino-cli.")
        sys.exit(1)

    print("-> Core not found. Installing arduino:mbed_portenta (this may take a few minutes)...")
    success, _ = ejecutar("arduino-cli core install arduino:mbed_portenta")
    if not success:
        print(" Failed to install the core. Please install it manually.")
        sys.exit(1)
    print(" Core installed successfully.")

def find_portenta_port():
    """Looks for a connected Portenta board using arduino-cli and returns its COM port."""
    print("-> Looking for a connected Portenta board...")
    success, output = ejecutar("arduino-cli board list --format json")
    if not success:
        print(" Could not get the list of connected boards.")
        sys.exit(1)

    try:
        data = json.loads(output)
        boards = (data.get("detected_ports") or []) if isinstance(data, dict) else (data or [])
        for board in boards:
            if not isinstance(board, dict): continue
            matching = board.get("matching_boards") or []
            for match in matching:
                fqbn = match.get("fqbn", "")
                name = match.get("name", "").lower()

                if "portenta" in name or FQBN_BASE in fqbn:
                    port = board.get("port", {}).get("address")
                    if port:
                        print(f" Board found: {match.get('name')} on port {port}")
                        return port
    except json.JSONDecodeError:
        pass

    print(" No Portenta board was detected automatically on this computer.")
    return None

def resolver_proyecto(parser, target, path_proyecto):
    """An explicit --path_proyecto wins; otherwise --target resolves the framework sketch."""
    if path_proyecto:
        if target:
            print(" [WARN] --path_proyecto was given; --target is ignored.")
        return os.path.abspath(path_proyecto)

    if not target:
        parser.error("specify the firmware to upload with --target {pil,hil}, or a path with --path_proyecto.")

    return os.path.join(REPO_ROOT, FIRMWARE_DIRS[target])

def main():
    parser = argparse.ArgumentParser(description="Compile and upload one of the framework firmwares to the Arduino Portenta M7.")
    parser.add_argument("--target",
                        choices=["pil", "hil"],
                        default=None,
                        help="Firmware to compile and upload: 'pil' (deployment/pil_firmware, image injected over serial) or 'hil' (deployment/hil_camera_firmware, camera in the loop). The board runs one at a time, which is why there is no 'both' option.")
    parser.add_argument("--path_proyecto",
                        default=None,
                        help="Path to an arbitrary Arduino project, as an alternative to --target.")

    args = parser.parse_args()
    path_proyecto = resolver_proyecto(parser, args.target, args.path_proyecto)

    if not os.path.exists(path_proyecto):
        print(f" The specified project path does not exist: {path_proyecto}")
        sys.exit(1)

    check_core_installed()

    puerto = find_portenta_port()
    if not puerto:
        print("-> Please connect the board and try again.")
        sys.exit(1)

    print(f"\n=== 1. Compiling project at: {path_proyecto} ===")
    cmd_compile = f"arduino-cli compile --fqbn {FQBN_FULL} --warnings none \"{path_proyecto}\""
    print(f"> {cmd_compile}")
    success, _ = ejecutar(cmd_compile)
    if not success:
        sys.exit(1)
    print(" Compilation successful.")

    print("\n=== 2. Uploading to the board ===")
    print("NOTE: Make sure the board is in Bootloader mode (blinking green) if you have upload problems.")
    cmd_upload = f"arduino-cli upload -p {puerto} --fqbn {FQBN_FULL} \"{path_proyecto}\""
    print(f"> {cmd_upload}")
    success, _ = ejecutar(cmd_upload)
    if not success:
        sys.exit(1)
    print("[OK] Project uploaded successfully.")

if __name__ == "__main__":
    main()
