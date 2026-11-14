"""
compile_upload_arduino.py
=========================
Compiles one of the framework sketches for a board and uploads it.

    python src/compile_upload_arduino.py --target pil                   # Portenta H7
    python src/compile_upload_arduino.py --target hil --board nicla     # Nicla Vision
    python src/compile_upload_arduino.py --target pil --board nicla --compile_only

The same two sketches (deployment/pil_firmware, deployment/hil_firmware) serve
every board in boards.py; --board selects the core and the FQBN, and the
sketch picks its board block at compile time.
"""
import argparse
import subprocess
import json
import sys
import os
import boards
import cli

try:
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

# Framework firmware sketches. The board runs ONE at a time, so --target
# picks which one gets uploaded (there is no "both" option).
FIRMWARE_DIRS = {
    'pil': os.path.join('deployment', 'pil_firmware'),
    'hil': os.path.join('deployment', 'hil_firmware'),
}

# Repo root, resolved from this file so as not to depend on the cwd.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run_command(command):
    """Runs a command in the terminal and returns success and its output."""
    res = subprocess.run(command, shell=True, capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    if res.returncode == 0:
        return True, res.stdout
    else:
        print(f" Error while running:\n{command}\nDetails:\n{res.stderr}")
        return False, res.stderr

def check_core_installed(core):
    """Checks whether the board's core is installed and, if not, installs it."""
    print("-> Checking installed cores...")
    success, output = run_command("arduino-cli core list --format json")
    if not success:
        print(" Could not check the list of cores. Is arduino-cli installed and on the PATH?")
        sys.exit(1)

    try:
        data = json.loads(output)
        cores = (data.get("platforms") or []) if isinstance(data, dict) else (data or [])
        for installed in cores:
            if isinstance(installed, dict) and installed.get("id") == core:
                print(f"Core {core} is already installed.")
                return
    except json.JSONDecodeError:
        print(" Error while parsing the JSON output of arduino-cli.")
        sys.exit(1)

    print(f"-> Core not found. Installing {core} (this may take a few minutes)...")
    success, _ = run_command(f"arduino-cli core install {core}")
    if not success:
        print(" Failed to install the core. Please install it manually.")
        sys.exit(1)
    print(" Core installed successfully.")

def find_board_port(board):
    """Looks for the connected board using arduino-cli and returns its serial port."""
    label = board["label"]
    fqbn_base = ":".join(board["fqbn"].split(":")[:3])
    print(f"-> Looking for a connected {label}...")
    success, output = run_command("arduino-cli board list --format json")
    if not success:
        print(" Could not get the list of connected boards.")
        sys.exit(1)

    try:
        data = json.loads(output)
        detected = (data.get("detected_ports") or []) if isinstance(data, dict) else (data or [])
        for entry in detected:
            if not isinstance(entry, dict): continue
            matching = entry.get("matching_boards") or []
            for match in matching:
                fqbn = match.get("fqbn", "")
                name = match.get("name", "").lower()

                if board["port_match"] in name or fqbn_base in fqbn:
                    port = entry.get("port", {}).get("address")
                    if port:
                        print(f" Board found: {match.get('name')} on port {port}")
                        return port
    except json.JSONDecodeError:
        pass

    print(f" No {label} was detected automatically on this computer.")
    return None

def resolve_sketch(parser, target, sketch_path):
    """An explicit --sketch_path wins; otherwise --target resolves the framework sketch."""
    if sketch_path:
        if target:
            print(" [WARN] --sketch_path was given; --target is ignored.")
        return os.path.abspath(sketch_path)

    if not target:
        parser.error("specify the firmware to upload with --target {pil,hil}, or a path with --sketch_path.")

    return os.path.join(REPO_ROOT, FIRMWARE_DIRS[target])

def main():
    parser = cli.ArgumentParser(description="Compile one of the framework firmwares and upload it to the board (Arduino Portenta H7 or Nicla Vision).")
    parser.add_argument("--target",
                        choices=["pil", "hil"],
                        default=None,
                        help="Firmware to compile and upload: 'pil' (deployment/pil_firmware, image injected over serial) or 'hil' (deployment/hil_firmware, camera in the loop). The board runs one at a time, which is why there is no 'both' option.")
    parser.add_argument("--board",
                        choices=sorted(boards.BOARDS),
                        default=boards.DEFAULT_BOARD,
                        help="Board to compile for (default: portenta).")
    parser.add_argument("--arena_kb",
                        type=int,
                        default=None,
                        help="Size of the tensor arena compiled into the sketch, in KB. Default: the sketch's own value for the board (4096 on the Portenta H7, in external SDRAM; 368 on the Nicla Vision, which only has its 512 KB of internal RAM).")
    parser.add_argument("--image_buffer_bytes",
                        type=int,
                        default=None,
                        help="Size of the image buffer compiled into the sketch, in bytes (width * height * channels of the model input). Default: the sketch's own value for the board (409600 on the Portenta H7; 19200 on the Nicla Vision, one 160x120 grayscale image).")
    parser.add_argument("--compile_only",
                        action="store_true",
                        help="Compile and report the size without uploading; no board needs to be connected.")
    parser.add_argument("--sketch_path",
                        default=None,
                        help="Path to an arbitrary Arduino sketch folder, as an alternative to --target.")
    # Former name of --sketch_path: still accepted, not listed in --help.
    parser.add_argument("--path_proyecto", dest="sketch_path", help=argparse.SUPPRESS)

    args = parser.parse_args()
    board = boards.BOARDS[args.board]
    sketch_path = resolve_sketch(parser, args.target, args.sketch_path)

    if not os.path.exists(sketch_path):
        print(f" The specified project path does not exist: {sketch_path}")
        sys.exit(1)

    check_core_installed(board["core"])

    port = None
    if not args.compile_only:
        port = find_board_port(board)
        if not port:
            print("-> Please connect the board and try again.")
            sys.exit(1)

    print(f"\n=== 1. Compiling project at: {sketch_path} ({board['label']}) ===")
    cmd_compile = f"arduino-cli compile --fqbn {board['fqbn']} --warnings none"
    # Both sizes are macros of the sketch (see its BOARD SUPPORT block).
    defines = []
    if args.arena_kb is not None:
        defines.append(f"-DTENSOR_ARENA_KB={args.arena_kb}")
    if args.image_buffer_bytes is not None:
        defines.append(f"-DIMAGE_BUFFER_BYTES={args.image_buffer_bytes}")
    if defines:
        cmd_compile += f" --build-property \"compiler.cpp.extra_flags={' '.join(defines)}\""
    cmd_compile += f" \"{sketch_path}\""
    print(f"> {cmd_compile}")
    success, output = run_command(cmd_compile)
    if not success:
        sys.exit(1)
    # Flash and RAM use: on the Nicla Vision the arena and the buffers live in
    # internal RAM, so the second line says how much is left for the stack.
    for line in output.splitlines():
        if "bytes" in line:
            print(f" {line.strip()}")
    print(" Compilation successful.")
    if args.compile_only:
        return

    print("\n=== 2. Uploading to the board ===")
    print("NOTE: Make sure the board is in Bootloader mode (blinking green) if you have upload problems.")
    cmd_upload = f"arduino-cli upload -p {port} --fqbn {board['fqbn']} \"{sketch_path}\""
    print(f"> {cmd_upload}")
    success, _ = run_command(cmd_upload)
    if not success:
        sys.exit(1)
    print("[OK] Project uploaded successfully.")

if __name__ == "__main__":
    main()
