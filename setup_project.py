import os

def create_structure():
    """
    Creates a robust directory structure for a TinyML project.
    """
    base_dirs = [
        "data/raw",
        "data/processed",
        "data/augmented",
        "models/checkpoints",
        "models/tflite",
        "src",
        "deployment/arduino_project",
        "notebooks",
        "tensorboard_logs",
        "config"
    ]

    files_to_create = {
        "requirements.txt": "tensorflow\nnumpy\npandas\nmatplotlib\njupyter\n",
        "deployment/arduino_project/arduino_project.ino": "// Arduino Sketch Template for TinyML\n\nvoid setup() {\n  // Initialize serial communication\n  Serial.begin(9600);\n  while (!Serial);\n  Serial.println(\"TinyML Model Initialized\");\n}\n\nvoid loop() {\n  // Main loop\n  delay(1000);\n}\n"
    }

    print("🚀 Initializing TinyML Project Structure...")

    # Create Directories
    for directory in base_dirs:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")
        # Add .gitkeep to ensure empty folders are tracked if using git
        with open(os.path.join(directory, ".gitkeep"), "w") as f:
            pass

    # Create Files
    for file_path, content in files_to_create.items():
        # Ensure parent directory exists for file
        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else ".", exist_ok=True)
        
        with open(file_path, "w") as f:
            f.write(content)
        print(f"📄 Created file: {file_path}")

    print("\n✨ Project structure successfully created!")

if __name__ == "__main__":
    create_structure()
