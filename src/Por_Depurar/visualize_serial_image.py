import serial
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
import sys

# Configuration matches Arduino sketch
WIDTH = 320
HEIGHT = 320
CHANNELS = 1 # Grayscale
FRAME_SIZE = WIDTH * HEIGHT * CHANNELS
BAUD_RATE = 115200

def get_serial_ports():
    """Lists serial port names on Mac"""
    if sys.platform.startswith('darwin'):
        import glob
        ports = glob.glob('/dev/tty.usbmodem*') + glob.glob('/dev/tty.usbserial*')
    else:
        ports = []
    return ports

def main():
    parser = argparse.ArgumentParser(description='Visualize images from Arduino Portenta H7')
    parser.add_argument('--port', default='COM9',help='Serial port (e.g., /dev/tty.usbmodem...)')
    args = parser.parse_args()

    port = args.port
    
    if not port:
        print("No port specified. Scanning for common Arduino ports...")
        available_ports = get_serial_ports()
        if not available_ports:
            print("No ports found. Please connect the device and specify --port.")
            return
        print(f"Found ports: {available_ports}")
        port = available_ports[0]
        print(f"Using: {port}")

    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=2)
        print(f"Connected to {port} at {BAUD_RATE} baud.")
    except serial.SerialException as e:
        print(f"Error connecting to port: {e}")
        return

    print("Waiting for data... (Press Ctrl+C to exit)")
    
    plt.ion() # Interactive mode on
    fig, ax = plt.subplots()
    # Initialize with random noise or black
    img_data = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    image_plot = ax.imshow(img_data, cmap='gray', vmin=0, vmax=255)
    plt.show()

    try:
        while True:
            # We expect a full frame. 
            # Note: This simple approach assumes we start reading at the beginning of a frame.
            # If out of sync, the image will look shifted.
            # The Arduino code has a 1000ms delay between frames, so we have a good chance
            # of syncing if we read faster or flush.
            
            # Use 'read' which blocks until sufficient data is received or timeout
            # If the Arduino sends strictly 76800 bytes then pauses, we try to read that chunk.
            
            # Simple sync attempt: Clear buffer if it's small (noise) or wait for a large burst?
            # For now, just read.
            
            # Reset input buffer to potentially sync with the start of a new burst if we are lagging
            # ser.reset_input_buffer() 
            # NOT resetting automatically to avoid dropping data, but user might need to restart.

            data = ser.read(FRAME_SIZE)
            
            if len(data) != FRAME_SIZE:
                if len(data) > 0:
                    print(f"Incomplete frame received: {len(data)} bytes. Waiting...")
                else:
                   # No data, just loop
                   plt.pause(0.1)
                continue

            # Convert to numpy array
            img = np.frombuffer(data, dtype=np.uint8).reshape((HEIGHT, WIDTH))

            # Update plot
            image_plot.set_data(img)
            plt.draw()
            plt.pause(0.001)
            
            print("Frame received.")

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        ser.close()
        print("Serial closed.")
        plt.close()

if __name__ == "__main__":
    main()
