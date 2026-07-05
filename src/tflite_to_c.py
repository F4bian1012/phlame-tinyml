import argparse
import os

def hex_to_c_array(hex_data, var_name):
    data_len = len(hex_data)
    
    # Alignment attribute for better performance on embedded devices like Portenta H7
    c_str =  f"#ifndef {var_name.upper()}_H\n"
    c_str += f"#define {var_name.upper()}_H\n\n"
    c_str += "// Auto-generated from .tflite model\n"
    c_str += "#ifdef __has_attribute\n"
    c_str += "#define HAVE_ATTRIBUTE(x) __has_attribute(x)\n"
    c_str += "#else\n"
    c_str += "#define HAVE_ATTRIBUTE(x) 0\n"
    c_str += "#endif\n"
    c_str += "#if HAVE_ATTRIBUTE(aligned) || (defined(__GNUC__) && !defined(__clang__))\n"
    c_str += "#define DATA_ALIGN_ATTRIBUTE __attribute__((aligned(16)))\n"
    c_str += "#else\n"
    c_str += "#define DATA_ALIGN_ATTRIBUTE\n"
    c_str += "#endif\n\n"
    c_str += "#ifdef __cplusplus\n"
    c_str += "extern \"C\" {\n"
    c_str += "#endif\n\n"
    
    c_str += f"extern const unsigned int {var_name}_len;\n"
    c_str += f"extern const unsigned char {var_name}[];\n\n"
    
    c_str += f"const unsigned int {var_name}_len = {data_len};\n\n"
    c_str += f"const unsigned char {var_name}[] DATA_ALIGN_ATTRIBUTE = {{\n"
    
    hex_array = [f'0x{val:02x}' for val in hex_data]
    for i in range(0, len(hex_array), 12):
        chunk = hex_array[i:i+12]
        c_str += '  ' + ', '.join(chunk) + ',\n'
        
    c_str += "};\n\n"
    c_str += "#ifdef __cplusplus\n"
    c_str += "}\n"
    c_str += "#endif\n\n"
    c_str += f"#endif // {var_name.upper()}_H\n"
    return c_str

def main():
    parser = argparse.ArgumentParser(description='Convert a .tflite model to a C array compatible with TFLite for Microcontrollers (e.g., Arduino Portenta H7).')
    parser.add_argument('tflite_path', type=str, help='Path to the input .tflite model file.')
    parser.add_argument('output_path', type=str, help='Path to the output .cpp or .h file where the C array will be saved.')
    parser.add_argument('--var_name', type=str, default='model_tflite', help='Name of the C variable. Default is "model_tflite".')

    args = parser.parse_args()

    if not os.path.exists(args.tflite_path):
        print(f"Error: The model file {args.tflite_path} does not exist.")
        return

    with open(args.tflite_path, 'rb') as f:
        tflite_content = f.read()

    c_content = hex_to_c_array(tflite_content, args.var_name)

    output_dir = os.path.dirname(os.path.abspath(args.output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_path, 'w') as f:
        f.write(c_content)

    print(f"[SUCCESS] Model converted beautifully!")
    print(f" - Original file: {args.tflite_path}")
    print(f" - Output file: {args.output_path}")
    print(f" - Variable name: {args.var_name}")
    print(f" - Model size: {len(tflite_content)} bytes")

if __name__ == '__main__':
    main()
