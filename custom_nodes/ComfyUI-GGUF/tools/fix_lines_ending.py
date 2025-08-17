import os

files = ["lcpp.patch", "lcpp_sd3.patch"]

def has_unix_line_endings(file_path):
    try:
        with open(file_path, 'rb') as file:
            content = file.read()
        return b'\r\n' not in content
    except Exception as e:
        print(f"Error checking '{file_path}': {e}")
        return False

def convert_to_linux_format(file_path):
    try:
        with open(file_path, 'rb') as file:
            content = file.read().replace(b'\r\n', b'\n')
        with open(file_path, 'wb') as file:
            file.write(content)
        print(f"'{file_path}' converted to Linux line endings (LF).")
    except Exception as e:
        print(f"Error processing '{file_path}': {e}")

for file in files:
    if os.path.exists(file):
        if has_unix_line_endings(file):
            print(f"'{file}' already has Unix line endings (LF). No conversion needed.")
        else:
            convert_to_linux_format(file)
    else:
        print(f"File '{file}' does not exist.")
