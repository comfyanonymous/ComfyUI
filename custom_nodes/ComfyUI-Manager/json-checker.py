import json
import argparse

def check_json_syntax(file_path):
    try:
        with open(file_path, 'r') as file:
            json_str = file.read()
            json.loads(json_str)
            print(f"[ OK ] {file_path}")
    except json.JSONDecodeError as e:
        print(f"[FAIL] {file_path}\n\n       {e}\n")
    except FileNotFoundError:
        print(f"[FAIL] {file_path}\n\n       File not found\n")

def main():
    parser = argparse.ArgumentParser(description="JSON File Syntax Checker")
    parser.add_argument("file_path", type=str, help="Path to the JSON file for syntax checking")

    args = parser.parse_args()
    check_json_syntax(args.file_path)

if __name__ == "__main__":
    main()
