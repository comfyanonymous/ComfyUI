import ctypes
import os
import sys

def main():
    dll_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'zluda', 'nvcuda.dll'))

    if not os.path.isfile(dll_path):
        print(f"ERROR: DLL not found: {dll_path}")
        sys.exit(1)

    try:
        zluda_dll = ctypes.CDLL(dll_path)
    except Exception as e:
        print(f"ERROR: Could not load DLL: {e}")
        sys.exit(1)

    try:
        zluda_get_nightly_flag = zluda_dll.zluda_get_nightly_flag
        zluda_get_nightly_flag.restype = ctypes.c_int

        flag = zluda_get_nightly_flag()

        if flag == 1:
            print("[nightly build]")
        elif flag == 0:
            print("[release build]")
        else:
            print(f"Unexpected flag value: {flag}")

    except Exception as e:
        print(f"ERROR: Could not call zluda_get_nightly_flag: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
