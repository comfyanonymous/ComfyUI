
import os
import sys
import shutil

def fix_python_lib():
    print(f"Current Python Executable: {sys.executable}")
    print(f"Base Prefix: {sys.base_prefix}")
    print(f"Prefix: {sys.prefix}")
    
    # Target directory in venv
    venv_libs = os.path.join(sys.prefix, 'libs')
    if not os.path.exists(venv_libs):
        print(f"Creating missing venv libs directory: {venv_libs}")
        os.makedirs(venv_libs)
    
    # Candidate locations for python310.lib
    # 1. Base Prefix / libs (Standard Python / Conda)
    candidates = [
        os.path.join(sys.base_prefix, 'libs', 'python310.lib'),
        os.path.join(sys.base_prefix, 'libs', 'python3.lib'),
        # Common Anaconda locations if base_prefix is weird
        r"C:\Users\jando\miniconda3\libs\python310.lib",
        r"C:\Users\jando\anaconda3\libs\python310.lib",
        r"C:\ProgramData\miniconda3\libs\python310.lib",
        # Check parent of base prefix just in case
        os.path.join(os.path.dirname(sys.executable), 'libs', 'python310.lib') 
    ]

    found_lib = None
    for cand in candidates:
        if os.path.exists(cand):
            found_lib = cand
            break
            
    if not found_lib:
        print("ERROR: Could not find python310.lib in standard locations.")
        print("Searched:", candidates)
        
        # Try a recursive search in base_prefix as last resort
        print(f"Scanning {sys.base_prefix} for python310.lib...")
        for root, dirs, files in os.walk(sys.base_prefix):
            if 'python310.lib' in files:
                found_lib = os.path.join(root, 'python310.lib')
                break
    
    if found_lib:
        print(f"Found library at: {found_lib}")
        dest = os.path.join(venv_libs, 'python310.lib')
        print(f"Copying to: {dest}")
        shutil.copy2(found_lib, dest)
        print("Success! python310.lib has been copied. Triton should now be able to compile.")
    else:
        print("FATAL: Unable to locate python310.lib. Please install the Windows SDK or ensure Python development libraries are present.")

if __name__ == "__main__":
    fix_python_lib()
