import os
import sys
import subprocess
import webbrowser
from datetime import datetime
import subprocess
import shutil

try:
    import inquirer  # You might need to install this package
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'inquirer'])
    import inquirer

BASE_DIR = "web"
BUILD_DIR = "web/dist"
SRC_DIR = "web/src"

# Determines if we need to rebuild the project based on the last modified source file date
def is_build_up_to_date():
    if not os.path.exists(BUILD_DIR):
        return False

    src_last_modified = max(os.path.getmtime(root) for root, _, _ in os.walk(SRC_DIR))
    build_last_modified = max(os.path.getmtime(root) for root, _, _ in os.walk(BUILD_DIR))
    return src_last_modified <= build_last_modified

def find_package_manager():
    """Check for available package managers and return the first one found."""
    for manager in ['yarn', 'npm', 'pnpm']:
        if shutil.which(manager):
            return manager
    raise RuntimeError("No suitable JavaScript package manager found. Please install yarn, npm, or pnpm so we can build the client.")

def install_dependencies():
    manager = find_package_manager()
    print(f"Installing dependencies with {manager}...")
    subprocess.run([manager, "install"], check=True, cwd='web', shell=True)

def build_project():
    manager = find_package_manager()
    print(f"Building project with {manager}...")
    if manager == 'yarn':
        subprocess.run([manager, "build"], check=True, cwd=BASE_DIR, shell=True)
    elif manager == 'npm':
        subprocess.run([manager, "run", "build"], check=True, cwd=BASE_DIR, shell=True)
    elif manager == 'pnpm':
        subprocess.run([manager, "run", "build"], check=True, cwd=BASE_DIR, shell=True)

def start_local_server():
    print("Starting local server...")
    subprocess.run(["python", "main.py"], check=True)

def handle_remote_server():
    print("Redirecting to void.tech for login...")
    webbrowser.open("https://void.tech/login")
    # TO DO: handle token login logic here

def main():
    if not is_build_up_to_date():
        install_dependencies()
        build_project()

    questions = [
        inquirer.List('server',
                      message="Do you want to use a local or remote server?",
                      choices=['local', 'remote'],
                      ),
    ]
    answers = inquirer.prompt(questions)

    if answers['server'] == 'local':
        start_local_server()
    else:
        handle_remote_server()

if __name__ == "__main__":
    main()
