import os
import subprocess


def get_enabled_subdirectories_with_files(base_directory):
    subdirs_with_files = []
    for subdir in os.listdir(base_directory):
        try:
            full_path = os.path.join(base_directory, subdir)
            if os.path.isdir(full_path) and not subdir.endswith(".disabled") and not subdir.startswith('.') and subdir != '__pycache__':
                print(f"## Install dependencies for '{subdir}'")
                requirements_file = os.path.join(full_path, "requirements.txt")
                install_script = os.path.join(full_path, "install.py")

                if os.path.exists(requirements_file) or os.path.exists(install_script):
                    subdirs_with_files.append((full_path, requirements_file, install_script))
        except Exception as e:
            print(f"EXCEPTION During Dependencies INSTALL on '{subdir}':\n{e}")

    return subdirs_with_files


def install_requirements(requirements_file_path):
    if os.path.exists(requirements_file_path):
        subprocess.run(["pip", "install", "-r", requirements_file_path])


def run_install_script(install_script_path):
    if os.path.exists(install_script_path):
        subprocess.run(["python", install_script_path])


custom_nodes_directory = "custom_nodes"
subdirs_with_files = get_enabled_subdirectories_with_files(custom_nodes_directory)


for subdir, requirements_file, install_script in subdirs_with_files:
    install_requirements(requirements_file)
    run_install_script(install_script)
