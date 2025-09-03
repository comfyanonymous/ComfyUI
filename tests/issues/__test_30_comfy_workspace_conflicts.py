import pytest
import subprocess
import sys
import shutil
import time
from pathlib import Path

# Timeout in seconds for the server to start and print the expected lines.
SERVER_START_TIMEOUT = 20
# Repository URLs and references
COMFYUI_REPO = "https://github.com/comfyanonymous/ComfyUI.git"
COMFYSTREAM_REPO = "https://github.com/doctorpangloss/comfystream.git"
COMFYSTREAM_COMMIT = "f2f7929def53a4853cc5a1c2774aea70775ce2ff"
COMFYUI_LTS_REPO = "https://github.com/hiddenswitch/ComfyUI.git"
COMFYUI_LTS_COMMIT = "75e39c27202c8e31f8ec84eea4fc560c4e34f2c8"


def run_command(cmd, cwd, desc, shell=False):
    """Helper function to run a command and raise an error if it fails."""
    print(f"\n--- {desc} ---")
    log_cmd = cmd if isinstance(cmd, str) else ' '.join(map(str, cmd))
    print(f"Running command: {log_cmd}")
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,  # We check manually to provide better error logs
        shell=shell
    )
    if result.returncode != 0:
        print("--- STDOUT ---")
        print(result.stdout)
        print("--- STDERR ---")
        print(result.stderr)
        pytest.fail(f"Command failed: {desc}", pytrace=False)
    print(f"--- Success: {desc} ---")
    return result


@pytest.fixture(scope="module")
def comfyui_workspace(tmp_path_factory):
    """
    A pytest fixture that sets up the entire ComfyUI workspace in a temporary
    directory for a single test module run.
    """
    workspace_root = tmp_path_factory.mktemp("comfyui_ws")
    comfyui_dir = workspace_root / "ComfyUI"

    # 1. Clone the ComfyUI repository
    run_command(["git", "clone", COMFYUI_REPO, str(comfyui_dir)], cwd=workspace_root, desc="Cloning ComfyUI")

    # 2. Set up the virtual environment using system uv
    run_command(["uv", "venv"], cwd=comfyui_dir, desc="Creating virtual environment with uv")
    venv_python = comfyui_dir / ".venv" / ("Scripts" if sys.platform == "win32" else "bin") / "python"

    # Determine activation command and construct shell commands
    if sys.platform == "win32":
        # On Windows, we 'call' the activate.bat script. Quotes handle spaces in path.
        activate_cmd = f'call "{comfyui_dir / ".venv" / "Scripts" / "activate.bat"}"'
    else:
        # On Unix-like systems, we '.' (source) the activate script for POSIX compliance.
        activate_cmd = f'. "{comfyui_dir / ".venv" / "bin" / "activate"}"'

    uv_pip_install_base = "uv pip install --torch-backend=auto"

    # 3. Install base requirements
    install_reqs_cmd = f"{activate_cmd} && {uv_pip_install_base} -r requirements.txt"
    run_command(install_reqs_cmd, cwd=comfyui_dir, desc="Installing requirements.txt", shell=True)

    # 4. Install comfystream with the specific comfyui override
    overrides_content = f"comfyui@git+{COMFYUI_LTS_REPO}@{COMFYUI_LTS_COMMIT}"
    overrides_file = workspace_root / "overrides.txt"
    overrides_file.write_text(overrides_content)

    # Using an absolute path for the overrides file is safest for shell execution.
    install_comfystream_cmd_str = (
        f'{activate_cmd} && {uv_pip_install_base} '
        f'git+{COMFYSTREAM_REPO}@{COMFYSTREAM_COMMIT} '
        f'--overrides={overrides_file.resolve()}'
    )
    run_command(install_comfystream_cmd_str, cwd=comfyui_dir, desc="Installing comfystream with overrides", shell=True)

    # 5. Additionally clone comfystream into the custom_nodes directory and check out the specific commit
    custom_nodes_dir = comfyui_dir / "custom_nodes"
    comfystream_custom_node_dir = custom_nodes_dir / "comfystream"
    custom_nodes_dir.mkdir(exist_ok=True)
    run_command(
        ["git", "clone", COMFYSTREAM_REPO, str(comfystream_custom_node_dir)],
        cwd=comfyui_dir,
        desc="Cloning comfystream into custom_nodes"
    )
    run_command(
        ["git", "checkout", COMFYSTREAM_COMMIT],
        cwd=comfystream_custom_node_dir,
        desc=f"Checking out comfystream commit {COMFYSTREAM_COMMIT[:7]}"
    )


    # Yield the necessary paths to the test function
    yield venv_python, comfyui_dir

    # Teardown is handled automatically by pytest's tmp_path_factory


def test_server_starts_with_comfystream(comfyui_workspace):
    """
    Tests if the ComfyUI server starts correctly with the comfystream package
    and prints the expected initialization messages.
    """
    venv_python, comfyui_dir = comfyui_workspace

    # --- Assert that the installed package from overrides is structured correctly ---
    site_packages_cmd = [
        str(venv_python),
        "-c",
        "import sysconfig; print(sysconfig.get_paths()['purelib'])"
    ]
    site_packages_result = run_command(site_packages_cmd, cwd=comfyui_dir, desc="Finding site-packages directory")
    site_packages_path = Path(site_packages_result.stdout.strip())
    comfy_api_init_path = site_packages_path / "comfy" / "api" / "__init__.py"

    assert comfy_api_init_path.is_file(), (
        f"The installed comfyui package is missing the api module. "
        f"Expected to find: {comfy_api_init_path}"
    )
    print(f"--- Success: Found {comfy_api_init_path} ---")

    main_py_path = comfyui_dir / "main.py"

    # Start the server as a background process
    process = None
    try:
        print("\n--- Starting ComfyUI Server ---")
        process = subprocess.Popen(
            [str(venv_python), str(main_py_path)],
            cwd=comfyui_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # Redirect stderr to stdout
            text=True,
            bufsize=1 # Line-buffered
        )

        output_lines = []
        found_gui_msg = False
        found_server_msg = False

        start_time = time.time()

        # Read output line-by-line with a timeout
        while time.time() - start_time < SERVER_START_TIMEOUT:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                pytest.fail("Server process terminated unexpectedly.")

            if line:
                print(line, end='')
                output_lines.append(line)
                if "Initializing LocalComfyStreamServer" in line:
                    found_server_msg = True
                if "To see the GUI go to: http://127.0.0.1:8188" in line:
                    found_gui_msg = True
                    if not found_server_msg:
                        pytest.fail(
                            "GUI message appeared before LocalComfyStreamServer was initialized. "
                            "This indicates a custom node loading problem."
                        )

            if found_gui_msg and found_server_msg:
                break

        # Final assertions
        assert found_gui_msg, "GUI startup message was not found in the output."
        assert found_server_msg, "LocalComfyStreamServer initialization message was not found."

    finally:
        if process:
            print("\n--- Terminating ComfyUI Server ---")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("Server did not terminate gracefully, killing.")
                process.kill()

            # Print any remaining output for debugging
            remaining_output = process.stdout.read()
            if remaining_output:
                print("\n--- Remaining Server Output ---")
                print(remaining_output)

        # --- Assert that the main .gitignore was not modified ---
        print("\n--- Verifying .gitignore integrity ---")
        gitignore_path = comfyui_dir / ".gitignore"
        assert gitignore_path.is_file(), ".gitignore file not found in ComfyUI directory."

        with open(gitignore_path, 'r') as f:
            gitignore_content = f.read()

        assert "__init__.py" not in gitignore_content, (
            "Found '__init__.py' in the main .gitignore file. "
            "The patcher should use .git/info/exclude instead."
        )
        print("--- Success: .gitignore was not modified. ---")

        # --- Assert that no __init__.py files are untracked by Git ---
        print("\n--- Verifying Git status for untracked files ---")
        git_status_result = run_command(
            ["git", "status", "--porcelain"],
            cwd=comfyui_dir,
            desc="Checking git status for untracked files"
        )

        untracked_inits = []
        for line in git_status_result.stdout.strip().splitlines():
            # Untracked files are prefixed with '??'
            if line.startswith("??") and line.endswith("__init__.py"):
                untracked_inits.append(line.strip().split("?? ")[1])

        assert not untracked_inits, (
            "Found untracked __init__.py files. The patcher failed to add them to .git/info/exclude.\n"
            f"Untracked files: {untracked_inits}"
        )
        print("--- Success: No untracked __init__.py files found. ---")

