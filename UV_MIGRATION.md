# UV Migration Guide for ComfyUI

This document provides instructions for migrating to UV-based package management for ComfyUI.

## What is UV?

[UV](https://github.com/astral-sh/uv) is a modern, ultra-fast package manager for Python. It's up to 10-100x faster than pip and provides better dependency resolution, lockfile support, and caching. UV is designed to be a drop-in replacement for pip, but with better performance and reliability.

## Installation Options

### Option 1: Install ComfyUI as a UV Package (Recommended)

This new approach treats ComfyUI as a proper Python package that can be installed using UV:

```bash
# Install UV if you don't have it already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv .venv
source .venv/bin/activate  # On Linux/macOS
# OR
.venv\Scripts\activate     # On Windows

# Install ComfyUI with all optional dependencies
uv pip install "comfyui[gpu,advanced]"

# Or install from a specific version:
uv pip install "comfyui==0.3.30[gpu,advanced]"
```

After installation, you can run ComfyUI simply by typing:

```bash
comfyui
```

### Option 2: Use the Automated Setup Script

If you have the ComfyUI source code, you can use our new setup script to handle everything:

```bash
# Clone the repository
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# Run the setup script
python uv_setup.py --gpu --advanced
```

### Option 3: Traditional Approach (Use UV as pip replacement)

If you prefer the traditional approach but want to use UV:

```bash
# Clone the repository
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# Install UV if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv .venv
source .venv/bin/activate  # On Linux/macOS
# OR
.venv\Scripts\activate     # On Windows

# Install dependencies
uv pip install -r requirements.txt
```

## Available Extras

When installing ComfyUI as a package, you can specify extras to include additional dependencies:

- `gpu`: NVIDIA CUDA libraries for GPU acceleration
- `advanced`: Additional dependencies for advanced features
- `dev`: Development dependencies for contributing to ComfyUI

Example:
```bash
uv pip install "comfyui[gpu,advanced,dev]"
```

## Command-Line Options

When installed as a package, you can use the `comfyui` command with various options:

```bash
comfyui --host 0.0.0.0 --port 8188 --auto-launch
```

Common options:
- `--host`: The IP address to listen on (default: 127.0.0.1)
- `--port`: The port to listen on (default: 8188)
- `--auto-launch`: Automatically open ComfyUI in your default browser
- `--cuda-device`: Specify which CUDA device to use
- `--output-directory`: Override the default output directory
- `--input-directory`: Override the default input directory
- `--verbose`: Enable verbose logging

## Migration Command Reference

Below is a quick reference for migrating from pip commands to UV:

| pip command | UV command | Description |
|-------------|------------|-------------|
| `pip install -r requirements.txt` | `uv pip install -r requirements.txt` | Install from requirements file |
| `pip install package` | `uv pip install package` or `uv add package` | Install a package |
| `pip install -e .` | `uv pip install -e .` | Install current directory in development mode |
| `pip freeze > requirements.txt` | `uv pip freeze > requirements.txt` | Create requirements file from installed packages |

## Benefits of UV Package Approach

Installing ComfyUI as a UV package offers several advantages:

1. **Simplified Installation**: One command to install everything
2. **Dependency Management**: Faster resolution and better handling of complex dependencies
3. **Reproducible Environments**: Lock files ensure consistent environments across systems
4. **Command-Line Interface**: Run ComfyUI from anywhere using the `comfyui` command
5. **Optional Dependencies**: Install only what you need via extras
6. **Package Updates**: Easily update to new versions with `uv pip install -U comfyui`
7. **Development Mode**: Better integration with development workflows

## Troubleshooting

- **Missing packages?** Try `uv pip install --upgrade -r requirements.txt` to reinstall all dependencies.
- **Package conflicts?** UV has improved dependency resolution, but if issues persist, try `uv pip install package --force-reinstall`.
- **Need to start fresh?** Run `python clean_venv.py` to remove your virtual environment and start over.

For additional help, please visit our [Discord](https://comfy.org/discord) or [GitHub repository](https://github.com/comfyanonymous/ComfyUI).
