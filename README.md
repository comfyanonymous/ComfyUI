# ComfyUI LTS

A vanilla, up-to-date fork of [ComfyUI](https://github.com/comfyanonymous/comfyui) intended for long term support (LTS) from [AppMana](https://appmana.com) and [Hidden Switch](https://hiddenswitch.com).

## Quickstart (Linux)

### UI Users

For users who want to run ComfyUI for generating images and videos.

1.  **Install `uv`**:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Create a Workspace**:
    ```bash
    mkdir comfyui-workspace
    cd comfyui-workspace
    ```

3.  **Install and Run**:
    ```bash
    # Create a virtual environment
    uv venv --python 3.12
    
    # Install ComfyUI LTS
    uv pip install --torch-backend=auto "comfyui@git+https://github.com/hiddenswitch/ComfyUI.git"
    
    # Run
    uv run comfyui
    ```

### Developers

For developers contributing to the codebase or building on top of it.

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/hiddenswitch/ComfyUI.git
    cd ComfyUI
    ```

2.  **Setup Environment**:
    ```bash
    # Create virtual environment
    uv venv --python 3.12
    source .venv/bin/activate
    
    # Install in editable mode with dev dependencies
    uv pip install -e .[dev]
    ```

3.  **Run**:
    ```bash
    uv run comfyui
    ```

## Documentation

Full documentation is available in [docs/index.md](docs/index.md).

### Table of Contents

- [New Features Compared to Upstream](docs/index.md#new-features-compared-to-upstream)
- [Getting Started](docs/index.md#getting-started)
    - [Installing](docs/index.md#installing)
    - [Model Downloading](docs/index.md#model-downloading)
- [LTS Custom Nodes](docs/index.md#lts-custom-nodes)
- [Large Language Models](docs/index.md#large-language-models)
- [Video Workflows](docs/index.md#video-workflows)
- [Custom Nodes](docs/index.md#custom-nodes)
- [Configuration](docs/index.md#configuration)
- [API Usage](docs/index.md#using-comfyui-as-an-api--programmatically)
- [Distributed / Multi-GPU](docs/index.md#distributed-multi-process-and-multi-gpu-comfy)
- [Docker Compose](docs/index.md#docker-compose)