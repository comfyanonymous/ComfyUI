# ComfyUI LTS

A vanilla, up-to-date fork of [ComfyUI](https://github.com/comfyanonymous/comfyui) intended for long term support (LTS) from [AppMana](https://appmana.com) and [Hidden Switch](https://hiddenswitch.com).

## Key Features and Differences

This LTS fork enhances vanilla ComfyUI with enterprise-grade features, focusing on stability, ease of deployment, and scalability while maintaining full compatibility.

### Deployment and Installation
- **Pip and UV Installable:** Install via `pip` or `uv` directly from GitHub. No manual cloning required for users.
- **Automatic Model Downloading:** Missing models (e.g., Stable Diffusion, FLUX, LLMs) are downloaded on-demand from Hugging Face or CivitAI.
- **Docker and Containers:** First-class support for Docker and Kubernetes with optimized containers for NVIDIA and AMD.

### Scalability and Performance
- **Distributed Inference:** Run scalable inference clusters with multiple workers and frontends using RabbitMQ.
- **Embedded Mode:** Use ComfyUI as a Python library (`import comfy`) inside your own applications without the web server.
- **LTS Custom Nodes:** A curated set of "Installable" custom nodes (ControlNet, AnimateDiff, IPAdapter) optimized for this fork.

### Enhanced Capabilities
- **LLM Support:** Native support for Large Language Models (LLaMA, Phi-3, etc.) and multi-modal workflows.
- **API and Configuration:** Enhanced API endpoints and extensive configuration options via CLI args, env vars, and config files.
- **Tests:** Automated test suite ensuring stability for new features.

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

### Core
- [Installation & Getting Started](docs/installing.md)
- [Hardware Compatibility](docs/compatibility.md)
- [Configuration](docs/configuration.md)
- [Troubleshooting](docs/troubleshooting.md)

### Features & Workflows
- [Large Language Models](docs/llm.md)
- [Video Workflows](docs/video.md) (AnimateDiff, SageAttention, etc.)
- [Other Features](docs/other_features.md) (SVG, Ideogram)

### Extending ComfyUI
- [Custom Nodes](docs/custom_nodes.md) (Installing & Authoring)
- [API Usage](docs/api.md) (Python, REST, Embedded)

### Deployment
- [Distributed / Multi-GPU](docs/distributed.md)
- [Docker & Containers](docs/docker.md)
