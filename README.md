<div align="center">

# ComfyUI
**A powerful and modular stable diffusion GUI with a graph/nodes interface.**

![ComfyUI Screenshot](https://comfyanonymous.github.io/ComfyUI_examples/comfyui_screenshot.png)
</div>

## Quick Start

### Option 1: Install as a Package (New!)

```bash
# Install ComfyUI with GPU support
uv pip install "comfyui[gpu]"

# Run ComfyUI
comfyui
```

### Option 2: From Source

```bash
# Clone the repository
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# Set up with UV (recommended)
./uv_setup.sh --gpu

# Or on Windows:
python uv_setup.py --gpu

# Run ComfyUI
python main.py
```

## Main Features

- Nodes/graph/flowchart-based UI
- Highly optimized for better performance and VRAM usage
- Advanced prompt system with wildcards and more
- Create and share workflows as JSON files
- Supports multiple model backends
- Powerful API for external integration
- Extensible with custom nodes

## Requirements

- Python 3.9+ (Python 3.11 recommended)
- GPU with at least 4GB VRAM (8GB+ recommended)
- [Optional] CUDA-compatible NVIDIA GPU for faster processing

## Documentation

- [Installation Guide](docs/installation.md)
- [User Guide](docs/user-guide.md)
- [API Documentation](docs/api.md)
- [Custom Nodes Guide](docs/custom-nodes.md)
- [UV Migration Guide](UV_MIGRATION.md)

## Contributing

Please read our [Contributing Guidelines](CONTRIBUTING.md) before submitting pull requests.

## License

ComfyUI is licensed under the GNU General Public License v3.0.

## Community

- [Discord Server](https://comfy.org/discord)
- [GitHub Issues](https://github.com/comfyanonymous/ComfyUI/issues)
- [Examples Repository](https://github.com/comfyanonymous/ComfyUI_examples)
