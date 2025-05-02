# ComfyUI Migration to UV Package Manager

## What is UV?

[UV](https://github.com/astral-sh/uv) is a modern Python package installer and resolver written in Rust. It serves as a drop-in replacement for pip with significant performance improvements:

- **Speed**: UV installs packages 10-100x faster than pip
- **Reliability**: Better dependency resolution to avoid conflicts
- **Compatibility**: Works with existing requirements.txt files
- **Modern**: Built with Rust for performance and safety

## Why Migrate from pip to UV?

- Faster installation of dependencies
- Improved virtual environment management
- Better handling of dependency conflicts
- Consistent installation experience across platforms
- Compatibility with existing pip workflows

## Installation

Install UV:

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH (bash/zsh)
source $HOME/.local/bin/env

# Or for fish shell
# source $HOME/.local/bin/env.fish
```

## Using UV with ComfyUI

### Basic Usage

```bash
# Create a virtual environment
uv venv
source .venv/bin/activate  # On Linux/macOS
# .venv\Scripts\activate   # On Windows

# Install dependencies
uv pip install -r requirements.txt

# Install advanced dependencies (if needed)
uv pip install -r requirements_advanced.txt
```

### UV Native Commands

Instead of using the pip-compatible interface, you can use UV's native commands:

```bash
# Install packages
uv add packagename

# Install dev dependencies
uv add --dev pytest black

# View dependency tree
uv tree

# Update dependencies
uv sync
```

## Migration Tips

1. UV is designed to be a drop-in replacement for pip, so most commands work similarly
2. Existing requirements.txt files are fully compatible
3. For best results, start with a fresh virtual environment
4. UV includes built-in lockfile support for reproducible environments

## Troubleshooting

If you encounter issues:

- Ensure you have the latest version of UV: `uv self update`
- Try running with verbose output: `uv -vvv pip install -r requirements.txt`
- Check the [UV documentation](https://github.com/astral-sh/uv) for known issues
