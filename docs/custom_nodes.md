# Custom Nodes

Custom Nodes can be added to ComfyUI by copying and pasting Python files into your `./custom_nodes` directory.

## Installing Custom Nodes

There are two kinds of custom nodes: vanilla custom nodes, which generally expect to be dropped into the `custom_nodes` directory and managed by a tool called the ComfyUI Extension manager ("vanilla" custom nodes) and this repository's opinionated, installable custom nodes ("installable").

### Installing ComfyUI Manager

ComfyUI-Manager is a popular extension to help you install and manage other custom nodes. To install it, you will need `git` on your system.

#### Manual Install

The installation process for ComfyUI-Manager requires two steps: installing its Python dependencies, and then cloning its code into the `custom_nodes` directory.

1.  **Install dependencies.**
    First, ensure you have installed `comfyui` from this repository as described in the Installing section. Then, run the following command from your ComfyUI workspace directory (the one containing your `.venv` folder) to install the extra dependencies for ComfyUI-Manager:

    ```shell
    uv pip install --torch-backend=auto --upgrade "comfyui[comfyui_manager]@git+https://github.com/hiddenswitch/ComfyUI.git"
    ```

2.  **Clone the repository.**
    Next, you need to clone the ComfyUI-Manager repository into the `custom_nodes` directory within your ComfyUI workspace. Your workspace is the directory you created during the initial setup where you ran `uv venv` (e.g., `~/Documents/ComfyUI_Workspace`).

    If the `custom_nodes` directory does not exist in your workspace, create it first (e.g., `mkdir custom_nodes`). Then, from your workspace directory, run the following command:

    ```shell
    git clone https://github.com/Comfy-Org/ComfyUI-Manager.git ./custom_nodes/ComfyUI-Manager
    ```
    This command will place the manager's code into `custom_nodes/ComfyUI-Manager/`.

3.  **Restart ComfyUI.**
    After the cloning is complete, restart ComfyUI. You should now see a "Manager" button in the menu.

### PyPi Install

[ComfyUI-Manager](https://github.com/Comfy-Org/ComfyUI-Manager/tree/manager-v4)

**ComfyUI-Manager** is an extension that allows you to easily install, update, and manage custom nodes for ComfyUI.

### Setup

1. Install the manager dependencies:
   ```bash
   pip install -r manager_requirements.txt
   ```

2. Enable the manager with the `--enable-manager` flag when running ComfyUI:
   ```bash
   python main.py --enable-manager
   ```

### Command Line Options

| Flag | Description |
|------|-------------|
| `--enable-manager` | Enable ComfyUI-Manager |
| `--enable-manager-legacy-ui` | Use the legacy manager UI instead of the new UI (requires `--enable-manager`) |
| `--disable-manager-ui` | Disable the manager UI and endpoints while keeping background features like security checks and scheduled installation completion (requires `--enable-manager`) |



### Vanilla Custom Nodes

Clone the repository containing the custom nodes into `custom_nodes/` in your working directory and install its requirements, or use the manager.

### Custom Nodes Authored for this Fork

Run `uv pip install "git+https://github.com/owner/repository"`, replacing the `git` repository with the installable custom nodes URL. This is just the GitHub URL.

## Authoring Custom Nodes

These instructions will allow you to quickly author installable custom nodes.

#### Using `pyproject.toml` for projects with existing `requirements.txt`

Suppose your custom nodes called `my_comfyui_nodes` has a folder layout that looks like this:

```
__init__.py
some_python_file.py
requirements.txt
LICENSE.txt
some_directory/some_code.py
```

First, add an `__init__.py` to `some_directory`, so that it is a Python package:

```
__init__.py
some_python_file.py
requirements.txt
LICENSE.txt
some_directory/__init__.py
some_directory/some_code.py
```

Then, if your `NODE_CLASS_MAPPINGS` are declared in `__init__.py`, use the following as a `pyproject.toml`, substituting your actual project name:

**pyproject.toml**

```toml
[project]
name = "my_comfyui_nodes"
description = "My nodes description."
version = "1.0.0"
license = { file = "LICENSE.txt" }
dynamic = ["dependencies"]

[project.urls]
Repository = "https://github.com/your-github-username/my-comfyui-nodes"
#  Used by Comfy Registry https://comfyregistry.org

[tool.comfy]
PublisherId = "your-github-username"
DisplayName = "my_comfyui_nodes"
Icon = ""

[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
packages = ["my_comfyui_nodes", "my_comfyui_nodes.some_directory"]
package-dir = { "my_comfyui_nodes" = ".", "my_comfyui_nodes.some_directory" = "some_directory" }

[tool.setuptools.dynamic]
dependencies = { file = ["requirements.txt"] }

[project.entry-points."comfyui.custom_nodes"]
my_comfyui_nodes = "my_comfyui_nodes"
```

Observe that the directory should now be listed as a package in the `packages` and `package-dir` statement.

#### Using `setup.py`

Create a `requirements.txt`:

```
comfyui
```

Observe `comfyui` is now a requirement for using your custom nodes. This will ensure you will be able to access `comfyui` as a library. For example, your code will now be able to import the folder paths using `from comfyui.cmd import folder_paths`. Because you will be using my fork, use this:

```
comfyui @ git+https://github.com/hiddenswitch/ComfyUI.git
```

Additionally, create a `pyproject.toml`:

```
[build-system]
requires = ["setuptools", "wheel", "pip"]
build-backend = "setuptools.build_meta"
```

This ensures you will be compatible with later versions of Python.

Finally, move your nodes to a directory with an empty `__init__.py`, i.e., a package. You should have a file structure like this:

```
# the root of your git repository
/.git
/pyproject.toml
/requirements.txt
/mypackage_custom_nodes/__init__.py
/mypackage_custom_nodes/some_nodes.py
```

Finally, create a `setup.py` at the root of your custom nodes package / repository. Here is an example:

**setup.py**

```python
from setuptools import setup, find_packages
import os.path

setup(
    name="mypackage",
    version="0.0.1",
    packages=find_packages(),
    install_requires=open(os.path.join(os.path.dirname(__file__), "requirements.txt")).readlines(),
    author='',
    author_email='',
    description='',
    entry_points={
        'comfyui.custom_nodes': [
            'mypackage = mypackage_custom_nodes',
        ],
    },
)
```

All `.py` files located in the package specified by the entrypoint with your package's name will be scanned for node class mappings declared like this:

**some_nodes.py**:

```py
from comfy.nodes.package_typing import CustomNode


class Binary_Preprocessor(CustomNode):
    ...


NODE_CLASS_MAPPINGS = {
    "BinaryPreprocessor": Binary_Preprocessor
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BinaryPreprocessor": "Binary Lines"
}
```

These packages will be scanned recursively.

Extending the `comfy.nodes.package_typing.CustomNode` provides type hints for authoring nodes.

## Adding Custom Configuration

Declare an entry point for configuration hooks in your **setup.py** that defines a function that takes and returns an
`configargparser.ArgParser` object:

**setup.py**

```python
setup(
    name="mypackage",
    ...
entry_points = {
    'comfyui.custom_nodes': [
        'mypackage = mypackage_custom_nodes',
    ],
    'comfyui.custom_config': [
        'mypackage = mypackage_custom_config:add_configuration',
    ]
},
)
```

**mypackage_custom_config.py**:

```python
import configargparse


def add_configuration(parser: configargparse.ArgParser) -> configargparse.ArgParser:
    parser.add_argument("--openai-api-key",
                        required=False,
                        type=str,
                        help="Configures the OpenAI API Key for the OpenAI nodes", env_var="OPENAI_API_KEY")
    return parser

```

You can now see your configuration option at the bottom of the `--help` command along with hints for how to use it:

```shell
$ comfyui --help
usage: comfyui.exe [-h] [-c CONFIG_FILE] [--write-out-config-file CONFIG_OUTPUT_PATH] [-w CWD] [-H [IP]] [--port PORT]
                   [--enable-cors-header [ORIGIN]] [--max-upload-size MAX_UPLOAD_SIZE] [--extra-model-paths-config PATH [PATH ...]]
...
                   [--openai-api-key OPENAI_API_KEY]

options:
  -h, --help            show this help message and exit
  -c CONFIG_FILE, --config CONFIG_FILE
                        config file path
  --write-out-config-file CONFIG_OUTPUT_PATH
                        takes the current command line args and writes them out to a config file at the given path, then exits
  -w CWD, --cwd CWD     Specify the working directory. If not set, this is the current working directory. models/, input/, output/ and other
                        directories will be located here by default. [env var: COMFYUI_CWD]
  -H [IP], --listen [IP]
                        Specify the IP address to listen on (default: 127.0.0.1). If --listen is provided without an argument, it defaults to
                        0.0.0.0. (listens on all) [env var: COMFYUI_LISTEN]
  --port PORT           Set the listen port. [env var: COMFYUI_PORT]
...
  --distributed-queue-name DISTRIBUTED_QUEUE_NAME
                        This name will be used by the frontends and workers to exchange prompt requests and replies. Progress updates will be
                        prefixed by the queue name, followed by a '.', then the user ID [env var: COMFYUI_DISTRIBUTED_QUEUE_NAME]
  --external-address EXTERNAL_ADDRESS
                        Specifies a base URL for external addresses reported by the API, such as for image paths. [env var:
                        COMFYUI_EXTERNAL_ADDRESS]
  --openai-api-key OPENAI_API_KEY
                        Configures the OpenAI API Key for the OpenAI nodes [env var: OPENAI_API_KEY]
```

You can now start `comfyui` with:

```shell
uv run comfyui --openai-api-key=abcdefg12345
```

or set the environment variable you specified:

```shell
export OPENAI_API_KEY=abcdefg12345
uv run comfyui
```

or add it to your config file:

**config.yaml**:

```txt
openapi-api-key: abcdefg12345
```

```shell
comfyui --config config.yaml
```

Since `comfyui` looks for a `config.yaml` in your current working directory by default, you can omit the argument if
`config.yaml` is located in your current working directory:

```shell
uv run comfyui
```

Your entry point for adding configuration options should **not** import your nodes. This gives you the opportunity to
use the configuration you added in your nodes; otherwise, if you imported your nodes in your configuration entry point,
the nodes will potentially be initialized without any configuration.

Access your configuration from `cli_args`:

```python
from comfy.cli_args import args
from comfy.cli_args_types import Configuration
from typing import Optional


# Add type hints when accessing args
class CustomConfiguration(Configuration):
    def __init__(self):
        super().__init__()
        self.openai_api_key: Optional[str] = None


args: CustomConfiguration


class OpenAINode(CustomNode):
    ...

    def execute(self):
        openai_api_key = args.open_api_key
```
