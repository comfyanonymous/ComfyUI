# `cm-cli`: ComfyUI-Manager CLI

`cm-cli` is a tool that allows you to use various functions of ComfyUI-Manager from the command line without launching ComfyUI.


```
-= ComfyUI-Manager CLI (V2.24) =-


python cm-cli.py [OPTIONS]

OPTIONS:
    [install|reinstall|uninstall|update|disable|enable|fix] node_name ... ?[--channel <channel name>] ?[--mode [remote|local|cache]]
    [update|disable|enable|fix] all ?[--channel <channel name>] ?[--mode [remote|local|cache]]
    [simple-show|show] [installed|enabled|not-installed|disabled|all|snapshot|snapshot-list] ?[--channel <channel name>] ?[--mode [remote|local|cache]]
    save-snapshot ?[--output <snapshot .json/.yaml>]
    restore-snapshot <snapshot .json/.yaml> ?[--pip-non-url] ?[--pip-non-local-url] ?[--pip-local-url]
    cli-only-mode [enable|disable]
    restore-dependencies
    clear
```

## How To Use?
* You can execute it via `python cm-cli.py`.
* For example, if you want to update all custom nodes:
    * In the ComfyUI-Manager directory, you can execute the command `python cm-cli.py update all`.
    * If running from the ComfyUI directory, you can specify the path to cm-cli.py like this: `python custom_nodes/ComfyUI-Manager/cm-cli.py update all`.

## Prerequisite
* It must be run in the same Python environment as the one running ComfyUI.
    * If using a venv, you must run it with the venv activated.
    * If using a portable version, and you are in the directory with the run_nvidia_gpu.bat file, you should execute the command as follows:
        `.\python_embeded\python.exe ComfyUI\custom_nodes\ComfyUI-Manager\cm-cli.py update all`
* The path for ComfyUI can be set with the COMFYUI_PATH environment variable. If omitted, a warning message will appear, and the path will be set relative to the installed location of ComfyUI-Manager:
        ```
        WARN: The `COMFYUI_PATH` environment variable is not set. Assuming `custom_nodes/ComfyUI-Manager/../../` as the ComfyUI path.
        ```

## Features

### 1. --channel, --mode
* For viewing information and managing custom nodes, you can set the information database through --channel and --mode.
* For instance, executing the command `python cm-cli.py update all --channel recent --mode remote` will operate based on the latest information from remote rather than local data embedded in the current ComfyUI-Manager repo and will only target the list in the recent channel.
* --channel, --mode are only available with the commands `simple-show, show, install, uninstall, update, disable, enable, fix`.

### 2. Viewing Management Information

`[simple-show|show] [installed|enabled|not-installed|disabled|all|snapshot|snapshot-list] ?[--channel <channel name>] ?[--mode [remote|local|cache]]`

* `[show|simple-show]` - `show` provides detailed information, while `simple-show` displays information more simply.

Executing a command like `python cm-cli.py show installed` will display detailed information about the installed custom nodes.

```
-= ComfyUI-Manager CLI (V2.24) =-

FETCH DATA from: https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/custom-node-list.json
[    ENABLED    ]  ComfyUI-Manager                                   (author: Dr.Lt.Data)
[    ENABLED    ]  ComfyUI-Impact-Pack                               (author: Dr.Lt.Data)
[    ENABLED    ]  ComfyUI-Inspire-Pack                              (author: Dr.Lt.Data)
[    ENABLED    ]  ComfyUI_experiments                               (author: comfyanonymous)
[    ENABLED    ]  ComfyUI-SAI_API                                   (author: Stability-AI)
[    ENABLED    ]  stability-ComfyUI-nodes                           (author: Stability-AI)
[    ENABLED    ]  comfyui_controlnet_aux                            (author: Fannovel16)
[    ENABLED    ]  ComfyUI-Frame-Interpolation                       (author: Fannovel16)
[    DISABLED   ]  ComfyUI-Loopchain                                 (author: Fannovel16)
```

Using a command like `python cm-cli.py simple-show installed` will simply display information about the installed custom nodes.

```
-= ComfyUI-Manager CLI (V2.24) =-

FETCH DATA from: https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/custom-node-list.json
ComfyUI-Manager                                   
ComfyUI-Impact-Pack                               
ComfyUI-Inspire-Pack                              
ComfyUI_experiments                               
ComfyUI-SAI_API                                   
stability-ComfyUI-nodes                           
comfyui_controlnet_aux                            
ComfyUI-Frame-Interpolation                       
ComfyUI-Loopchain                                 
```

`[installed|enabled|not-installed|disabled|all|snapshot|snapshot-list]`
   * `enabled`, `disabled`: Shows nodes that have been enabled or disabled among the installed custom nodes.
   * `installed`: Shows all nodes that have been installed, regardless of whether they are enabled or disabled.
   * `not-installed`: Shows a list of custom nodes that have not been installed.
   * `all`: Shows a list of all custom nodes.
   * `snapshot`: Displays snapshot information of the currently installed custom nodes. When viewed with `show`, it is displayed in JSON format, and with `simple-show`, it is displayed simply, along with the commit hash.
   * `snapshot-list`: Shows a list of snapshot files stored in ComfyUI-Manager/snapshots.

### 3. Managing Custom Nodes

`[install|reinstall|uninstall|update|disable|enable|fix] node_name ... ?[--channel <channel name>] ?[--mode [remote|local|cache]]`

* You can apply management functions by listing the names of custom nodes, such as `python cm-cli.py install ComfyUI-Impact-Pack ComfyUI-Inspire-Pack ComfyUI_experiments`.
* The names of the custom nodes are as shown by `show` and are the names of the git repositories.
(Plans are to update the use of nicknames in the future.)

`[update|disable|enable|fix] all ?[--channel <channel name>] ?[--mode [remote|local|cache]]`

* The `update, disable, enable, fix` functions can be specified for all.

* Detailed Operations
    * `install`: Installs the specified custom nodes.
    * `reinstall`: Removes and then reinstalls the specified custom nodes.
    * `uninstall`: Uninstalls the specified custom nodes.
    * `update`: Updates the specified custom nodes.
    * `disable`: Disables the specified custom nodes.
    * `enable`: Enables the specified custom nodes.
    * `fix`: Attempts to fix dependencies for the specified custom nodes.


### 4. Snapshot Management
* `python cm-cli.py save-snapshot [--output <snapshot .json/.yaml>]`: Saves the current snapshot.
  * With `--output`, you can save a file in .yaml format to any specified path.
* `python cm-cli.py restore-snapshot <snapshot .json/.yaml>`: Restores to the specified snapshot.
  * If a file exists at the snapshot path, that snapshot is loaded.
  * If no file exists at the snapshot path, it is implicitly assumed to be in ComfyUI-Manager/snapshots.
  * `--pip-non-url`: Restore for pip packages registered on PyPI.
  * `--pip-non-local-url`: Restore for pip packages registered at web URLs.
  * `--pip-local-url`: Restore for pip packages specified by local paths. 


### 5. CLI Only Mode

You can set whether to use ComfyUI-Manager solely via CLI.

`cli-only-mode [enable|disable]`

* This mode can be used if you want to restrict the use of ComfyUI-Manager through the GUI for security or policy reasons.
    * When CLI only mode is enabled, ComfyUI-Manager is loaded in a very restricted state, the internal web API is disabled, and the Manager button is not displayed in the main menu.

### 6. Dependency Restoration

`restore-dependencies`

* This command can be used if custom nodes are installed under the `ComfyUI/custom_nodes` path but their dependencies are not installed.
* It is useful when starting a new cloud instance, like colab, where dependencies need to be reinstalled and installation scripts re-executed.
* It can also be utilized if ComfyUI is reinstalled and only the custom_nodes path has been backed up and restored.

### 7. Clear

In the GUI, installations, updates, or snapshot restorations are scheduled to execute the next time ComfyUI is launched. The `clear` command clears this scheduled state, ensuring no pre-execution actions are applied.