import os
import sys
import traceback
import json
import asyncio
import subprocess
import shutil
import concurrent
import threading
from typing import Optional

import typer
from rich import print
from typing_extensions import List, Annotated
import re
import git

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "glob"))
import manager_core as core
import cm_global

comfyui_manager_path = os.path.dirname(__file__)
comfy_path = os.environ.get('COMFYUI_PATH')

if comfy_path is None:
    print(f"\n[bold yellow]WARN: The `COMFYUI_PATH` environment variable is not set. Assuming `custom_nodes/ComfyUI-Manager/../../` as the ComfyUI path.[/bold yellow]", file=sys.stderr)
    comfy_path = os.path.abspath(os.path.join(comfyui_manager_path, '..', '..'))

startup_script_path = os.path.join(comfyui_manager_path, "startup-scripts")
custom_nodes_path = os.path.join(comfy_path, 'custom_nodes')

script_path = os.path.join(startup_script_path, "install-scripts.txt")
restore_snapshot_path = os.path.join(startup_script_path, "restore-snapshot.json")
pip_overrides_path = os.path.join(comfyui_manager_path, "pip_overrides.json")
git_script_path = os.path.join(comfyui_manager_path, "git_helper.py")

cm_global.pip_downgrade_blacklist = ['torch', 'torchsde', 'torchvision', 'transformers', 'safetensors', 'kornia']
cm_global.pip_overrides = {}
if os.path.exists(pip_overrides_path):
    with open(pip_overrides_path, 'r', encoding="UTF-8", errors="ignore") as json_file:
        cm_global.pip_overrides = json.load(json_file)


def check_comfyui_hash():
    repo = git.Repo(comfy_path)
    core.comfy_ui_revision = len(list(repo.iter_commits('HEAD')))

    comfy_ui_hash = repo.head.commit.hexsha
    cm_global.variables['comfyui.revision'] = core.comfy_ui_revision

    core.comfy_ui_commit_datetime = repo.head.commit.committed_datetime


check_comfyui_hash()  # This is a preparation step for manager_core


def read_downgrade_blacklist():
    try:
        import configparser
        config_path = os.path.join(os.path.dirname(__file__), "config.ini")
        config = configparser.ConfigParser()
        config.read(config_path)
        default_conf = config['default']

        if 'downgrade_blacklist' in default_conf:
            items = default_conf['downgrade_blacklist'].split(',')
            items = [x.strip() for x in items if x != '']
            cm_global.pip_downgrade_blacklist += items
            cm_global.pip_downgrade_blacklist = list(set(cm_global.pip_downgrade_blacklist))
    except:
        pass


read_downgrade_blacklist()  # This is a preparation step for manager_core


class Ctx:
    def __init__(self):
        self.channel = 'default'
        self.mode = 'remote'
        self.processed_install = set()
        self.custom_node_map_cache = None

    def set_channel_mode(self, channel, mode):
        if mode is not None:
            self.mode = mode

        valid_modes = ["remote", "local", "cache"]
        if mode and mode.lower() not in valid_modes:
            typer.echo(
                f"Invalid mode: {mode}. Allowed modes are 'remote', 'local', 'cache'.",
                err=True,
            )
            exit(1)

        if channel is not None:
            self.channel = channel

    def post_install(self, url):
        try:
            repository_name = url.split("/")[-1].strip()
            repo_path = os.path.join(custom_nodes_path, repository_name)
            repo_path = os.path.abspath(repo_path)

            requirements_path = os.path.join(repo_path, 'requirements.txt')
            install_script_path = os.path.join(repo_path, 'install.py')

            if os.path.exists(requirements_path):
                with (open(requirements_path, 'r', encoding="UTF-8", errors="ignore") as file):
                    for line in file:
                        package_name = core.remap_pip_package(line.strip())
                        if package_name and not core.is_installed(package_name):
                            install_cmd = [sys.executable, "-m", "pip", "install", package_name]
                            output = subprocess.check_output(install_cmd, cwd=repo_path, text=True)
                            for msg_line in output.split('\n'):
                                if 'Requirement already satisfied:' in msg_line:
                                    print('.', end='')
                                else:
                                    print(msg_line)

            if os.path.exists(install_script_path) and f'{repo_path}/install.py' not in self.processed_install:
                self.processed_install.add(f'{repo_path}/install.py')
                install_cmd = [sys.executable, install_script_path]
                output = subprocess.check_output(install_cmd, cwd=repo_path, text=True)
                for msg_line in output.split('\n'):
                    if 'Requirement already satisfied:' in msg_line:
                        print('.', end='')
                    else:
                        print(msg_line)

        except Exception:
            print(f"ERROR: Restoring '{url}' is failed.")

    def restore_dependencies(self):
        node_paths = [os.path.join(custom_nodes_path, name) for name in os.listdir(custom_nodes_path)
                      if os.path.isdir(os.path.join(custom_nodes_path, name)) and not name.endswith('.disabled')]

        total = len(node_paths)
        i = 1
        for x in node_paths:
            print(f"----------------------------------------------------------------------------------------------------")
            print(f"Restoring [{i}/{total}]: {x}")
            self.post_install(x)
            i += 1

    def load_custom_nodes(self):
        channel_dict = core.get_channel_dict()
        if self.channel not in channel_dict:
            print(f"[bold red]ERROR: Invalid channel is specified `--channel {self.channel}`[/bold red]", file=sys.stderr)
            exit(1)

        if self.mode not in ['remote', 'local', 'cache']:
            print(f"[bold red]ERROR: Invalid mode is specified `--mode {self.mode}`[/bold red]", file=sys.stderr)
            exit(1)

        channel_url = channel_dict[self.channel]

        res = {}
        json_obj = asyncio.run(core.get_data_by_mode(self.mode, 'custom-node-list.json', channel_url=channel_url))
        for x in json_obj['custom_nodes']:
            for y in x['files']:
                if 'github.com' in y and not (y.endswith('.py') or y.endswith('.js')):
                    repo_name = y.split('/')[-1]
                    res[repo_name] = (x, False)

            if 'id' in x:
                if x['id'] not in res:
                    res[x['id']] = (x, True)

        return res

    def get_custom_node_map(self):
        if self.custom_node_map_cache is not None:
            return self.custom_node_map_cache

        self.custom_node_map_cache = self.load_custom_nodes()

        return self.custom_node_map_cache

    def lookup_node_path(self, node_name, robust=False):
        if '..' in node_name:
            print(f"\n[bold red]ERROR: Invalid node name '{node_name}'[/bold red]\n")
            exit(2)

        custom_node_map = self.get_custom_node_map()
        if node_name in custom_node_map:
            node_url = custom_node_map[node_name][0]['files'][0]
            repo_name = node_url.split('/')[-1]
            node_path = os.path.join(custom_nodes_path, repo_name)
            return node_path, custom_node_map[node_name][0]
        elif robust:
            node_path = os.path.join(custom_nodes_path, node_name)
            return node_path, None

        print(f"\n[bold red]ERROR: Invalid node name '{node_name}'[/bold red]\n")
        exit(2)


cm_ctx = Ctx()


def install_node(node_name, is_all=False, cnt_msg=''):
    if core.is_valid_url(node_name):
        # install via urls
        res = core.gitclone_install([node_name])
        if not res:
            print(f"[bold red]ERROR: An error occurred while installing '{node_name}'.[/bold red]")
        else:
            print(f"{cnt_msg} [INSTALLED] {node_name:50}")
    else:
        node_path, node_item = cm_ctx.lookup_node_path(node_name)

        if os.path.exists(node_path):
            if not is_all:
                print(f"{cnt_msg} [ SKIPPED ] {node_name:50} => Already installed")
        elif os.path.exists(node_path + '.disabled'):
            enable_node(node_name)
        else:
            res = core.gitclone_install(node_item['files'], instant_execution=True, msg_prefix=f"[{cnt_msg}] ")
            if not res:
                print(f"[bold red]ERROR: An error occurred while installing '{node_name}'.[/bold red]")
            else:
                print(f"{cnt_msg} [INSTALLED] {node_name:50}")


def reinstall_node(node_name, is_all=False, cnt_msg=''):
    node_path, node_item = cm_ctx.lookup_node_path(node_name)

    if os.path.exists(node_path):
        shutil.rmtree(node_path)
    if os.path.exists(node_path + '.disabled'):
        shutil.rmtree(node_path + '.disabled')

    install_node(node_name, is_all=is_all, cnt_msg=cnt_msg)


def fix_node(node_name, is_all=False, cnt_msg=''):
    node_path, node_item = cm_ctx.lookup_node_path(node_name, robust=True)

    files = node_item['files'] if node_item is not None else [node_path]

    if os.path.exists(node_path):
        print(f"{cnt_msg} [   FIXING  ]: {node_name:50} => Disabled")
        res = core.gitclone_fix(files, instant_execution=True)
        if not res:
            print(f"ERROR: An error occurred while fixing '{node_name}'.")
    elif not is_all and os.path.exists(node_path + '.disabled'):
        print(f"{cnt_msg} [  SKIPPED  ]: {node_name:50} => Disabled")
    elif not is_all:
        print(f"{cnt_msg} [  SKIPPED  ]: {node_name:50} => Not installed")


def uninstall_node(node_name, is_all=False, cnt_msg=''):
    node_path, node_item = cm_ctx.lookup_node_path(node_name, robust=True)

    files = node_item['files'] if node_item is not None else [node_path]

    if os.path.exists(node_path) or os.path.exists(node_path + '.disabled'):
        res = core.gitclone_uninstall(files)
        if not res:
            print(f"ERROR: An error occurred while uninstalling '{node_name}'.")
        else:
            print(f"{cnt_msg} [UNINSTALLED] {node_name:50}")
    else:
        print(f"{cnt_msg} [  SKIPPED  ]: {node_name:50} => Not installed")


def update_node(node_name, is_all=False, cnt_msg=''):
    node_path, node_item = cm_ctx.lookup_node_path(node_name, robust=True)

    files = node_item['files'] if node_item is not None else [node_path]

    res = core.gitclone_update(files, skip_script=True, msg_prefix=f"[{cnt_msg}] ")

    if not res:
        print(f"ERROR: An error occurred while updating '{node_name}'.")
        return None

    return node_path


def update_parallel(nodes):
    is_all = False
    if 'all' in nodes:
        is_all = True
        nodes = [x for x in cm_ctx.get_custom_node_map().keys() if os.path.exists(os.path.join(custom_nodes_path, x)) or os.path.exists(os.path.join(custom_nodes_path, x) + '.disabled')]

    nodes = [x for x in nodes if x.lower() not in ['comfy', 'comfyui', 'all']]

    total = len(nodes)

    lock = threading.Lock()
    processed = []

    i = 0

    def process_custom_node(x):
        nonlocal i
        nonlocal processed

        with lock:
            i += 1

        try:
            node_path = update_node(x, is_all=is_all, cnt_msg=f'{i}/{total}')
            with lock:
                processed.append(node_path)
        except Exception as e:
            print(f"ERROR: {e}")
            traceback.print_exc()

    with concurrent.futures.ThreadPoolExecutor(4) as executor:
        for item in nodes:
            executor.submit(process_custom_node, item)

    i = 1
    for node_path in processed:
        if node_path is None:
            print(f"[{i}/{total}] Post update: ERROR")
        else:
            print(f"[{i}/{total}] Post update: {node_path}")
            cm_ctx.post_install(node_path)
        i += 1


def update_comfyui():
    res = core.update_path(comfy_path, instant_execution=True)
    if res == 'fail':
        print("Updating ComfyUI has failed.")
    elif res == 'updated':
        print("ComfyUI is updated.")
    else:
        print("ComfyUI is already up to date.")


def enable_node(node_name, is_all=False, cnt_msg=''):
    if node_name == 'ComfyUI-Manager':
        return

    node_path, node_item = cm_ctx.lookup_node_path(node_name, robust=True)

    if os.path.exists(node_path + '.disabled'):
        current_name = node_path + '.disabled'
        os.rename(current_name, node_path)
        print(f"{cnt_msg} [ENABLED] {node_name:50}")
    elif os.path.exists(node_path):
        print(f"{cnt_msg} [SKIPPED] {node_name:50} => Already enabled")
    elif not is_all:
        print(f"{cnt_msg} [SKIPPED] {node_name:50} => Not installed")


def disable_node(node_name, is_all=False, cnt_msg=''):
    if node_name == 'ComfyUI-Manager':
        return

    node_path, node_item = cm_ctx.lookup_node_path(node_name, robust=True)

    if os.path.exists(node_path):
        current_name = node_path
        new_name = node_path + '.disabled'
        os.rename(current_name, new_name)
        print(f"{cnt_msg} [DISABLED] {node_name:50}")
    elif os.path.exists(node_path + '.disabled'):
        print(f"{cnt_msg} [ SKIPPED] {node_name:50} => Already disabled")
    elif not is_all:
        print(f"{cnt_msg} [ SKIPPED] {node_name:50} => Not installed")


def show_list(kind, simple=False):
    for k, v in cm_ctx.get_custom_node_map().items():
        if v[1]:
            continue

        node_path = os.path.join(custom_nodes_path, k)

        states = set()
        if os.path.exists(node_path):
            prefix = '[    ENABLED    ] '
            states.add('installed')
            states.add('enabled')
            states.add('all')
        elif os.path.exists(node_path + '.disabled'):
            prefix = '[    DISABLED   ] '
            states.add('installed')
            states.add('disabled')
            states.add('all')
        else:
            prefix = '[ NOT INSTALLED ] '
            states.add('not-installed')
            states.add('all')

        if kind in states:
            if simple:
                print(f"{k:50}")
            else:
                short_id = v[0].get('id', "")
                print(f"{prefix} {k:50} {short_id:20} (author: {v[0]['author']})")

    # unregistered nodes
    candidates = os.listdir(os.path.realpath(custom_nodes_path))

    for k in candidates:
        fullpath = os.path.join(custom_nodes_path, k)

        if os.path.isfile(fullpath):
            continue

        if k in ['__pycache__']:
            continue

        states = set()
        if k.endswith('.disabled'):
            prefix = '[    DISABLED   ] '
            states.add('installed')
            states.add('disabled')
            states.add('all')
            k = k[:-9]
        else:
            prefix = '[    ENABLED    ] '
            states.add('installed')
            states.add('enabled')
            states.add('all')

        if k not in cm_ctx.get_custom_node_map():
            if kind in states:
                if simple:
                    print(f"{k:50}")
                else:
                    print(f"{prefix} {k:50} {'':20} (author: N/A)")


def show_snapshot(simple_mode=False):
    json_obj = core.get_current_snapshot()

    if simple_mode:
        print(f"[{json_obj['comfyui']}] comfyui")
        for k, v in json_obj['git_custom_nodes'].items():
            print(f"[{v['hash']}] {k}")
        for v in json_obj['file_custom_nodes']:
            print(f"[                   N/A                  ] {v['filename']}")

    else:
        formatted_json = json.dumps(json_obj, ensure_ascii=False, indent=4)
        print(formatted_json)


def show_snapshot_list(simple_mode=False):
    snapshot_path = os.path.join(comfyui_manager_path, 'snapshots')

    files = os.listdir(snapshot_path)
    json_files = [x for x in files if x.endswith('.json')]
    for x in sorted(json_files):
        print(x)


def cancel():
    if os.path.exists(script_path):
        os.remove(script_path)

    if os.path.exists(restore_snapshot_path):
        os.remove(restore_snapshot_path)


def auto_save_snapshot():
    path = core.save_snapshot_with_postfix('cli-autosave')
    print(f"Current snapshot is saved as `{path}`")


def for_each_nodes(nodes, act, allow_all=True):
    is_all = False
    if allow_all and 'all' in nodes:
        is_all = True
        nodes = [x for x in cm_ctx.get_custom_node_map().keys() if os.path.exists(os.path.join(custom_nodes_path, x)) or os.path.exists(os.path.join(custom_nodes_path, x) + '.disabled')]

    nodes = [x for x in nodes if x.lower() not in ['comfy', 'comfyui', 'all']]

    total = len(nodes)
    i = 1
    for x in nodes:
        try:
            act(x, is_all=is_all, cnt_msg=f'{i}/{total}')
        except Exception as e:
            print(f"ERROR: {e}")
            traceback.print_exc()
        i += 1


app = typer.Typer()


@app.command(help="Display help for commands")
def help(ctx: typer.Context):
    print(ctx.find_root().get_help())
    ctx.exit(0)


@app.command(help="Install custom nodes")
def install(
        nodes: List[str] = typer.Argument(
            ..., help="List of custom nodes to install"
        ),
        channel: Annotated[
            str,
            typer.Option(
                show_default=False,
                help="Specify the operation mode"
            ),
        ] = None,
        mode: str = typer.Option(
            None,
            help="[remote|local|cache]"
        ),
):
    cm_ctx.set_channel_mode(channel, mode)
    for_each_nodes(nodes, act=install_node)


@app.command(help="Reinstall custom nodes")
def reinstall(
        nodes: List[str] = typer.Argument(
            ..., help="List of custom nodes to reinstall"
        ),
        channel: Annotated[
            str,
            typer.Option(
                show_default=False,
                help="Specify the operation mode"
            ),
        ] = None,
        mode: str = typer.Option(
            None,
            help="[remote|local|cache]"
        ),
):
    cm_ctx.set_channel_mode(channel, mode)
    for_each_nodes(nodes, act=reinstall_node)


@app.command(help="Uninstall custom nodes")
def uninstall(
        nodes: List[str] = typer.Argument(
            ..., help="List of custom nodes to uninstall"
        ),
        channel: Annotated[
            str,
            typer.Option(
                show_default=False,
                help="Specify the operation mode"
            ),
        ] = None,
        mode: str = typer.Option(
            None,
            help="[remote|local|cache]"
        ),
):
    cm_ctx.set_channel_mode(channel, mode)
    for_each_nodes(nodes, act=uninstall_node)


@app.command(help="Disable custom nodes")
def update(
        nodes: List[str] = typer.Argument(
            ...,
            help="[all|List of custom nodes to update]"
        ),
        channel: Annotated[
            str,
            typer.Option(
                show_default=False,
                help="Specify the operation mode"
            ),
        ] = None,
        mode: str = typer.Option(
            None,
            help="[remote|local|cache]"
        ),
):
    cm_ctx.set_channel_mode(channel, mode)

    if 'all' in nodes:
        auto_save_snapshot()

    for x in nodes:
        if x.lower() in ['comfyui', 'comfy', 'all']:
            update_comfyui()
            break

    update_parallel(nodes)


@app.command(help="Disable custom nodes")
def disable(
        nodes: List[str] = typer.Argument(
            ...,
            help="[all|List of custom nodes to disable]"
        ),
        channel: Annotated[
            str,
            typer.Option(
                show_default=False,
                help="Specify the operation mode"
            ),
        ] = None,
        mode: str = typer.Option(
            None,
            help="[remote|local|cache]"
        ),
):
    cm_ctx.set_channel_mode(channel, mode)

    if 'all' in nodes:
        auto_save_snapshot()

    for_each_nodes(nodes, disable_node, allow_all=True)


@app.command(help="Enable custom nodes")
def enable(
        nodes: List[str] = typer.Argument(
            ...,
            help="[all|List of custom nodes to enable]"
        ),
        channel: Annotated[
            str,
            typer.Option(
                show_default=False,
                help="Specify the operation mode"
            ),
        ] = None,
        mode: str = typer.Option(
            None,
            help="[remote|local|cache]"
        ),
):
    cm_ctx.set_channel_mode(channel, mode)

    if 'all' in nodes:
        auto_save_snapshot()

    for_each_nodes(nodes, enable_node, allow_all=True)


@app.command(help="Fix dependencies of custom nodes")
def fix(
        nodes: List[str] = typer.Argument(
            ...,
            help="[all|List of custom nodes to fix]"
        ),
        channel: Annotated[
            str,
            typer.Option(
                show_default=False,
                help="Specify the operation mode"
            ),
        ] = None,
        mode: str = typer.Option(
            None,
            help="[remote|local|cache]"
        ),
):
    cm_ctx.set_channel_mode(channel, mode)

    if 'all' in nodes:
        auto_save_snapshot()

    for_each_nodes(nodes, fix_node, allow_all=True)


@app.command("show", help="Show node list (simple mode)")
def show(
        arg: str = typer.Argument(
            help="[installed|enabled|not-installed|disabled|all|snapshot|snapshot-list]"
        ),
        channel: Annotated[
            str,
            typer.Option(
                show_default=False,
                help="Specify the operation mode"
            ),
        ] = None,
        mode: str = typer.Option(
            None,
            help="[remote|local|cache]"
        ),
):
    valid_commands = [
        "installed",
        "enabled",
        "not-installed",
        "disabled",
        "all",
        "snapshot",
        "snapshot-list",
    ]
    if arg not in valid_commands:
        typer.echo(f"Invalid command: `show {arg}`", err=True)
        exit(1)

    cm_ctx.set_channel_mode(channel, mode)
    if arg == 'snapshot':
        show_snapshot()
    elif arg == 'snapshot-list':
        show_snapshot_list()
    else:
        show_list(arg)


@app.command("simple-show", help="Show node list (simple mode)")
def simple_show(
        arg: str = typer.Argument(
            help="[installed|enabled|not-installed|disabled|all|snapshot|snapshot-list]"
        ),
        channel: Annotated[
            str,
            typer.Option(
                show_default=False,
                help="Specify the operation mode"
            ),
        ] = None,
        mode: str = typer.Option(
            None,
            help="[remote|local|cache]"
        ),
):
    valid_commands = [
        "installed",
        "enabled",
        "not-installed",
        "disabled",
        "all",
        "snapshot",
        "snapshot-list",
    ]
    if arg not in valid_commands:
        typer.echo(f"[bold red]Invalid command: `show {arg}`[/bold red]", err=True)
        exit(1)

    cm_ctx.set_channel_mode(channel, mode)
    if arg == 'snapshot':
        show_snapshot(True)
    elif arg == 'snapshot-list':
        show_snapshot_list(True)
    else:
        show_list(arg, True)


@app.command('cli-only-mode', help="Set whether to use ComfyUI-Manager in CLI-only mode.")
def cli_only_mode(
        mode: str = typer.Argument(
            ..., help="[enable|disable]"
        )):
    cli_mode_flag = os.path.join(os.path.dirname(__file__), '.enable-cli-only-mode')
    if mode.lower() == 'enable':
        with open(cli_mode_flag, 'w') as file:
            pass
        print(f"\nINFO: `cli-only-mode` is enabled\n")
    elif mode.lower() == 'disable':
        if os.path.exists(cli_mode_flag):
            os.remove(cli_mode_flag)
        print(f"\nINFO: `cli-only-mode` is disabled\n")
    else:
        print(f"\n[bold red]Invalid value for cli-only-mode: {mode}[/bold red]\n")
        exit(1)


@app.command(
    "deps-in-workflow", help="Generate dependencies file from workflow (.json/.png)"
)
def deps_in_workflow(
        workflow: Annotated[
            str, typer.Option(show_default=False, help="Workflow file (.json/.png)")
        ],
        output: Annotated[
            str, typer.Option(show_default=False, help="Output file (.json)")
        ],
        channel: Annotated[
            str,
            typer.Option(
                show_default=False,
                help="Specify the operation mode"
            ),
        ] = None,
        mode: str = typer.Option(
            None,
            help="[remote|local|cache]"
        ),
):
    cm_ctx.set_channel_mode(channel, mode)

    input_path = workflow
    output_path = output

    if not os.path.exists(input_path):
        print(f"[bold red]File not found: {input_path}[/bold red]")
        exit(1)

    used_exts, unknown_nodes = asyncio.run(core.extract_nodes_from_workflow(input_path, mode=cm_ctx.mode, channel_url=cm_ctx.channel))

    custom_nodes = {}
    for x in used_exts:
        custom_nodes[x] = {'state': core.simple_check_custom_node(x),
                           'hash': '-'
                           }

    res = {
        'custom_nodes': custom_nodes,
        'unknown_nodes': list(unknown_nodes)
    }

    with open(output_path, "w", encoding='utf-8') as output_file:
        json.dump(res, output_file, indent=4)

    print(f"Workflow dependencies are being saved into {output_path}.")


@app.command("save-snapshot", help="Save a snapshot of the current ComfyUI environment. If output path isn't provided. Save to ComfyUI-Manager/snapshots path.")
def save_snapshot(
        output: Annotated[
            str,
            typer.Option(
                show_default=False, help="Specify the output file path. (.json/.yaml)"
            ),
        ] = None,
):
    path = core.save_snapshot_with_postfix('snapshot', output)
    print(f"Current snapshot is saved as `{path}`")


@app.command("restore-snapshot", help="Restore snapshot from snapshot file")
def restore_snapshot(
        snapshot_name: str,
        pip_non_url: Optional[bool] = typer.Option(
            default=None,
            show_default=False,
            is_flag=True,
            help="Restore for pip packages registered on PyPI.",
        ),
        pip_non_local_url: Optional[bool] = typer.Option(
            default=None,
            show_default=False,
            is_flag=True,
            help="Restore for pip packages registered at web URLs.",
        ),
        pip_local_url: Optional[bool] = typer.Option(
            default=None,
            show_default=False,
            is_flag=True,
            help="Restore for pip packages specified by local paths.",
        ),
):
    extras = []
    if pip_non_url:
        extras.append('--pip-non-url')

    if pip_non_local_url:
        extras.append('--pip-non-local-url')

    if pip_local_url:
        extras.append('--pip-local-url')

    print(f"PIPs restore mode: {extras}")

    if os.path.exists(snapshot_name):
        snapshot_path = os.path.abspath(snapshot_name)
    else:
        snapshot_path = os.path.join(core.comfyui_manager_path, 'snapshots', snapshot_name)
        if not os.path.exists(snapshot_path):
            print(f"[bold red]ERROR: `{snapshot_path}` is not exists.[/bold red]")
            exit(1)

    try:
        cloned_repos = []
        checkout_repos = []
        skipped_repos = []
        enabled_repos = []
        disabled_repos = []
        is_failed = False

        def extract_infos(msg):
            nonlocal is_failed

            for x in msg:
                if x.startswith("CLONE: "):
                    cloned_repos.append(x[7:])
                elif x.startswith("CHECKOUT: "):
                    checkout_repos.append(x[10:])
                elif x.startswith("SKIPPED: "):
                    skipped_repos.append(x[9:])
                elif x.startswith("ENABLE: "):
                    enabled_repos.append(x[8:])
                elif x.startswith("DISABLE: "):
                    disabled_repos.append(x[9:])
                elif 'APPLY SNAPSHOT: False' in x:
                    is_failed = True

        print(f"Restore snapshot.")
        cmd_str = [sys.executable, git_script_path, '--apply-snapshot', snapshot_path] + extras
        output = subprocess.check_output(cmd_str, cwd=custom_nodes_path, text=True)
        msg_lines = output.split('\n')
        extract_infos(msg_lines)

        for url in cloned_repos:
            cm_ctx.post_install(url)

        # print summary
        for x in cloned_repos:
            print(f"[ INSTALLED ] {x}")
        for x in checkout_repos:
            print(f"[  CHECKOUT ] {x}")
        for x in enabled_repos:
            print(f"[  ENABLED  ] {x}")
        for x in disabled_repos:
            print(f"[  DISABLED ] {x}")

        if is_failed:
            print(output)
            print("[bold red]ERROR: Failed to restore snapshot.[/bold red]")

    except Exception:
        print("[bold red]ERROR: Failed to restore snapshot.[/bold red]")
        traceback.print_exc()
        raise typer.Exit(code=1)


@app.command(
    "restore-dependencies", help="Restore dependencies from whole installed custom nodes."
)
def restore_dependencies():
    node_paths = [os.path.join(custom_nodes_path, name) for name in os.listdir(custom_nodes_path)
                  if os.path.isdir(os.path.join(custom_nodes_path, name)) and not name.endswith('.disabled')]

    total = len(node_paths)
    i = 1
    for x in node_paths:
        print(f"----------------------------------------------------------------------------------------------------")
        print(f"Restoring [{i}/{total}]: {x}")
        cm_ctx.post_install(x)
        i += 1


@app.command(
    "post-install", help="Install dependencies and execute installation script"
)
def post_install(
        path: str = typer.Argument(
            help="path to custom node",
        )):
    path = os.path.expanduser(path)
    cm_ctx.post_install(path)


@app.command(
    "install-deps",
    help="Install dependencies from dependencies file(.json) or workflow(.png/.json)",
)
def install_deps(
        deps: str = typer.Argument(
            help="Dependency spec file (.json)",
        ),
        channel: Annotated[
            str,
            typer.Option(
                show_default=False,
                help="Specify the operation mode"
            ),
        ] = None,
        mode: str = typer.Option(
            None,
            help="[remote|local|cache]"
        ),
):
    cm_ctx.set_channel_mode(channel, mode)
    auto_save_snapshot()

    if not os.path.exists(deps):
        print(f"[bold red]File not found: {deps}[/bold red]")
        exit(1)
    else:
        with open(deps, 'r', encoding="UTF-8", errors="ignore") as json_file:
            try:
                json_obj = json.load(json_file)
            except:
                print(f"[bold red]Invalid json file: {deps}[/bold red]")
                exit(1)

            for k in json_obj['custom_nodes'].keys():
                state = core.simple_check_custom_node(k)
                if state == 'installed':
                    continue
                elif state == 'not-installed':
                    core.gitclone_install([k], instant_execution=True)
                else:  # disabled
                    core.gitclone_set_active([k], False)

        print("Dependency installation and activation complete.")


@app.command(help="Clear reserved startup action in ComfyUI-Manager")
def clear():
    cancel()


@app.command("export-custom-node-ids", help="Export custom node ids")
def export_custom_node_ids(
        path: str,
        channel: Annotated[
            str,
            typer.Option(
                show_default=False,
                help="Specify the operation mode"
            ),
        ] = None,
        mode: str = typer.Option(
            None,
            help="[remote|local|cache]"
        )):
    cm_ctx.set_channel_mode(channel, mode)

    with open(path, "w", encoding='utf-8') as output_file:
        for x in cm_ctx.get_custom_node_map().keys():
            print(x, file=output_file)


if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    sys.exit(app())

print(f"")
