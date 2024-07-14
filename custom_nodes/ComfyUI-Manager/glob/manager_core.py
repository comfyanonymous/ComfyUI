import os
import sys
import subprocess
import re
import shutil
import configparser
import platform
from datetime import datetime
import git
from git.remote import RemoteProgress
from urllib.parse import urlparse
from tqdm.auto import tqdm
import aiohttp
import threading
import json
import time
import yaml
import zipfile

glob_path = os.path.join(os.path.dirname(__file__))  # ComfyUI-Manager/glob
sys.path.append(glob_path)

import cm_global
from manager_util import *

version = [2, 46, 2]
version_str = f"V{version[0]}.{version[1]}" + (f'.{version[2]}' if len(version) > 2 else '')


comfyui_manager_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
custom_nodes_path = os.path.abspath(os.path.join(comfyui_manager_path, '..'))

comfy_path = os.environ.get('COMFYUI_PATH')
if comfy_path is None:
    comfy_path = os.path.abspath(os.path.join(custom_nodes_path, '..'))

channel_list_path = os.path.join(comfyui_manager_path, 'channels.list')
config_path = os.path.join(comfyui_manager_path, "config.ini")
startup_script_path = os.path.join(comfyui_manager_path, "startup-scripts")
git_script_path = os.path.join(comfyui_manager_path, "git_helper.py")
cache_dir = os.path.join(comfyui_manager_path, '.cache')
cached_config = None
js_path = None

comfy_ui_required_revision = 1930
comfy_ui_required_commit_datetime = datetime(2024, 1, 24, 0, 0, 0)

comfy_ui_revision = "Unknown"
comfy_ui_commit_datetime = datetime(1900, 1, 1, 0, 0, 0)


cache_lock = threading.Lock()


channel_dict = None
channel_list = None
pip_map = None


def remap_pip_package(pkg):
    if pkg in cm_global.pip_overrides:
        res = cm_global.pip_overrides[pkg]
        print(f"[ComfyUI-Manager] '{pkg}' is remapped to '{res}'")
        return res
    else:
        return pkg


def get_installed_packages():
    global pip_map

    if pip_map is None:
        try:
            result = subprocess.check_output([sys.executable, '-m', 'pip', 'list'], universal_newlines=True)

            pip_map = {}
            for line in result.split('\n'):
                x = line.strip()
                if x:
                    y = line.split()
                    if y[0] == 'Package' or y[0].startswith('-'):
                        continue

                    pip_map[y[0]] = y[1]
        except subprocess.CalledProcessError as e:
            print(f"[ComfyUI-Manager] Failed to retrieve the information of installed pip packages.")
            return set()

    return pip_map


def clear_pip_cache():
    global pip_map
    pip_map = None


def is_blacklisted(name):
    name = name.strip()

    pattern = r'([^<>!=]+)([<>!=]=?)(.*)'
    match = re.search(pattern, name)

    if match:
        name = match.group(1)

    if name in cm_global.pip_downgrade_blacklist:
        pips = get_installed_packages()

        if match is None:
            if name in pips:
                return True
        elif match.group(2) in ['<=', '==', '<']:
            if name in pips:
                if StrictVersion(pips[name]) >= StrictVersion(match.group(3)):
                    return True

    return False


def is_installed(name):
    name = name.strip()

    if name.startswith('#'):
        return True

    pattern = r'([^<>!=]+)([<>!=]=?)(.*)'
    match = re.search(pattern, name)

    if match:
        name = match.group(1)

    if name in cm_global.pip_downgrade_blacklist:
        pips = get_installed_packages()

        if match is None:
            if name in pips:
                return True
        elif match.group(2) in ['<=', '==', '<']:
            if name in pips:
                if StrictVersion(pips[name]) >= StrictVersion(match.group(3)):
                    print(f"[ComfyUI-Manager] skip black listed pip installation: '{name}'")
                    return True

    return name.lower() in get_installed_packages()


def get_channel_dict():
    global channel_dict

    if channel_dict is None:
        channel_dict = {}

        if not os.path.exists(channel_list_path):
            shutil.copy(channel_list_path+'.template', channel_list_path)

        with open(os.path.join(comfyui_manager_path, 'channels.list'), 'r') as file:
            channels = file.read()
            for x in channels.split('\n'):
                channel_info = x.split("::")
                if len(channel_info) == 2:
                    channel_dict[channel_info[0]] = channel_info[1]

    return channel_dict


def get_channel_list():
    global channel_list

    if channel_list is None:
        channel_list = []
        for k, v in get_channel_dict().items():
            channel_list.append(f"{k}::{v}")

    return channel_list


class ManagerFuncs:
    def __init__(self):
        pass

    def get_current_preview_method(self):
        return "none"

    def run_script(self, cmd, cwd='.'):
        if len(cmd) > 0 and cmd[0].startswith("#"):
            print(f"[ComfyUI-Manager] Unexpected behavior: `{cmd}`")
            return 0

        new_env = os.environ.copy()
        new_env["COMFYUI_PATH"] = comfy_path
        subprocess.check_call(cmd, cwd=cwd, env=new_env)

        return 0


manager_funcs = ManagerFuncs()


def write_config():
    config = configparser.ConfigParser()
    config['default'] = {
        'preview_method': manager_funcs.get_current_preview_method(),
        'badge_mode': get_config()['badge_mode'],
        'git_exe':  get_config()['git_exe'],
        'channel_url': get_config()['channel_url'],
        'share_option': get_config()['share_option'],
        'bypass_ssl': get_config()['bypass_ssl'],
        "file_logging": get_config()['file_logging'],
        'default_ui': get_config()['default_ui'],
        'component_policy': get_config()['component_policy'],
        'double_click_policy': get_config()['double_click_policy'],
        'windows_selector_event_loop_policy': get_config()['windows_selector_event_loop_policy'],
        'model_download_by_agent': get_config()['model_download_by_agent'],
        'downgrade_blacklist': get_config()['downgrade_blacklist'],
        'security_level': get_config()['security_level'],
    }
    with open(config_path, 'w') as configfile:
        config.write(configfile)


def read_config():
    try:
        config = configparser.ConfigParser()
        config.read(config_path)
        default_conf = config['default']

        # policy migration: disable_unsecure_features -> security_level
        if 'disable_unsecure_features' in default_conf:
            if default_conf['disable_unsecure_features'].lower() == 'true':
                security_level = 'strong'
            else:
                security_level = 'normal'
        else:
            security_level = default_conf['security_level'] if 'security_level' in default_conf else 'normal'

        return {
                    'preview_method': default_conf['preview_method'] if 'preview_method' in default_conf else manager_funcs.get_current_preview_method(),
                    'badge_mode': default_conf['badge_mode'] if 'badge_mode' in default_conf else 'none',
                    'git_exe': default_conf['git_exe'] if 'git_exe' in default_conf else '',
                    'channel_url': default_conf['channel_url'] if 'channel_url' in default_conf else 'https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main',
                    'share_option': default_conf['share_option'] if 'share_option' in default_conf else 'all',
                    'bypass_ssl': default_conf['bypass_ssl'].lower() == 'true' if 'bypass_ssl' in default_conf else False,
                    'file_logging': default_conf['file_logging'].lower() == 'true' if 'file_logging' in default_conf else True,
                    'default_ui': default_conf['default_ui'] if 'default_ui' in default_conf else 'none',
                    'component_policy': default_conf['component_policy'] if 'component_policy' in default_conf else 'workflow',
                    'double_click_policy': default_conf['double_click_policy'] if 'double_click_policy' in default_conf else 'copy-all',
                    'windows_selector_event_loop_policy': default_conf['windows_selector_event_loop_policy'].lower() == 'true' if 'windows_selector_event_loop_policy' in default_conf else False,
                    'model_download_by_agent': default_conf['model_download_by_agent'].lower() == 'true' if 'model_download_by_agent' in default_conf else False,
                    'downgrade_blacklist': default_conf['downgrade_blacklist'] if 'downgrade_blacklist' in default_conf else '',
                    'security_level': security_level
               }

    except Exception:
        return {
            'preview_method': manager_funcs.get_current_preview_method(),
            'badge_mode': 'none',
            'git_exe': '',
            'channel_url': 'https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main',
            'share_option': 'all',
            'bypass_ssl': False,
            'file_logging': True,
            'default_ui': 'none',
            'component_policy': 'workflow',
            'double_click_policy': 'copy-all',
            'windows_selector_event_loop_policy': False,
            'model_download_by_agent': False,
            'downgrade_blacklist': '',
            'security_level': 'normal',
        }


def get_config():
    global cached_config

    if cached_config is None:
        cached_config = read_config()

    return cached_config


def switch_to_default_branch(repo):
    show_result = repo.git.remote("show", "origin")
    matches = re.search(r"\s*HEAD branch:\s*(.*)", show_result)
    if matches:
        default_branch = matches.group(1)
        repo.git.checkout(default_branch)


def try_install_script(url, repo_path, install_cmd, instant_execution=False):
    if not instant_execution and ((len(install_cmd) > 0 and install_cmd[0].startswith('#')) or (platform.system() == "Windows" and comfy_ui_commit_datetime.date() >= comfy_ui_required_commit_datetime.date())):
        if not os.path.exists(startup_script_path):
            os.makedirs(startup_script_path)

        script_path = os.path.join(startup_script_path, "install-scripts.txt")
        with open(script_path, "a") as file:
            obj = [repo_path] + install_cmd
            file.write(f"{obj}\n")

        return True
    else:
        if len(install_cmd) == 5 and install_cmd[2:4] == ['pip', 'install']:
            if is_blacklisted(install_cmd[4]):
                print(f"[ComfyUI-Manager] skip black listed pip installation: '{install_cmd[4]}'")
                return True

        print(f"\n## ComfyUI-Manager: EXECUTE => {install_cmd}")
        code = manager_funcs.run_script(install_cmd, cwd=repo_path)

        if platform.system() != "Windows":
            try:
                if comfy_ui_commit_datetime.date() < comfy_ui_required_commit_datetime.date():
                    print("\n\n###################################################################")
                    print(f"[WARN] ComfyUI-Manager: Your ComfyUI version ({comfy_ui_revision})[{comfy_ui_commit_datetime.date()}] is too old. Please update to the latest version.")
                    print(f"[WARN] The extension installation feature may not work properly in the current installed ComfyUI version on Windows environment.")
                    print("###################################################################\n\n")
            except:
                pass

        if code != 0:
            if url is None:
                url = os.path.dirname(repo_path)
            print(f"install script failed: {url}")
            return False


# use subprocess to avoid file system lock by git (Windows)
def __win_check_git_update(path, do_fetch=False, do_update=False):
    if do_fetch:
        command = [sys.executable, git_script_path, "--fetch", path]
    elif do_update:
        command = [sys.executable, git_script_path, "--pull", path]
    else:
        command = [sys.executable, git_script_path, "--check", path]

    new_env = os.environ.copy()
    new_env["COMFYUI_PATH"] = comfy_path
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=custom_nodes_path)
    output, _ = process.communicate()
    output = output.decode('utf-8').strip()

    if 'detected dubious' in output:
        # fix and try again
        safedir_path = path.replace('\\', '/')
        try:
            print(f"[ComfyUI-Manager] Try fixing 'dubious repository' error on '{safedir_path}' repo")
            process = subprocess.Popen(['git', 'config', '--global', '--add', 'safe.directory', safedir_path], env=new_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output, _ = process.communicate()

            process = subprocess.Popen(command, env=new_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output, _ = process.communicate()
            output = output.decode('utf-8').strip()
        except Exception:
            print(f'[ComfyUI-Manager] failed to fixing')

        if 'detected dubious' in output:
            print(f'\n[ComfyUI-Manager] Failed to fixing repository setup. Please execute this command on cmd: \n'
                  f'-----------------------------------------------------------------------------------------\n'
                  f'git config --global --add safe.directory "{safedir_path}"\n'
                  f'-----------------------------------------------------------------------------------------\n')

    if do_update:
        if "CUSTOM NODE PULL: Success" in output:
            process.wait()
            print(f"\rUpdated: {path}")
            return True, True    # updated
        elif "CUSTOM NODE PULL: None" in output:
            process.wait()
            return False, True   # there is no update
        else:
            print(f"\rUpdate error: {path}")
            process.wait()
            return False, False  # update failed
    else:
        if "CUSTOM NODE CHECK: True" in output:
            process.wait()
            return True, True
        elif "CUSTOM NODE CHECK: False" in output:
            process.wait()
            return False, True
        else:
            print(f"\rFetch error: {path}")
            print(f"\n{output}\n")
            process.wait()
            return False, True


def __win_check_git_pull(path):
    new_env = os.environ.copy()
    new_env["COMFYUI_PATH"] = comfy_path
    command = [sys.executable, git_script_path, "--pull", path]
    process = subprocess.Popen(command, env=new_env, cwd=custom_nodes_path)
    process.wait()


def execute_install_script(url, repo_path, lazy_mode=False, instant_execution=False):
    install_script_path = os.path.join(repo_path, "install.py")
    requirements_path = os.path.join(repo_path, "requirements.txt")

    if lazy_mode:
        install_cmd = ["#LAZY-INSTALL-SCRIPT",  sys.executable]
        try_install_script(url, repo_path, install_cmd)
    else:
        if os.path.exists(requirements_path):
            print("Install: pip packages")
            with open(requirements_path, "r") as requirements_file:
                for line in requirements_file:
                    package_name = remap_pip_package(line.strip())
                    if package_name and not package_name.startswith('#'):
                        install_cmd = [sys.executable, "-m", "pip", "install", package_name]
                        if package_name.strip() != "" and not package_name.startswith('#'):
                            try_install_script(url, repo_path, install_cmd, instant_execution=instant_execution)

        if os.path.exists(install_script_path):
            print(f"Install: install script")
            install_cmd = [sys.executable, "install.py"]
            try_install_script(url, repo_path, install_cmd, instant_execution=instant_execution)

    return True


def git_repo_has_updates(path, do_fetch=False, do_update=False):
    if do_fetch:
        print(f"\x1b[2K\rFetching: {path}", end='')
    elif do_update:
        print(f"\x1b[2K\rUpdating: {path}", end='')

    # Check if the path is a git repository
    if not os.path.exists(os.path.join(path, '.git')):
        raise ValueError('Not a git repository')

    if platform.system() == "Windows":
        updated, success = __win_check_git_update(path, do_fetch, do_update)
        if updated and success:
            execute_install_script(None, path, lazy_mode=True)
        return updated, success
    else:
        # Fetch the latest commits from the remote repository
        repo = git.Repo(path)

        current_branch = repo.active_branch
        branch_name = current_branch.name

        remote_name = 'origin'
        remote = repo.remote(name=remote_name)

        # Get the current commit hash
        commit_hash = repo.head.commit.hexsha

        if do_fetch or do_update:
            remote.fetch()

        if do_update:
            if repo.head.is_detached:
                switch_to_default_branch(repo)

            remote_commit_hash = repo.refs[f'{remote_name}/{branch_name}'].object.hexsha

            if commit_hash == remote_commit_hash:
                repo.close()
                return False, True

            try:
                remote.pull()
                repo.git.submodule('update', '--init', '--recursive')
                new_commit_hash = repo.head.commit.hexsha

                if commit_hash != new_commit_hash:
                    execute_install_script(None, path)
                    print(f"\x1b[2K\rUpdated: {path}")
                    return True, True
                else:
                    return False, False

            except Exception as e:
                print(f"\nUpdating failed: {path}\n{e}", file=sys.stderr)
                return False, False

        if repo.head.is_detached:
            repo.close()
            return True, True

        # Get commit hash of the remote branch
        current_branch = repo.active_branch
        branch_name = current_branch.name

        remote_commit_hash = repo.refs[f'{remote_name}/{branch_name}'].object.hexsha

        # Compare the commit hashes to determine if the local repository is behind the remote repository
        if commit_hash != remote_commit_hash:
            # Get the commit dates
            commit_date = repo.head.commit.committed_datetime
            remote_commit_date = repo.refs[f'{remote_name}/{branch_name}'].object.committed_datetime

            # Compare the commit dates to determine if the local repository is behind the remote repository
            if commit_date < remote_commit_date:
                repo.close()
                return True, True

        repo.close()

    return False, True


class GitProgress(RemoteProgress):
    def __init__(self):
        super().__init__()
        self.pbar = tqdm()

    def update(self, op_code, cur_count, max_count=None, message=''):
        self.pbar.total = max_count
        self.pbar.n = cur_count
        self.pbar.pos = 0
        self.pbar.refresh()


def is_valid_url(url):
    try:
        # Check for HTTP/HTTPS URL format
        result = urlparse(url)
        if all([result.scheme, result.netloc]):
            return True
    finally:
        # Check for SSH git URL format
        pattern = re.compile(r"^(.+@|ssh:\/\/).+:.+$")
        if pattern.match(url):
            return True
    return False


def gitclone_install(files, instant_execution=False, msg_prefix=''):
    print(f"{msg_prefix}Install: {files}")
    for url in files:
        if not is_valid_url(url):
            print(f"Invalid git url: '{url}'")
            return False

        if url.endswith("/"):
            url = url[:-1]
        try:
            print(f"Download: git clone '{url}'")
            repo_name = os.path.splitext(os.path.basename(url))[0]
            repo_path = os.path.join(custom_nodes_path, repo_name)

            # Clone the repository from the remote URL
            if not instant_execution and platform.system() == 'Windows':
                res = manager_funcs.run_script([sys.executable, git_script_path, "--clone", custom_nodes_path, url], cwd=custom_nodes_path)
                if res != 0:
                    return False
            else:
                repo = git.Repo.clone_from(url, repo_path, recursive=True, progress=GitProgress())
                repo.git.clear_cache()
                repo.close()

            if not execute_install_script(url, repo_path, instant_execution=instant_execution):
                return False

        except Exception as e:
            print(f"Install(git-clone) error: {url} / {e}", file=sys.stderr)
            return False

    print("Installation was successful.")
    return True


def git_pull(path):
    # Check if the path is a git repository
    if not os.path.exists(os.path.join(path, '.git')):
        raise ValueError('Not a git repository')

    # Pull the latest changes from the remote repository
    if platform.system() == "Windows":
        return __win_check_git_pull(path)
    else:
        repo = git.Repo(path)

        if repo.is_dirty():
            repo.git.stash()

        if repo.head.is_detached:
            switch_to_default_branch(repo)

        current_branch = repo.active_branch
        remote_name = current_branch.tracking_branch().remote_name
        remote = repo.remote(name=remote_name)

        remote.pull()
        repo.git.submodule('update', '--init', '--recursive')

        repo.close()

    return True


async def get_data(uri, silent=False):
    if not silent:
        print(f"FETCH DATA from: {uri}", end="")

    if uri.startswith("http"):
        async with aiohttp.ClientSession(trust_env=True, connector=aiohttp.TCPConnector(verify_ssl=False)) as session:
            async with session.get(uri) as resp:
                json_text = await resp.text()
    else:
        with cache_lock:
            with open(uri, "r", encoding="utf-8") as f:
                json_text = f.read()

    json_obj = json.loads(json_text)
    if not silent:
        print(f" [DONE]")
    return json_obj


def simple_hash(input_string):
    hash_value = 0
    for char in input_string:
        hash_value = (hash_value * 31 + ord(char)) % (2**32)

    return hash_value


def is_file_created_within_one_day(file_path):
    if not os.path.exists(file_path):
        return False

    file_creation_time = os.path.getctime(file_path)
    current_time = datetime.now().timestamp()
    time_difference = current_time - file_creation_time

    return time_difference <= 86400


async def get_data_by_mode(mode, filename, channel_url=None):
    if channel_url in get_channel_dict():
        channel_url = get_channel_dict()[channel_url]

    try:
        if mode == "local":
            uri = os.path.join(comfyui_manager_path, filename)
            json_obj = await get_data(uri)
        else:
            if channel_url is None:
                uri = get_config()['channel_url'] + '/' + filename
            else:
                uri = channel_url + '/' + filename

            cache_uri = str(simple_hash(uri))+'_'+filename
            cache_uri = os.path.join(cache_dir, cache_uri)

            if mode == "cache":
                if is_file_created_within_one_day(cache_uri):
                    json_obj = await get_data(cache_uri)
                else:
                    json_obj = await get_data(uri)
                    with cache_lock:
                        with open(cache_uri, "w", encoding='utf-8') as file:
                            json.dump(json_obj, file, indent=4, sort_keys=True)
            else:
                json_obj = await get_data(uri)
                with cache_lock:
                    with open(cache_uri, "w", encoding='utf-8') as file:
                        json.dump(json_obj, file, indent=4, sort_keys=True)
    except Exception as e:
        print(f"[ComfyUI-Manager] Due to a network error, switching to local mode.\n=> {filename}\n=> {e}")
        uri = os.path.join(comfyui_manager_path, filename)
        json_obj = await get_data(uri)

    return json_obj


def gitclone_fix(files, instant_execution=False):
    print(f"Try fixing: {files}")
    for url in files:
        if not is_valid_url(url):
            print(f"Invalid git url: '{url}'")
            return False

        if url.endswith("/"):
            url = url[:-1]
        try:
            repo_name = os.path.splitext(os.path.basename(url))[0]
            repo_path = os.path.join(custom_nodes_path, repo_name)

            if os.path.exists(repo_path+'.disabled'):
                repo_path = repo_path+'.disabled'

            if not execute_install_script(url, repo_path, instant_execution=instant_execution):
                return False

        except Exception as e:
            print(f"Install(git-clone) error: {url} / {e}", file=sys.stderr)
            return False

    print(f"Attempt to fixing '{files}' is done.")
    return True


def pip_install(packages):
    install_cmd = ['#FORCE', sys.executable, "-m", "pip", "install", '-U'] + packages
    try_install_script('pip install via manager', '..', install_cmd)


def rmtree(path):
    retry_count = 3

    while True:
        try:
            retry_count -= 1

            if platform.system() == "Windows":
                manager_funcs.run_script(['attrib', '-R', path + '\\*', '/S'])
            shutil.rmtree(path)

            return True

        except Exception as ex:
            print(f"ex: {ex}")
            time.sleep(3)

            if retry_count < 0:
                raise ex

            print(f"Uninstall retry({retry_count})")


def gitclone_uninstall(files):
    import os

    print(f"Uninstall: {files}")
    for url in files:
        if url.endswith("/"):
            url = url[:-1]
        try:
            dir_name = os.path.splitext(os.path.basename(url))[0].replace(".git", "")
            dir_path = os.path.join(custom_nodes_path, dir_name)

            # safety check
            if dir_path == '/' or dir_path[1:] == ":/" or dir_path == '':
                print(f"Uninstall(git-clone) error: invalid path '{dir_path}' for '{url}'")
                return False

            install_script_path = os.path.join(dir_path, "uninstall.py")
            disable_script_path = os.path.join(dir_path, "disable.py")
            if os.path.exists(install_script_path):
                uninstall_cmd = [sys.executable, "uninstall.py"]
                code = manager_funcs.run_script(uninstall_cmd, cwd=dir_path)

                if code != 0:
                    print(f"An error occurred during the execution of the uninstall.py script. Only the '{dir_path}' will be deleted.")
            elif os.path.exists(disable_script_path):
                disable_script = [sys.executable, "disable.py"]
                code = manager_funcs.run_script(disable_script, cwd=dir_path)
                if code != 0:
                    print(f"An error occurred during the execution of the disable.py script. Only the '{dir_path}' will be deleted.")

            if os.path.exists(dir_path):
                rmtree(dir_path)
            elif os.path.exists(dir_path + ".disabled"):
                rmtree(dir_path + ".disabled")
        except Exception as e:
            print(f"Uninstall(git-clone) error: {url} / {e}", file=sys.stderr)
            return False

    print("Uninstallation was successful.")
    return True


def gitclone_set_active(files, is_disable):
    import os

    if is_disable:
        action_name = "Disable"
    else:
        action_name = "Enable"

    print(f"{action_name}: {files}")
    for url in files:
        if url.endswith("/"):
            url = url[:-1]
        try:
            dir_name = os.path.splitext(os.path.basename(url))[0].replace(".git", "")
            dir_path = os.path.join(custom_nodes_path, dir_name)

            # safety check
            if dir_path == '/' or dir_path[1:] == ":/" or dir_path == '':
                print(f"{action_name}(git-clone) error: invalid path '{dir_path}' for '{url}'")
                return False

            if is_disable:
                current_path = dir_path
                new_path = dir_path + ".disabled"
            else:
                current_path = dir_path + ".disabled"
                new_path = dir_path

            os.rename(current_path, new_path)

            if is_disable:
                if os.path.exists(os.path.join(new_path, "disable.py")):
                    disable_script = [sys.executable, "disable.py"]
                    try_install_script(url, new_path, disable_script)
            else:
                if os.path.exists(os.path.join(new_path, "enable.py")):
                    enable_script = [sys.executable, "enable.py"]
                    try_install_script(url, new_path, enable_script)

        except Exception as e:
            print(f"{action_name}(git-clone) error: {url} / {e}", file=sys.stderr)
            return False

    print(f"{action_name} was successful.")
    return True


def gitclone_update(files, instant_execution=False, skip_script=False, msg_prefix=""):
    import os

    print(f"{msg_prefix}Update: {files}")
    for url in files:
        if url.endswith("/"):
            url = url[:-1]
        try:
            repo_name = os.path.splitext(os.path.basename(url))[0].replace(".git", "")
            repo_path = os.path.join(custom_nodes_path, repo_name)

            if os.path.exists(repo_path+'.disabled'):
                repo_path = repo_path+'.disabled'

            git_pull(repo_path)

            if not skip_script:
                if instant_execution:
                    if not execute_install_script(url, repo_path, lazy_mode=False, instant_execution=True):
                        return False
                else:
                    if not execute_install_script(url, repo_path, lazy_mode=True):
                        return False

        except Exception as e:
            print(f"Update(git-clone) error: {url} / {e}", file=sys.stderr)
            return False

    if not skip_script:
        print("Update was successful.")
    return True


def update_path(repo_path, instant_execution=False):
    if not os.path.exists(os.path.join(repo_path, '.git')):
        return "fail"

    # version check
    repo = git.Repo(repo_path)

    if repo.head.is_detached:
        switch_to_default_branch(repo)

    current_branch = repo.active_branch
    branch_name = current_branch.name

    if current_branch.tracking_branch() is None:
        print(f"[ComfyUI-Manager] There is no tracking branch ({current_branch})")
        remote_name = 'origin'
    else:
        remote_name = current_branch.tracking_branch().remote_name
    remote = repo.remote(name=remote_name)

    try:
        remote.fetch()
    except Exception as e:
        if 'detected dubious' in str(e):
            print(f"[ComfyUI-Manager] Try fixing 'dubious repository' error on 'ComfyUI' repository")
            safedir_path = comfy_path.replace('\\', '/')
            subprocess.run(['git', 'config', '--global', '--add', 'safe.directory', safedir_path])
            try:
                remote.fetch()
            except Exception:
                print(f"\n[ComfyUI-Manager] Failed to fixing repository setup. Please execute this command on cmd: \n"
                      f"-----------------------------------------------------------------------------------------\n"
                      f'git config --global --add safe.directory "{safedir_path}"\n'
                      f"-----------------------------------------------------------------------------------------\n")

    commit_hash = repo.head.commit.hexsha
    remote_commit_hash = repo.refs[f'{remote_name}/{branch_name}'].object.hexsha

    if commit_hash != remote_commit_hash:
        git_pull(repo_path)
        execute_install_script("ComfyUI", repo_path, instant_execution=instant_execution)
        return "updated"
    else:
        return "skipped"


def lookup_customnode_by_url(data, target):
    for x in data['custom_nodes']:
        if target in x['files']:
            dir_name = os.path.splitext(os.path.basename(target))[0].replace(".git", "")
            dir_path = os.path.join(custom_nodes_path, dir_name)
            if os.path.exists(dir_path):
                x['installed'] = 'True'
            elif os.path.exists(dir_path + ".disabled"):
                x['installed'] = 'Disabled'
            return x

    return None


def simple_check_custom_node(url):
    dir_name = os.path.splitext(os.path.basename(url))[0].replace(".git", "")
    dir_path = os.path.join(custom_nodes_path, dir_name)
    if os.path.exists(dir_path):
        return 'installed'
    elif os.path.exists(dir_path+'.disabled'):
        return 'disabled'

    return 'not-installed'


def check_a_custom_node_installed(item, do_fetch=False, do_update_check=True, do_update=False):
    item['installed'] = 'None'

    if item['install_type'] == 'git-clone' and len(item['files']) == 1:
        url = item['files'][0]

        if url.endswith("/"):
            url = url[:-1]

        dir_name = os.path.splitext(os.path.basename(url))[0].replace(".git", "")
        dir_path = os.path.join(custom_nodes_path, dir_name)
        if os.path.exists(dir_path):
            try:
                item['installed'] = 'True'  # default

                if cm_global.try_call(api="cm.is_import_failed_extension", name=dir_name):
                    item['installed'] = 'Fail'

                if do_update_check:
                    update_state, success = git_repo_has_updates(dir_path, do_fetch, do_update)
                    if (do_update_check or do_update) and update_state:
                        item['installed'] = 'Update'
                    elif do_update and not success:
                        item['installed'] = 'Fail'
            except:
                if cm_global.try_call(api="cm.is_import_failed_extension", name=dir_name):
                    item['installed'] = 'Fail'
                else:
                    item['installed'] = 'True'

        elif os.path.exists(dir_path + ".disabled"):
            item['installed'] = 'Disabled'

        else:
            item['installed'] = 'False'

    elif item['install_type'] == 'copy' and len(item['files']) == 1:
        dir_name = os.path.basename(item['files'][0])

        if item['files'][0].endswith('.py'):
            base_path = custom_nodes_path
        elif 'js_path' in item:
            base_path = os.path.join(js_path, item['js_path'])
        else:
            base_path = js_path

        file_path = os.path.join(base_path, dir_name)
        if os.path.exists(file_path):
            if cm_global.try_call(api="cm.is_import_failed_extension", name=dir_name):
                item['installed'] = 'Fail'
            else:
                item['installed'] = 'True'
        elif os.path.exists(file_path + ".disabled"):
            item['installed'] = 'Disabled'
        else:
            item['installed'] = 'False'


def get_installed_pip_packages():
    # extract pip package infos
    pips = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True).split('\n')

    res = {}
    for x in pips:
        if x.strip() == "":
            continue

        if ' @ ' in x:
            spec_url = x.split(' @ ')
            res[spec_url[0]] = spec_url[1]
        else:
            res[x] = ""

    return res


def get_current_snapshot():
    # Get ComfyUI hash
    repo_path = comfy_path

    if not os.path.exists(os.path.join(repo_path, '.git')):
        print(f"ComfyUI update fail: The installed ComfyUI does not have a Git repository.")
        return {}

    repo = git.Repo(repo_path)
    comfyui_commit_hash = repo.head.commit.hexsha

    git_custom_nodes = {}
    file_custom_nodes = []

    # Get custom nodes hash
    for path in os.listdir(custom_nodes_path):
        fullpath = os.path.join(custom_nodes_path, path)

        if os.path.isdir(fullpath):
            is_disabled = path.endswith(".disabled")

            try:
                git_dir = os.path.join(fullpath, '.git')

                if not os.path.exists(git_dir):
                    continue

                repo = git.Repo(fullpath)
                commit_hash = repo.head.commit.hexsha
                url = repo.remotes.origin.url
                git_custom_nodes[url] = {
                    'hash': commit_hash,
                    'disabled': is_disabled
                }

            except:
                print(f"Failed to extract snapshots for the custom node '{path}'.")

        elif path.endswith('.py'):
            is_disabled = path.endswith(".py.disabled")
            filename = os.path.basename(path)
            item = {
                'filename': filename,
                'disabled': is_disabled
            }

            file_custom_nodes.append(item)

    pip_packages = get_installed_pip_packages()

    return {
        'comfyui': comfyui_commit_hash,
        'git_custom_nodes': git_custom_nodes,
        'file_custom_nodes': file_custom_nodes,
        'pips': pip_packages,
    }


def save_snapshot_with_postfix(postfix, path=None):
    if path is None:
        now = datetime.now()

        date_time_format = now.strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"{date_time_format}_{postfix}"

        path = os.path.join(comfyui_manager_path, 'snapshots', f"{file_name}.json")
    else:
        file_name = path.replace('\\', '/').split('/')[-1]
        file_name = file_name.split('.')[-2]

    snapshot = get_current_snapshot()
    if path.endswith('.json'):
        with open(path, "w") as json_file:
            json.dump(snapshot, json_file, indent=4)

        return file_name + '.json'

    elif path.endswith('.yaml'):
        with open(path, "w") as yaml_file:
            snapshot = {'custom_nodes': snapshot}
            yaml.dump(snapshot, yaml_file, allow_unicode=True)

        return path


async def extract_nodes_from_workflow(filepath, mode='local', channel_url='default'):
    # prepare json data
    workflow = None
    if filepath.endswith('.json'):
        with open(filepath, "r", encoding="UTF-8", errors="ignore") as json_file:
            try:
                workflow = json.load(json_file)
            except:
                print(f"Invalid workflow file: {filepath}")
                exit(-1)

    elif filepath.endswith('.png'):
        from PIL import Image
        with Image.open(filepath) as img:
            if 'workflow' not in img.info:
                print(f"The specified .png file doesn't have a workflow: {filepath}")
                exit(-1)
            else:
                try:
                    workflow = json.loads(img.info['workflow'])
                except:
                    print(f"This is not a valid .png file containing a ComfyUI workflow: {filepath}")
                    exit(-1)

    if workflow is None:
        print(f"Invalid workflow file: {filepath}")
        exit(-1)

    # extract nodes
    used_nodes = set()

    def extract_nodes(sub_workflow):
        for x in sub_workflow['nodes']:
            node_name = x.get('type')

            # skip virtual nodes
            if node_name in ['Reroute', 'Note']:
                continue

            if node_name is not None and not node_name.startswith('workflow/'):
                used_nodes.add(node_name)

    if 'nodes' in workflow:
        extract_nodes(workflow)

        if 'extra' in workflow:
            if 'groupNodes' in workflow['extra']:
                for x in workflow['extra']['groupNodes'].values():
                    extract_nodes(x)

    # lookup dependent custom nodes
    ext_map = await get_data_by_mode(mode, 'extension-node-map.json', channel_url)

    rext_map = {}
    preemption_map = {}
    patterns = []
    for k, v in ext_map.items():
        if k == 'https://github.com/comfyanonymous/ComfyUI':
            for x in v[0]:
                if x not in preemption_map:
                    preemption_map[x] = []

                preemption_map[x] = k
            continue

        for x in v[0]:
            if x not in rext_map:
                rext_map[x] = []

            rext_map[x].append(k)

        if 'preemptions' in v[1]:
            for x in v[1]['preemptions']:
                if x not in preemption_map:
                    preemption_map[x] = []

                preemption_map[x] = k

        if 'nodename_pattern' in v[1]:
            patterns.append((v[1]['nodename_pattern'], k))

    # identify used extensions
    used_exts = set()
    unknown_nodes = set()

    for node_name in used_nodes:
        ext = preemption_map.get(node_name)

        if ext is None:
            ext = rext_map.get(node_name)
            if ext is not None:
                ext = ext[0]

        if ext is None:
            for pat_ext in patterns:
                if re.search(pat_ext[0], node_name):
                    ext = pat_ext[1]
                    break

        if ext == 'https://github.com/comfyanonymous/ComfyUI':
            pass
        elif ext is not None:
            if 'Fooocus' in ext:
                print(f">> {node_name}")

            used_exts.add(ext)
        else:
            unknown_nodes.add(node_name)

    return used_exts, unknown_nodes


def unzip(model_path):
    if not os.path.exists(model_path):
        print(f"[ComfyUI-Manager] unzip: File not found: {model_path}")
        return False

    base_dir = os.path.dirname(model_path)
    filename = os.path.basename(model_path)
    target_dir = os.path.join(base_dir, filename[:-4])

    os.makedirs(target_dir, exist_ok=True)

    with zipfile.ZipFile(model_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)

    # Check if there's only one directory inside the target directory
    contents = os.listdir(target_dir)
    if len(contents) == 1 and os.path.isdir(os.path.join(target_dir, contents[0])):
        nested_dir = os.path.join(target_dir, contents[0])
        # Move each file and sub-directory in the nested directory up to the target directory
        for item in os.listdir(nested_dir):
            shutil.move(os.path.join(nested_dir, item), os.path.join(target_dir, item))
        # Remove the now empty nested directory
        os.rmdir(nested_dir)

    os.remove(model_path)
    return True

