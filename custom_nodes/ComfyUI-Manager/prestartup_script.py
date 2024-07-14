import datetime
import os
import subprocess
import sys
import atexit
import threading
import re
import locale
import platform
import json
import ast
import logging

glob_path = os.path.join(os.path.dirname(__file__), "glob")
sys.path.append(glob_path)

import security_check
from manager_util import *
import cm_global

security_check.security_check()

cm_global.pip_downgrade_blacklist = ['torch', 'torchsde', 'torchvision', 'transformers', 'safetensors', 'kornia']


def skip_pip_spam(x):
    return ('Requirement already satisfied:' in x) or ("DEPRECATION: Loading egg at" in x)


message_collapses = [skip_pip_spam]
import_failed_extensions = set()
cm_global.variables['cm.on_revision_detected_handler'] = []
enable_file_logging = True


def register_message_collapse(f):
    global message_collapses
    message_collapses.append(f)


def is_import_failed_extension(name):
    global import_failed_extensions
    return name in import_failed_extensions


def check_file_logging():
    global enable_file_logging
    try:
        import configparser
        config_path = os.path.join(os.path.dirname(__file__), "config.ini")
        config = configparser.ConfigParser()
        config.read(config_path)
        default_conf = config['default']

        if 'file_logging' in default_conf and default_conf['file_logging'].lower() == 'false':
            enable_file_logging = False
    except Exception:
        pass


check_file_logging()

comfy_path = os.environ.get('COMFYUI_PATH')
if comfy_path is None:
    comfy_path = os.path.abspath(os.path.dirname(sys.modules['__main__'].__file__))

sys.__comfyui_manager_register_message_collapse = register_message_collapse
sys.__comfyui_manager_is_import_failed_extension = is_import_failed_extension
cm_global.register_api('cm.register_message_collapse', register_message_collapse)
cm_global.register_api('cm.is_import_failed_extension', is_import_failed_extension)


comfyui_manager_path = os.path.dirname(__file__)
custom_nodes_path = os.path.abspath(os.path.join(comfyui_manager_path, ".."))
startup_script_path = os.path.join(comfyui_manager_path, "startup-scripts")
restore_snapshot_path = os.path.join(startup_script_path, "restore-snapshot.json")
git_script_path = os.path.join(comfyui_manager_path, "git_helper.py")
pip_overrides_path = os.path.join(comfyui_manager_path, "pip_overrides.json")


cm_global.pip_overrides = {}
if os.path.exists(pip_overrides_path):
    with open(pip_overrides_path, 'r', encoding="UTF-8", errors="ignore") as json_file:
        cm_global.pip_overrides = json.load(json_file)


def remap_pip_package(pkg):
    if pkg in cm_global.pip_overrides:
        res = cm_global.pip_overrides[pkg]
        print(f"[ComfyUI-Manager] '{pkg}' is remapped to '{res}'")
        return res
    else:
        return pkg


std_log_lock = threading.Lock()


class TerminalHook:
    def __init__(self):
        self.hooks = {}

    def add_hook(self, k, v):
        self.hooks[k] = v

    def remove_hook(self, k):
        if k in self.hooks:
            del self.hooks[k]

    def write_stderr(self, msg):
        for v in self.hooks.values():
            try:
                v.write_stderr(msg)
            except Exception:
                pass

    def write_stdout(self, msg):
        for v in self.hooks.values():
            try:
                v.write_stdout(msg)
            except Exception:
                pass


terminal_hook = TerminalHook()
sys.__comfyui_manager_terminal_hook = terminal_hook


def handle_stream(stream, prefix):
    stream.reconfigure(encoding=locale.getpreferredencoding(), errors='replace')
    for msg in stream:
        if prefix == '[!]' and ('it/s]' in msg or 's/it]' in msg) and ('%|' in msg or 'it [' in msg):
            if msg.startswith('100%'):
                print('\r' + msg, end="", file=sys.stderr),
            else:
                print('\r' + msg[:-1], end="", file=sys.stderr),
        else:
            if prefix == '[!]':
                print(prefix, msg, end="", file=sys.stderr)
            else:
                print(prefix, msg, end="")


def process_wrap(cmd_str, cwd_path, handler=None, env=None):
    process = subprocess.Popen(cmd_str, cwd=cwd_path, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

    if handler is None:
        handler = handle_stream

    stdout_thread = threading.Thread(target=handler, args=(process.stdout, ""))
    stderr_thread = threading.Thread(target=handler, args=(process.stderr, "[!]"))

    stdout_thread.start()
    stderr_thread.start()

    stdout_thread.join()
    stderr_thread.join()

    return process.wait()


try:
    if '--port' in sys.argv:
        port_index = sys.argv.index('--port')
        if port_index + 1 < len(sys.argv):
            port = int(sys.argv[port_index + 1])
            postfix = f"_{port}"
        else:
            postfix = ""
    else:
        postfix = ""

    # Logger setup
    if enable_file_logging:
        if os.path.exists(f"comfyui{postfix}.log"):
            if os.path.exists(f"comfyui{postfix}.prev.log"):
                if os.path.exists(f"comfyui{postfix}.prev2.log"):
                    os.remove(f"comfyui{postfix}.prev2.log")
                os.rename(f"comfyui{postfix}.prev.log", f"comfyui{postfix}.prev2.log")
            os.rename(f"comfyui{postfix}.log", f"comfyui{postfix}.prev.log")

        log_file = open(f"comfyui{postfix}.log", "w", encoding="utf-8", errors="ignore")

    log_lock = threading.Lock()

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    if original_stdout.encoding.lower() == 'utf-8':
        write_stdout = original_stdout.write
        write_stderr = original_stderr.write
    else:
        def wrapper_stdout(msg):
            original_stdout.write(msg.encode('utf-8').decode(original_stdout.encoding, errors="ignore"))
            
        def wrapper_stderr(msg):
            original_stderr.write(msg.encode('utf-8').decode(original_stderr.encoding, errors="ignore"))

        write_stdout = wrapper_stdout
        write_stderr = wrapper_stderr

    pat_tqdm = r'\d+%.*\[(.*?)\]'
    pat_import_fail = r'seconds \(IMPORT FAILED\):.*[/\\]custom_nodes[/\\](.*)$'

    is_start_mode = True


    class ComfyUIManagerLogger:
        def __init__(self, is_stdout):
            self.is_stdout = is_stdout
            self.encoding = "utf-8"
            self.last_char = ''

        def fileno(self):
            try:
                if self.is_stdout:
                    return original_stdout.fileno()
                else:
                    return original_stderr.fileno()
            except AttributeError:
                # Handle error
                raise ValueError("The object does not have a fileno method")

        def isatty(self):
            return False

        def write(self, message):
            global is_start_mode

            if any(f(message) for f in message_collapses):
                return

            if is_start_mode:
                match = re.search(pat_import_fail, message)
                if match:
                    import_failed_extensions.add(match.group(1))

                if 'Starting server' in message:
                    is_start_mode = False

            if not self.is_stdout:
                match = re.search(pat_tqdm, message)
                if match:
                    message = re.sub(r'([#|])\d', r'\1▌', message)
                    message = re.sub('#', '█', message)
                    if '100%' in message:
                        self.sync_write(message)
                    else:
                        write_stderr(message)
                        original_stderr.flush()
                else:
                    self.sync_write(message)
            else:
                self.sync_write(message)

        def sync_write(self, message, file_only=False):
            with log_lock:
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')[:-3]
                if self.last_char != '\n':
                    log_file.write(message)
                else:
                    log_file.write(f"[{timestamp}] {message}")
                log_file.flush()
                self.last_char = message if message == '' else message[-1]

            if not file_only:
                with std_log_lock:
                    if self.is_stdout:
                        write_stdout(message)
                        original_stdout.flush()
                        terminal_hook.write_stderr(message)
                    else:
                        write_stderr(message)
                        original_stderr.flush()
                        terminal_hook.write_stdout(message)

        def flush(self):
            log_file.flush()

            with std_log_lock:
                if self.is_stdout:
                    original_stdout.flush()
                else:
                    original_stderr.flush()

        def close(self):
            self.flush()

        def reconfigure(self, *args, **kwargs):
            pass

        # You can close through sys.stderr.close_log()
        def close_log(self):
            sys.stderr = original_stderr
            sys.stdout = original_stdout
            log_file.close()
            
    def close_log():
        sys.stderr = original_stderr
        sys.stdout = original_stdout
        log_file.close()


    if enable_file_logging:
        sys.stdout = ComfyUIManagerLogger(True)
        stderr_wrapper = ComfyUIManagerLogger(False)
        sys.stderr = stderr_wrapper

        atexit.register(close_log)
    else:
        sys.stdout.close_log = lambda: None
        stderr_wrapper = None


    class LoggingHandler(logging.Handler):
        def emit(self, record):
            global is_start_mode

            message = record.getMessage()

            if is_start_mode:
                match = re.search(pat_import_fail, message)
                if match:
                    import_failed_extensions.add(match.group(1))

                if 'Starting server' in message:
                    is_start_mode = False

            if stderr_wrapper:
                stderr_wrapper.sync_write(message+'\n', file_only=True)


    logging.getLogger().addHandler(LoggingHandler())


except Exception as e:
    print(f"[ComfyUI-Manager] Logging failed: {e}")


try:
    import git
except ModuleNotFoundError:
    my_path = os.path.dirname(__file__)
    requirements_path = os.path.join(my_path, "requirements.txt")

    print(f"## ComfyUI-Manager: installing dependencies. (GitPython)")
    try:
        result = subprocess.check_output([sys.executable, '-s', '-m', 'pip', 'install', '-r', requirements_path])
    except subprocess.CalledProcessError as e:
        print(f"## [ERROR] ComfyUI-Manager: Attempting to reinstall dependencies using an alternative method.")
        try:
            result = subprocess.check_output([sys.executable, '-s', '-m', 'pip', 'install', '--user', '-r', requirements_path])
        except subprocess.CalledProcessError as e:
            print(f"## [ERROR] ComfyUI-Manager: Failed to install the GitPython package in the correct Python environment. Please install it manually in the appropriate environment. (You can seek help at https://app.element.io/#/room/%23comfyui_space%3Amatrix.org)")

try:
    import git
    print(f"## ComfyUI-Manager: installing dependencies done.")
except:
    # maybe we should sys.exit() here? there is at least two screens worth of error messages still being pumped after our error messages
    print(f"## [ERROR] ComfyUI-Manager: GitPython package seems to be installed, but failed to load somehow. Make sure you have a working git client installed")


print("** ComfyUI startup time:", datetime.datetime.now())
print("** Platform:", platform.system())
print("** Python version:", sys.version)
print("** Python executable:", sys.executable)
print("** ComfyUI Path:", comfy_path)

if enable_file_logging:
    print("** Log path:", os.path.abspath('comfyui.log'))
else:
    print("** Log path: file logging is disabled")


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


read_downgrade_blacklist()


def check_bypass_ssl():
    try:
        import configparser
        import ssl
        config_path = os.path.join(os.path.dirname(__file__), "config.ini")
        config = configparser.ConfigParser()
        config.read(config_path)
        default_conf = config['default']

        if 'bypass_ssl' in default_conf and default_conf['bypass_ssl'].lower() == 'true':
            print(f"[ComfyUI-Manager] WARN: Unsafe - SSL verification bypass option is Enabled. (see ComfyUI-Manager/config.ini)")
            ssl._create_default_https_context = ssl._create_unverified_context  # SSL certificate error fix.
    except Exception:
        pass


check_bypass_ssl()


# Perform install
processed_install = set()
script_list_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "startup-scripts", "install-scripts.txt")
pip_map = None


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

    pkg = get_installed_packages().get(name.lower())
    if pkg is None:
        return False  # update if not installed

    if match is None:
        return True   # don't update if version is not specified

    if match.group(2) in ['>', '>=']:
        if StrictVersion(pkg) < StrictVersion(match.group(3)):
            return False
        elif StrictVersion(pkg) > StrictVersion(match.group(3)):
            print(f"[SKIP] Downgrading pip package isn't allowed: {name.lower()} (cur={pkg})")

    return True       # prevent downgrade


if os.path.exists(restore_snapshot_path):
    try:
        cloned_repos = []

        def msg_capture(stream, prefix):
            stream.reconfigure(encoding=locale.getpreferredencoding(), errors='replace')
            for msg in stream:
                if msg.startswith("CLONE: "):
                    cloned_repos.append(msg[7:])
                    if prefix == '[!]':
                        print(prefix, msg, end="", file=sys.stderr)
                    else:
                        print(prefix, msg, end="")

                elif prefix == '[!]' and ('it/s]' in msg or 's/it]' in msg) and ('%|' in msg or 'it [' in msg):
                    if msg.startswith('100%'):
                        print('\r' + msg, end="", file=sys.stderr),
                    else:
                        print('\r'+msg[:-1], end="", file=sys.stderr),
                else:
                    if prefix == '[!]':
                        print(prefix, msg, end="", file=sys.stderr)
                    else:
                        print(prefix, msg, end="")

        print(f"[ComfyUI-Manager] Restore snapshot.")
        cmd_str = [sys.executable, git_script_path, '--apply-snapshot', restore_snapshot_path]

        new_env = os.environ.copy()
        new_env["COMFYUI_PATH"] = comfy_path
        exit_code = process_wrap(cmd_str, custom_nodes_path, handler=msg_capture, env=new_env)

        repository_name = ''
        for url in cloned_repos:
            try:
                repository_name = url.split("/")[-1].strip()
                repo_path = os.path.join(custom_nodes_path, repository_name)
                repo_path = os.path.abspath(repo_path)

                requirements_path = os.path.join(repo_path, 'requirements.txt')
                install_script_path = os.path.join(repo_path, 'install.py')

                this_exit_code = 0

                if os.path.exists(requirements_path):
                    with open(requirements_path, 'r', encoding="UTF-8", errors="ignore") as file:
                        for line in file:
                            package_name = remap_pip_package(line.strip())
                            if package_name and not is_installed(package_name):
                                if not package_name.startswith('#'):
                                    install_cmd = [sys.executable, "-m", "pip", "install", package_name]
                                    this_exit_code += process_wrap(install_cmd, repo_path)

                if os.path.exists(install_script_path) and f'{repo_path}/install.py' not in processed_install:
                    processed_install.add(f'{repo_path}/install.py')
                    install_cmd = [sys.executable, install_script_path]
                    print(f">>> {install_cmd} / {repo_path}")

                    new_env = os.environ.copy()
                    new_env["COMFYUI_PATH"] = comfy_path
                    this_exit_code += process_wrap(install_cmd, repo_path, env=new_env)

                if this_exit_code != 0:
                    print(f"[ComfyUI-Manager] Restoring '{repository_name}' is failed.")

            except Exception as e:
                print(e)
                print(f"[ComfyUI-Manager] Restoring '{repository_name}' is failed.")

        if exit_code != 0:
            print(f"[ComfyUI-Manager] Restore snapshot failed.")
        else:
            print(f"[ComfyUI-Manager] Restore snapshot done.")

    except Exception as e:
        print(e)
        print(f"[ComfyUI-Manager] Restore snapshot failed.")

    os.remove(restore_snapshot_path)


def execute_lazy_install_script(repo_path, executable):
    global processed_install

    install_script_path = os.path.join(repo_path, "install.py")
    requirements_path = os.path.join(repo_path, "requirements.txt")

    if os.path.exists(requirements_path):
        print(f"Install: pip packages for '{repo_path}'")
        with open(requirements_path, "r") as requirements_file:
            for line in requirements_file:
                package_name = remap_pip_package(line.strip())
                if package_name and not is_installed(package_name):
                    install_cmd = [executable, "-m", "pip", "install", package_name]
                    process_wrap(install_cmd, repo_path)

    if os.path.exists(install_script_path) and f'{repo_path}/install.py' not in processed_install:
        processed_install.add(f'{repo_path}/install.py')
        print(f"Install: install script for '{repo_path}'")
        install_cmd = [executable, "install.py"]

        new_env = os.environ.copy()
        new_env["COMFYUI_PATH"] = comfy_path
        process_wrap(install_cmd, repo_path, env=new_env)


# Check if script_list_path exists
if os.path.exists(script_list_path):
    print("\n#######################################################################")
    print("[ComfyUI-Manager] Starting dependency installation/(de)activation for the extension\n")

    executed = set()
    # Read each line from the file and convert it to a list using eval
    with open(script_list_path, 'r', encoding="UTF-8", errors="ignore") as file:
        for line in file:
            if line in executed:
                continue

            executed.add(line)

            try:
                script = ast.literal_eval(line)

                if script[1].startswith('#') and script[1] != '#FORCE':
                    if script[1] == "#LAZY-INSTALL-SCRIPT":
                        execute_lazy_install_script(script[0], script[2])

                elif os.path.exists(script[0]):
                    if script[1] == "#FORCE":
                        del script[1]
                    else:
                        if 'pip' in script[1:] and 'install' in script[1:] and is_installed(script[-1]):
                            continue

                    print(f"\n## ComfyUI-Manager: EXECUTE => {script[1:]}")
                    print(f"\n## Execute install/(de)activation script for '{script[0]}'")

                    new_env = os.environ.copy()
                    new_env["COMFYUI_PATH"] = comfy_path
                    exit_code = process_wrap(script[1:], script[0], env=new_env)

                    if exit_code != 0:
                        print(f"install/(de)activation script failed: {script[0]}")
                else:
                    print(f"\n## ComfyUI-Manager: CANCELED => {script[1:]}")

            except Exception as e:
                print(f"[ERROR] Failed to execute install/(de)activation script: {line} / {e}")

    # Remove the script_list_path file
    if os.path.exists(script_list_path):
        os.remove(script_list_path)
        
    print("\n[ComfyUI-Manager] Startup script completed.")
    print("#######################################################################\n")

del processed_install
del pip_map


def check_windows_event_loop_policy():
    try:
        import configparser
        config_path = os.path.join(os.path.dirname(__file__), "config.ini")
        config = configparser.ConfigParser()
        config.read(config_path)
        default_conf = config['default']

        if 'windows_selector_event_loop_policy' in default_conf and default_conf['windows_selector_event_loop_policy'].lower() == 'true':
            try:
                import asyncio
                import asyncio.windows_events
                asyncio.set_event_loop_policy(asyncio.windows_events.WindowsSelectorEventLoopPolicy())
                print(f"[ComfyUI-Manager] Windows event loop policy mode enabled")
            except Exception as e:
                print(f"[ComfyUI-Manager] WARN: Windows initialization fail: {e}")
    except Exception:
        pass


if platform.system() == 'Windows':
    check_windows_event_loop_policy()
