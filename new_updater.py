import logging
import os
import shutil

base_path = os.path.dirname(os.path.realpath(__file__))

# A user-edited bat file may start with `@echo off`, a UTF-8 BOM, or extra
# whitespace before the canonical update command. Strip those prefixes before
# the signature check so we still recognise the file as the legacy format and
# can hand it the latest contents.
_LEGACY_UPDATER_SIGNATURE = b"..\\python_embeded\\python.exe .\\update.py"
_UTF8_BOM = b"\xef\xbb\xbf"


def _looks_like_legacy_updater_bat(contents: bytes) -> bool:
    if contents.startswith(_UTF8_BOM):
        contents = contents[len(_UTF8_BOM):]
    contents = contents.lstrip()
    if contents.lower().startswith(b"@echo off"):
        contents = contents[len(b"@echo off"):].lstrip()
    return contents.startswith(_LEGACY_UPDATER_SIGNATURE)


def update_windows_updater():
    top_path = os.path.dirname(base_path)
    updater_path = os.path.join(base_path, ".ci/update_windows/update.py")
    bat_path = os.path.join(base_path, ".ci/update_windows/update_comfyui.bat")

    dest_updater_path = os.path.join(top_path, "update/update.py")
    dest_bat_path = os.path.join(top_path, "update/update_comfyui.bat")
    dest_bat_deps_path = os.path.join(top_path, "update/update_comfyui_and_python_dependencies.bat")

    try:
        with open(dest_bat_path, 'rb') as f:
            contents = f.read()
    except FileNotFoundError:
        return
    except OSError as e:
        logging.warning("Could not read %s: %s", dest_bat_path, e)
        return

    if not _looks_like_legacy_updater_bat(contents):
        return

    try:
        shutil.copy(updater_path, dest_updater_path)
    except OSError as e:
        logging.warning("Failed to update %s: %s", dest_updater_path, e)
        return

    try:
        with open(dest_bat_deps_path, 'rb') as f:
            contents = f.read()
        contents = contents.replace(b'..\\python_embeded\\python.exe .\\update.py ..\\ComfyUI\\', b'call update_comfyui.bat nopause')
        with open(dest_bat_deps_path, 'wb') as f:
            f.write(contents)
    except FileNotFoundError:
        pass
    except OSError as e:
        logging.warning("Failed to update %s: %s", dest_bat_deps_path, e)

    try:
        shutil.copy(bat_path, dest_bat_path)
    except OSError as e:
        logging.warning("Failed to update %s: %s", dest_bat_path, e)
        return
    print("Updated the windows standalone package updater.")  # noqa: T201
