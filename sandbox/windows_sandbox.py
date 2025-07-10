import logging
import win32con
import win32process
import win32security
import subprocess
import os
from win32com.shell import shellcon, shell
import win32api
import win32event
import folder_paths


LOW_INTEGRITY_SID_STRING = "S-1-16-4096"

# Use absolute path to prevent command injection
ICACLS_PATH = r"C:\Windows\System32\icacls.exe"


def set_process_integrity_level_to_low():
    current_process = win32process.GetCurrentProcess()
    token = win32security.OpenProcessToken(
        current_process,
        win32con.TOKEN_ALL_ACCESS,
    )

    low_integrity_sid = win32security.ConvertStringSidToSid(LOW_INTEGRITY_SID_STRING)
    win32security.SetTokenInformation(
        token, win32security.TokenIntegrityLevel, (low_integrity_sid, 0)
    )

    logging.info("Sandbox enabled: Process now running with low integrity token")

    win32api.CloseHandle(token)


def does_permit_low_integrity_write(icacls_output):
    """
    Checks if an icacls output indicates that the path is writable by low
    integrity processes.

    Note that currently it is a bit of a crude check - it is possible for
    a low integrity process to have write access to a directory without
    having these exact ACLs reported by icacls. Implement a more robust
    check if this situation ever occurs.
    """
    permissions = [l.strip() for l in icacls_output.split("\n")]
    LOW_INTEGRITY_LABEL = r"Mandatory Label\Low Mandatory Level"

    for p in permissions:
        if LOW_INTEGRITY_LABEL not in p:
            continue

        # Check the Low integrity label line - it should be something like
        # Mandatory Label\Low Mandatory Level:(OI)(CI)(NW) or
        # Mandatory Label\Low Mandatory Level:(I)(OI)(CI)(NW)
        return all(
            [
                # OI: Object Inheritance - all files in the directory with have low
                # integrity
                "(OI)" in p,
                # CI: Container Inheritance - all subdirectories will have low
                # integrity
                "(CI)" in p,
                # NW: No Writeup - processes with lower integrity cannot write to
                # this directory
                "(NW)" in p,
            ]
        )


def path_is_low_integrity_writable(path):
    """Check if the path has a writable ACL by low integrity process"""
    result = subprocess.run([ICACLS_PATH, path], capture_output=True, text=True)

    if result.returncode != 0:
        # icacls command failed. Can happen because path doesn't exist
        # or we're not allowed to access acl information of the path.
        return False

    return does_permit_low_integrity_write(result.stdout)


def ensure_directories_exist(dirs):
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)


def check_directory_acls(dirs):
    acls_correct = True
    for dir in dirs:
        if not path_is_low_integrity_writable(dir):
            logging.info(
                f'Directory "{dir}" must be writable by low integrity '
                "processes for sandbox mode."
            )
            acls_correct = False

    return acls_correct


def setup_permissions(dirs):
    """
    Sets the correct low integrity write permissions for the given directories
    using an UAC elevation prompt. We need admin elevation because if the Comfy
    directory is not under the user's profile directory (e.g. any location in a
    non-C: drive), the regular user does not have permission to set the
    integrity level ACLs.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bat_path = os.path.join(script_dir, "setup_sandbox_permissions.bat")

    execute_info = {
        "lpVerb": "runas",  # Run as administrator
        "lpFile": bat_path,
        "lpParameters": " ".join(dirs),
        "nShow": win32con.SW_SHOWNORMAL,
        # This flag is necessary to wait for the process to finish.
        "fMask": shellcon.SEE_MASK_NOCLOSEPROCESS,
    }

    # This is equivalent to right-clicking the bat file and selecting "Run as
    # administrator"
    proc_info = shell.ShellExecuteEx(**execute_info)
    hProcess = proc_info["hProcess"]

    # Setup script should less than a second. Time out at 10 seconds.
    win32event.WaitForSingleObject(hProcess, 10 * 1000)
    exit_code = win32process.GetExitCodeProcess(hProcess)

    try:
        if exit_code == win32con.STATUS_PENDING:
            raise Exception("Sandbox permission script timed out")
        if exit_code != 0:
            raise Exception(
                "Sandbox permission setup script failed. " f"Exit code: {exit_code}"
            )
    finally:
        win32api.CloseHandle(hProcess)


def try_enable_sandbox():
    write_permitted_dirs = [
        folder_paths.get_write_permitted_base_directory(),
        folder_paths.get_output_directory(),
        folder_paths.get_user_directory(),
    ]
    write_permitted_dirs.extend(folder_paths.get_folder_paths("custom_nodes"))

    ensure_directories_exist(write_permitted_dirs)

    if check_directory_acls(write_permitted_dirs):
        set_process_integrity_level_to_low()
        return True

    # Directory permissions are not set up correctly. Try to fix.
    logging.critical(
        "Some directories do not have the correct permissions for sandbox mode "
        "to work. Would you like ComfyUI to fix these permissions? You will "
        "receive a UAC elevation prompt. [y/n]"
    )
    if input() != "y":
        return False

    setup_permissions(write_permitted_dirs)

    # Check directory permissions again before enabling sandbox.
    if check_directory_acls(write_permitted_dirs):
        set_process_integrity_level_to_low()
        return True

    # Directory permissions are still not set up correctly. Give up.
    return False
