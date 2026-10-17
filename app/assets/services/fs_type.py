"""Which filesystem a directory is on, to decide whether its mtimes can be trusted."""

import ctypes
import ctypes.util
import logging
import os
import re
import sys

# Filesystems whose directory mtimes the scan trusts to skip re-listing an unchanged
# directory, per OS, each measured to update a directory's mtime on every add, delete and
# rename in it. Network clients can serve a stale one from their attribute cache (an SMB
# share measured stale for 30s to over 5 minutes), FAT/exFAT never update it on Windows,
# and the rest are untested, so their directories are listed on every scan.
MTIME_TRUSTED_FILESYSTEMS = {"linux": {"ext4"}, "darwin": {"apfs"}, "win32": {"ntfs"}}

_DRIVE_REMOVABLE = 2
_DRIVE_FIXED = 3
# Set on real NTFS volumes; a user-mode filesystem (WinFsp, Dokan: rclone, cloud drives)
# can call itself NTFS but has no change journal.
_FILE_SUPPORTS_USN_JOURNAL = 0x02000000


def fs_type_from_mountinfo(mountinfo: str, path: str) -> str | None:
    """The type of the mount holding ``path`` (absolute, symlinks resolved), from the
    text of /proc/self/mountinfo: the longest matching mount point, the later one when
    a point is mounted over."""
    best_len, best_type = -1, None
    for line in mountinfo.splitlines():
        left, sep, right = line.partition(" - ")
        fields = left.split()
        if not sep or len(fields) < 5 or not right.split():
            continue
        mount_point = re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), fields[4])
        prefix = mount_point.rstrip("/") + "/"
        if (path == mount_point or path.startswith(prefix)) and len(mount_point) >= best_len:
            best_len, best_type = len(mount_point), right.split()[0]
    return best_type


def _linux_fs_type(path: str) -> str | None:
    with open("/proc/self/mountinfo", encoding="utf-8", errors="replace") as f:
        return fs_type_from_mountinfo(f.read(), path)


class _MacStatfs(ctypes.Structure):
    _fields_ = [
        ("f_bsize", ctypes.c_uint32), ("f_iosize", ctypes.c_int32),
        ("f_blocks", ctypes.c_uint64), ("f_bfree", ctypes.c_uint64),
        ("f_bavail", ctypes.c_uint64), ("f_files", ctypes.c_uint64),
        ("f_ffree", ctypes.c_uint64), ("f_fsid", ctypes.c_int32 * 2),
        ("f_owner", ctypes.c_uint32), ("f_type", ctypes.c_uint32),
        ("f_flags", ctypes.c_uint32), ("f_fssubtype", ctypes.c_uint32),
        ("f_fstypename", ctypes.c_char * 16), ("f_mntonname", ctypes.c_char * 1024),
        ("f_mntfromname", ctypes.c_char * 1024), ("f_flags_ext", ctypes.c_uint32),
        ("f_reserved", ctypes.c_uint32 * 7),
    ]


def _mac_fs_type(path: str) -> str | None:
    libc = ctypes.CDLL(ctypes.util.find_library("c"))
    try:  # Intel exports the 64-bit-inode struct under this name; Apple silicon has no suffix.
        statfs = libc["statfs$INODE64"]
    except AttributeError:
        statfs = libc.statfs
    buf = _MacStatfs()
    if statfs(os.fsencode(path), ctypes.byref(buf)) != 0:
        return None
    return buf.f_fstypename.decode(errors="replace")


def _windows_fs_type(path: str) -> str | None:
    kernel32 = ctypes.WinDLL("kernel32")  # type: ignore[attr-defined]
    volume = ctypes.create_unicode_buffer(261)
    if not kernel32.GetVolumePathNameW(path, volume, len(volume)):
        return None
    if kernel32.GetDriveTypeW(volume.value) not in (_DRIVE_FIXED, _DRIVE_REMOVABLE):
        return "remote"
    name = ctypes.create_unicode_buffer(261)
    flags = ctypes.c_uint32()
    if not kernel32.GetVolumeInformationW(
        volume.value, None, 0, None, None, ctypes.byref(flags), name, len(name)
    ):
        return None
    if name.value.lower() == "ntfs" and not flags.value & _FILE_SUPPORTS_USN_JOURNAL:
        return "ntfs (user-mode)"
    return name.value


def filesystem_type(path: str) -> str | None:
    """The filesystem type holding ``path``, e.g. "ext4", "apfs", "NTFS", "remote" for a
    Windows drive that isn't fixed or removable; None when it cannot be determined."""
    path = os.path.realpath(path)
    if sys.platform == "win32":
        return _windows_fs_type(path)
    if sys.platform == "darwin":
        return _mac_fs_type(path)
    if sys.platform.startswith("linux"):
        return _linux_fs_type(path)
    return None


def directory_mtimes_trusted(path: str) -> bool:
    """Whether ``path``'s filesystem keeps directory mtimes current
    (MTIME_TRUSTED_FILESYSTEMS); False when it cannot be determined."""
    try:
        fs_type = filesystem_type(path)
    except Exception:
        fs_type = None
    trusted = fs_type is not None and fs_type.lower() in MTIME_TRUSTED_FILESYSTEMS.get(sys.platform, ())
    logging.debug("Asset scan: %s is on %s; directory mtimes %s", path, fs_type, "trusted" if trusted else "not trusted")
    return trusted
