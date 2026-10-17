import pytest

from app.assets.services import fs_type
from app.assets.services.fs_type import (
    directory_mtimes_trusted,
    fs_type_from_mountinfo,
)

MOUNTINFO = """\
22 1 259:2 / / rw,relatime shared:1 - ext4 /dev/nvme0n1p2 rw
25 22 0:22 / /tmp rw,nosuid shared:5 - tmpfs tmpfs rw
40 22 0:40 / /mnt/nas rw,relatime shared:20 - nfs4 10.0.0.3:/export rw,vers=4.1
41 22 0:41 / /mnt/My\\040Share rw,relatime shared:21 - cifs //nas/share rw
42 22 0:42 / /home/me/remote rw,nosuid shared:22 - fuse.sshfs me@host:/ rw
43 40 259:3 / /mnt/nas/local rw,relatime shared:23 - xfs /dev/sdb1 rw
44 22 0:44 / /mnt/over rw shared:24 - vfat /dev/sdc1 rw
45 22 0:45 / /mnt/over rw shared:25 - btrfs /dev/sdd1 rw
"""


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/home/me/ComfyUI/output", "ext4"),
        ("/tmp", "tmpfs"),
        ("/tmp/x/output", "tmpfs"),
        ("/tmpfoo/output", "ext4"),  # a prefix of the name, not a parent directory
        ("/mnt/nas/outputs/2026", "nfs4"),
        ("/mnt/nas/local/output", "xfs"),  # a local disk mounted inside the NFS mount
        ("/mnt/My Share/output", "cifs"),  # mountinfo escapes the space as \040
        ("/home/me/remote/output", "fuse.sshfs"),
        ("/mnt/over/output", "btrfs"),  # the later mount over the same point wins
    ],
)
def test_fs_type_from_mountinfo_finds_the_innermost_mount(path, expected):
    assert fs_type_from_mountinfo(MOUNTINFO, path) == expected


def test_fs_type_from_mountinfo_skips_malformed_lines():
    assert fs_type_from_mountinfo("garbage\n22 1 259:2 / / rw -\n", "/x") is None


@pytest.mark.parametrize(
    ("platform", "detected", "trusted"),
    [
        ("linux", "ext4", True),
        ("darwin", "apfs", True),
        ("win32", "NTFS", True),
        # untested here, so listed every scan
        ("linux", "xfs", False),
        ("linux", "btrfs", False),
        ("linux", "tmpfs", False),
        ("win32", "ReFS", False),
        # never update a directory's mtime, or serve a stale one
        ("win32", "FAT32", False),
        ("win32", "exFAT", False),
        ("linux", "vfat", False),
        ("linux", "nfs4", False),
        ("linux", "cifs", False),
        ("darwin", "smbfs", False),
        ("win32", "remote", False),
        ("win32", "ntfs (user-mode)", False),
        # another OS's driver for a trusted filesystem
        ("linux", "ntfs", False),
        ("linux", "ntfs3", False),
        ("linux", "fuseblk", False),
        ("darwin", "ntfs", False),
        ("linux", "apfs", False),
        ("linux", None, False),
    ],
)
def test_directory_mtimes_are_trusted_only_on_measured_local_filesystems(platform, detected, trusted, monkeypatch):
    monkeypatch.setattr(fs_type.sys, "platform", platform)
    monkeypatch.setattr(fs_type, "filesystem_type", lambda _path: detected)
    assert directory_mtimes_trusted("/any") is trusted


def test_a_failed_detection_is_not_trusted(monkeypatch):
    def broken(_path):
        raise OSError("no mountinfo")

    monkeypatch.setattr(fs_type, "filesystem_type", broken)
    assert directory_mtimes_trusted("/any") is False


class _Kernel32:
    """GetVolumePathNameW, GetDriveTypeW and GetVolumeInformationW for one volume."""

    def __init__(self, drive_type: int, name: str, flags: int) -> None:
        self.drive_type, self.name, self.flags = drive_type, name, flags

    def GetVolumePathNameW(self, _path, buf, _size):
        buf.value = "C:\\"
        return 1

    def GetDriveTypeW(self, _volume):
        return self.drive_type

    def GetVolumeInformationW(self, _volume, _vname, _vsize, _serial, _maxlen, flags, name, _size):
        flags._obj.value = self.flags
        name.value = self.name
        return 1


@pytest.mark.parametrize(
    ("drive_type", "name", "flags", "expected"),
    [
        (3, "NTFS", 0x02000000, "NTFS"),  # fixed, with a change journal
        (2, "NTFS", 0x02000000, "NTFS"),  # removable
        (3, "NTFS", 0, "ntfs (user-mode)"),  # WinFsp/Dokan calling itself NTFS
        (4, "NTFS", 0x02000000, "remote"),  # a network drive
        (5, "CDFS", 0, "remote"),  # anything but fixed or removable
        (3, "exFAT", 0, "exFAT"),
    ],
)
def test_windows_detection_needs_a_fixed_drive_and_a_change_journal_for_ntfs(
    drive_type, name, flags, expected, monkeypatch
):
    monkeypatch.setattr(fs_type.ctypes, "WinDLL", lambda _name: _Kernel32(drive_type, name, flags), raising=False)
    assert fs_type._windows_fs_type("C:\\x") == expected


@pytest.mark.parametrize(
    ("platform", "probe"),
    [("linux", "_linux_fs_type"), ("darwin", "_mac_fs_type"), ("win32", "_windows_fs_type")],
)
def test_filesystem_type_dispatches_on_the_platform_with_the_resolved_path(
    platform, probe, monkeypatch, tmp_path
):
    target = tmp_path / "real"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target)
    seen: list[str] = []
    monkeypatch.setattr(fs_type.sys, "platform", platform)
    monkeypatch.setattr(fs_type, probe, lambda path: seen.append(path) or "probed")

    assert fs_type.filesystem_type(str(link)) == "probed"
    assert seen == [str(target.resolve())]


def test_filesystem_type_is_unknown_on_other_platforms(monkeypatch):
    monkeypatch.setattr(fs_type.sys, "platform", "freebsd14")
    assert fs_type.filesystem_type("/") is None


@pytest.mark.skipif(not fs_type.sys.platform.startswith("linux"), reason="reads /proc/self/mountinfo")
def test_this_hosts_mountinfo_names_a_type_for_a_real_directory(tmp_path):
    assert fs_type.filesystem_type(str(tmp_path))
