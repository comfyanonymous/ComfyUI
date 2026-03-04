import ctypes
import errno
import logging
import os
import sys
import tempfile
from dataclasses import dataclass

# Landlock constants copied from linux/landlock.h
PR_SET_NO_NEW_PRIVS = 38

LANDLOCK_RULE_PATH_BENEATH = 1
LANDLOCK_CREATE_RULESET_VERSION = 1

LANDLOCK_ACCESS_FS_EXECUTE = 1 << 0
LANDLOCK_ACCESS_FS_WRITE_FILE = 1 << 1
LANDLOCK_ACCESS_FS_READ_FILE = 1 << 2
LANDLOCK_ACCESS_FS_READ_DIR = 1 << 3
LANDLOCK_ACCESS_FS_REMOVE_DIR = 1 << 4
LANDLOCK_ACCESS_FS_REMOVE_FILE = 1 << 5
LANDLOCK_ACCESS_FS_MAKE_CHAR = 1 << 6
LANDLOCK_ACCESS_FS_MAKE_DIR = 1 << 7
LANDLOCK_ACCESS_FS_MAKE_REG = 1 << 8
LANDLOCK_ACCESS_FS_MAKE_SOCK = 1 << 9
LANDLOCK_ACCESS_FS_MAKE_FIFO = 1 << 10
LANDLOCK_ACCESS_FS_MAKE_BLOCK = 1 << 11
LANDLOCK_ACCESS_FS_MAKE_SYM = 1 << 12
LANDLOCK_ACCESS_FS_REFER = 1 << 13
LANDLOCK_ACCESS_FS_TRUNCATE = 1 << 14
LANDLOCK_ACCESS_FS_IOCTL_DEV = 1 << 15  # ABI v5+

# Pre-computed access masks
FS_READ_ACCESS = (
    LANDLOCK_ACCESS_FS_READ_FILE
    | LANDLOCK_ACCESS_FS_READ_DIR
    | LANDLOCK_ACCESS_FS_EXECUTE
)
FS_WRITE_ACCESS = (
    FS_READ_ACCESS
    | LANDLOCK_ACCESS_FS_WRITE_FILE
    | LANDLOCK_ACCESS_FS_MAKE_DIR
    | LANDLOCK_ACCESS_FS_MAKE_REG
    | LANDLOCK_ACCESS_FS_MAKE_SOCK
    | LANDLOCK_ACCESS_FS_MAKE_FIFO
    | LANDLOCK_ACCESS_FS_MAKE_BLOCK
    | LANDLOCK_ACCESS_FS_MAKE_CHAR
    | LANDLOCK_ACCESS_FS_MAKE_SYM
    | LANDLOCK_ACCESS_FS_REMOVE_DIR
    | LANDLOCK_ACCESS_FS_REMOVE_FILE
)

# Syscall numbers are ABI-stable across all 64-bit Linux architectures
SYS_LANDLOCK_CREATE_RULESET = 444
SYS_LANDLOCK_ADD_RULE = 445
SYS_LANDLOCK_RESTRICT_SELF = 446


class _RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _PathBeneathAttr(ctypes.Structure):
    _fields_ = [
        ("allowed_access", ctypes.c_uint64),
        ("parent_fd", ctypes.c_int32),
        ("reserved", ctypes.c_uint32),
    ]


@dataclass(frozen=True)
class LandlockRules:
    read_paths: set[str]
    write_paths: set[str]
    ioctl_paths: set[str]


def _normalize_paths(paths: set[str]) -> set[str]:
    normalized = set()
    for path in paths:
        if not path:
            continue
        normalized.add(os.path.realpath(path))
    return normalized


class LandlockEnforcer:
    def __init__(self, logger: logging.Logger | None = None):
        self.log = logger or logging.getLogger(__name__)
        self.libc = ctypes.CDLL(None, use_errno=True)
        self.libc.syscall.restype = ctypes.c_long
        self.libc.prctl.restype = ctypes.c_int

    def _syscall(self, syscall_nr, *args) -> tuple[int | None, int]:
        ctypes.set_errno(0)
        res = self.libc.syscall(ctypes.c_long(syscall_nr), *args)
        if res == -1:
            return None, ctypes.get_errno()
        return res, 0

    def _abi_version(self) -> int:
        res, err = self._syscall(
            SYS_LANDLOCK_CREATE_RULESET,
            ctypes.c_void_p(0),
            ctypes.c_size_t(0),
            ctypes.c_uint(LANDLOCK_CREATE_RULESET_VERSION),
        )
        if res is None:
            if err in (errno.ENOSYS, errno.EOPNOTSUPP):
                return 0
            return -err
        return res

    def _create_ruleset(self, handled_access: int) -> tuple[int | None, int]:
        ruleset = _RulesetAttr(ctypes.c_uint64(handled_access))
        return self._syscall(
            SYS_LANDLOCK_CREATE_RULESET,
            ctypes.byref(ruleset),
            ctypes.c_size_t(ctypes.sizeof(ruleset)),
            ctypes.c_uint(0),
        )

    def _add_rule(self, ruleset_fd: int, path: str, access_mask: int, allow_ioctl: bool) -> bool:
        if allow_ioctl:
            access_mask |= LANDLOCK_ACCESS_FS_IOCTL_DEV

        try:
            dir_fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
        except OSError as exc:
            self.log.warning("Landlock: skipping %s (%s)", path, exc)
            return False

        try:
            rule = _PathBeneathAttr(
                ctypes.c_uint64(access_mask), ctypes.c_int32(dir_fd), ctypes.c_uint32(0)
            )
            res, err = self._syscall(
                SYS_LANDLOCK_ADD_RULE,
                ctypes.c_int(ruleset_fd),
                ctypes.c_int(LANDLOCK_RULE_PATH_BENEATH),
                ctypes.byref(rule),
                ctypes.c_uint(0),
            )
            if res is None:
                self.log.warning("Landlock: failed to add %s (errno=%s)", path, err)
                return False
            return True
        finally:
            os.close(dir_fd)

    def _restrict_self(self, ruleset_fd: int) -> bool:
        if self.libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            self.log.warning(
                "Landlock: prctl(PR_SET_NO_NEW_PRIVS) failed (errno=%s)",
                ctypes.get_errno(),
            )
            return False

        res, err = self._syscall(SYS_LANDLOCK_RESTRICT_SELF, ctypes.c_int(ruleset_fd), ctypes.c_uint(0))
        if res is None:
            self.log.warning("Landlock: restrict_self failed (errno=%s)", err)
            return False
        return True

    def apply(self, rules: LandlockRules) -> bool:
        if not sys.platform.startswith("linux"):
            self.log.info("Landlock: not a Linux platform, skipping.")
            return False

        abi_version = self._abi_version()
        if abi_version <= 0:
            self.log.info("Landlock: not available on this kernel (abi=%s).", abi_version)
            return False

        read_access = FS_READ_ACCESS
        handled_write_access = FS_WRITE_ACCESS
        allowed_write_access = (
            read_access
            | LANDLOCK_ACCESS_FS_WRITE_FILE
            | LANDLOCK_ACCESS_FS_MAKE_DIR
            | LANDLOCK_ACCESS_FS_MAKE_REG
            | LANDLOCK_ACCESS_FS_REMOVE_DIR
            | LANDLOCK_ACCESS_FS_REMOVE_FILE
        )  # leave other handled rights (symlinks, device nodes, sockets) denied
        if abi_version >= 2:
            handled_write_access |= LANDLOCK_ACCESS_FS_TRUNCATE
            allowed_write_access |= LANDLOCK_ACCESS_FS_TRUNCATE
        if abi_version >= 3:
            handled_write_access |= LANDLOCK_ACCESS_FS_REFER
            allowed_write_access |= LANDLOCK_ACCESS_FS_REFER

        handled_access = handled_write_access | read_access
        ioctl_supported = abi_version >= 5
        if ioctl_supported:
            handled_access |= LANDLOCK_ACCESS_FS_IOCTL_DEV

        write_paths = _normalize_paths(rules.write_paths)
        read_paths = _normalize_paths(rules.read_paths) - write_paths
        # In theory these could require write or read access. Though in practice it's just /dev.
        ioctl_paths = _normalize_paths(rules.ioctl_paths)

        if ioctl_paths and not ioctl_supported:
            self.log.info(
                "Landlock: ioctl access requested but ABI %s has no support; continuing without ioctl.",
                abi_version,
            )

        ruleset_fd = None
        ruleset_fd, err = self._create_ruleset(handled_access)
        if ruleset_fd is None:
            self.log.warning("Landlock: failed to create ruleset (errno=%s)", err)
            return False

        try:
            for path in write_paths:
                if path != os.path.sep:
                    try:
                        os.makedirs(path, exist_ok=True)
                    except Exception as exc:
                        self.log.warning("Landlock: unable to prepare %s (%s)", path, exc)
                        return False
                if not self._add_rule(
                    ruleset_fd, path, allowed_write_access, ioctl_supported and path in ioctl_paths
                ):
                    return False

            for path in read_paths:
                self._add_rule(ruleset_fd, path, read_access, ioctl_supported and path in ioctl_paths)

            if not self._restrict_self(ruleset_fd):
                return False
        finally:
            if ruleset_fd is not None:
                os.close(ruleset_fd)

        if write_paths:
            self.log.info(
                "Landlock enabled (ABI %s). Writable roots: %s",
                abi_version,
                ", ".join(sorted(write_paths)),
            )
        else:
            self.log.info("Landlock enabled (ABI %s). No writable roots configured.", abi_version)
        return True


_landlock_applied = False


def build_default_rules(args) -> LandlockRules:
    import folder_paths
    from urllib.parse import urlparse

    write_paths: set[str] = {
        folder_paths.get_output_directory(),
        folder_paths.get_input_directory(),
        folder_paths.get_temp_directory(),
        folder_paths.get_user_directory(),
    }

    ioctl_paths: set[str] = set()

    # Torch and some backends use system temp and /dev/shm
    write_paths.add(tempfile.gettempdir())

    if args.temp_directory:
        write_paths.add(os.path.join(os.path.abspath(args.temp_directory), "temp"))

    db_url = getattr(args, "database_url", None)
    if db_url and db_url.startswith("sqlite"):
        parsed = urlparse(db_url)
        if parsed.scheme == "sqlite" and parsed.path:
            write_paths.add(os.path.abspath(os.path.dirname(parsed.path)))

    for path in args.landlock_allow_writable or []:
        if path:
            write_paths.add(path)

    # Build read paths - only what's actually needed
    read_paths: set[str] = set()

    # ComfyUI codebase
    read_paths.add(folder_paths.base_path)

    # All configured model directories (includes extra_model_paths.yaml)
    for folder_name in folder_paths.folder_names_and_paths:
        for path in folder_paths.folder_names_and_paths[folder_name][0]:
            read_paths.add(path)

    # Python installation and site-packages
    read_paths.add(sys.prefix)
    if sys.base_prefix != sys.prefix:
        read_paths.add(sys.base_prefix)
    for path in sys.path:
        if path and os.path.isdir(path):
            read_paths.add(path)

    # System libraries (required for shared libs, CUDA, etc.)
    for system_path in ["/usr", "/lib", "/lib64", "/opt", "/etc", "/proc", "/sys"]:
        if os.path.exists(system_path):
            read_paths.add(system_path)

    # NixOS: /nix/store contains the entire system
    if os.path.exists("/nix"):
        read_paths.add("/nix")

    # /dev needs write + ioctl for CUDA/GPU access
    write_paths.add("/dev")
    ioctl_paths.add("/dev")

    # User-specified additional read paths
    for path in getattr(args, "landlock_allow_readable", None) or []:
        if path:
            read_paths.add(path)

    return LandlockRules(read_paths=_normalize_paths(read_paths), write_paths=_normalize_paths(write_paths), ioctl_paths=_normalize_paths(ioctl_paths))


def enable_landlock(args, logger: logging.Logger | None = None) -> bool:
    global _landlock_applied

    if _landlock_applied:
        return True
    if not getattr(args, "enable_landlock", False):
        return False

    enforcer = LandlockEnforcer(logger)
    try:
        _landlock_applied = enforcer.apply(build_default_rules(args))
    except Exception:
        enforcer.log.exception("Landlock: unexpected failure while applying ruleset.")
        _landlock_applied = False
    return _landlock_applied
