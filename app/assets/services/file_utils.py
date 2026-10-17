import os
import stat
import time
from typing import Callable, NamedTuple

from app.assets.services.gil import yield_gil

# Longer run window for directory walks: the output rescan repeats after every prompt,
# so pausing every 2ms would add up to a much slower rescan.
RESCAN_YIELD_RUN = 0.010


def get_mtime_ns(stat_result: os.stat_result) -> int:
    """Extract mtime in nanoseconds from a stat result."""
    return getattr(
        stat_result, "st_mtime_ns", int(stat_result.st_mtime * 1_000_000_000)
    )


def get_size_and_mtime_ns(path: str, follow_symlinks: bool = True) -> tuple[int, int]:
    """Get file size in bytes and mtime in nanoseconds."""
    st = os.stat(path, follow_symlinks=follow_symlinks)
    return st.st_size, get_mtime_ns(st)


def verify_file_unchanged(
    mtime_db: int | None,
    size_db: int | None,
    stat_result: os.stat_result,
) -> bool:
    """Check if a file is unchanged based on mtime and size.

    Returns True if the file's mtime and size match the database values.
    Returns False if mtime_db is None or values don't match.

    size_db=None means don't check size; 0 is a valid recorded size.
    """
    if mtime_db is None:
        return False
    actual_mtime_ns = get_mtime_ns(stat_result)
    if int(mtime_db) != int(actual_mtime_ns):
        return False
    if size_db is not None:
        return int(stat_result.st_size) == int(size_db)
    return True


def is_visible(name: str) -> bool:
    """Return True if a file or directory name is visible (not hidden)."""
    return not name.startswith(".")


# dir path -> (visible file names, visible subdir names)
DirListings = dict[str, tuple[list[str], list[str]]]

# How a listing classified each entry, in listing order, so a cached listing replays as
# the listing would.
_DIR, _FILE, _REGULAR, _LINK = "d", "f", "r", "l"

# A cached listing is reused only once its directory's mtime was this far behind when it
# was listed. Closer, a change landing in the same timestamp tick (whole seconds on ext4
# with 128-byte inodes) would leave the mtime unchanged while the names differ.
RECENT_CHANGE_WINDOW_NS = 2_000_000_000


class CachedListing(NamedTuple):
    key: tuple[int, int, int]  # see _listing_key
    listed_at_ns: int  # wall clock just before the directory was stat'ed
    entries: list[tuple[str, str]]  # (name, kind) of each visible entry


# normalized absolute dir path -> its last listing
ListingCache = dict[str, CachedListing]


class ListingWalk(NamedTuple):
    files: list[str]
    listings: DirListings
    dirs_listed: int
    # file path -> its stat, taken while its directory was listed; absent when that stat failed
    # or the walk was asked not to stat
    stats: dict[str, os.stat_result]
    # files under directories served from the cache, which have no stat, known regular files
    reused_regular: frozenset[str] = frozenset()
    dirs_reused: int = 0
    # with a cache, the listings of this walk's trusted directories, for the next walk
    cache: ListingCache | None = None
    complete: bool = True


def _listing_key(st: os.stat_result) -> tuple[int, int, int]:
    # The inode and change time catch a directory replaced by one with the same mtime (a
    # rename over it, an rsync -a restore) and a chmod. Windows has no change time.
    return st.st_mtime_ns, st.st_ino, 0 if os.name == "nt" else st.st_ctime_ns


def _reusable(cached: CachedListing | None, st: os.stat_result) -> bool:
    return (
        cached is not None
        and cached.key == _listing_key(st)
        and cached.listed_at_ns - st.st_mtime_ns >= RECENT_CHANGE_WINDOW_NS
    )


def _is_link(entry: os.DirEntry) -> bool:
    try:
        return entry.is_symlink() or getattr(entry, "is_junction", bool)()
    except OSError:
        return False


def _list_visible_entries(
    dirpath: str, interrupted: Callable[[], bool] | None = None, stat_files: bool = False
) -> tuple[list[tuple[str, str]], dict[str, os.stat_result]] | None:
    """One directory's visible entries as (name, kind), and with ``stat_files`` each
    file's stat by name (a failed stat is left out). Anything whose is_dir() is false or
    raises is a file, as os.walk has it. Symlinks are classified by _replay_entries.

    ``interrupted`` is checked, without blocking, before each entry, so a directory of
    100k entries on a slow share can be left part way. True closes the directory and
    returns None: an open directory handle stops Windows renaming or moving any folder
    above it.

    With ``stat_files`` each file is stat'ed as it is listed: Windows answers from the
    listing itself, and a network share's client from the attributes the listing just
    brought back, where a later stat goes back to the server."""
    entries: list[tuple[str, str]] = []
    stats: dict[str, os.stat_result] = {}
    with os.scandir(dirpath) as scan:
        for entry in scan:
            if interrupted is not None and interrupted():
                return None
            yield_gil(run=RESCAN_YIELD_RUN)
            if not is_visible(entry.name):
                continue
            if _is_link(entry):
                entries.append((entry.name, _LINK))
                continue
            try:
                is_dir = entry.is_dir()
            except OSError:
                is_dir = False
            if is_dir:
                entries.append((entry.name, _DIR))
                continue
            try:
                kind = _REGULAR if entry.is_file() else _FILE
            except OSError:
                kind = _FILE
            entries.append((entry.name, kind))
            if stat_files:
                try:
                    stats[entry.name] = entry.stat()
                except OSError:
                    pass
    return entries, stats


def _replay_entries(
    dirpath: str, entries: list[tuple[str, str]]
) -> tuple[list[str], list[str], set[str], set[str], dict[str, os.stat_result]]:
    """(file names, subdir names, subdirs reached through a link, regular file names,
    link stats) of a listing. Each symlink is stat'ed here, also when the listing comes
    from the cache, as its target can appear or vanish without its directory changing. One
    whose target is gone is left out, so a row for it reads as vanished; one that can't be
    stat'ed for another reason (permissions, a loop) stays a file, for the scan's own stat."""
    files: list[str] = []
    subdirs: list[str] = []
    linked: set[str] = set()
    regular: set[str] = set()
    link_stats: dict[str, os.stat_result] = {}
    for name, kind in entries:
        if kind == _DIR:
            subdirs.append(name)
            continue
        if kind != _LINK:
            files.append(name)
            if kind == _REGULAR:
                regular.add(name)
            continue
        try:
            st = os.stat(os.path.join(dirpath, name))
        except (FileNotFoundError, NotADirectoryError):
            continue
        except OSError:
            files.append(name)
            continue
        if stat.S_ISDIR(st.st_mode):
            subdirs.append(name)
            linked.add(name)
            continue
        files.append(name)
        link_stats[name] = st
        if stat.S_ISREG(st.st_mode):
            regular.add(name)
    return files, subdirs, linked, regular, link_stats


def is_gone(path: str) -> bool:
    """True only when stat says the path does not exist. Any other error (permissions,
    I/O) leaves it undecided, so a scan keeps the file rather than guess."""
    try:
        os.stat(path)
    except (FileNotFoundError, NotADirectoryError):
        return True
    except OSError:
        return False
    return False


def walk_listings(
    base_dir: str,
    should_stop: Callable[[], bool] | None = None,
    interrupted: Callable[[], bool] | None = None,
    cancelled: Callable[[], bool] | None = None,
    stat_files: bool = False,
    follow_aliases: bool = False,
    cache: ListingCache | None = None,
    trusted: Callable[[str], bool] | None = None,
) -> ListingWalk:
    """Every visible file under ``base_dir``, following symlinks, every directory listing
    read on the way, and with ``stat_files`` each file's stat.

    The traversal is os.walk's (top-down, depth-first in listing order, following
    symlinked directories), with a device/inode guard against symlink cycles and hidden
    names left out. The guard lists each directory once; with ``follow_aliases`` it skips
    only a directory that is its own ancestor, so a second path to one is walked too.
    ``listings`` holds exactly the directories this walk listed or took from ``cache``,
    keyed by normalized absolute path, so it doubles as the record of which directories
    the walk can vouch for.

    With ``cache``, a directory whose mtime, inode and change time match its cached
    listing, taken at least RECENT_CHANGE_WINDOW_NS after that mtime, reuses it instead of
    being listed, if ``trusted`` says its filesystem keeps directory mtimes current. That
    is asked at the root, where the walk enters another device, and through a link (a
    Windows junction to a share keeps the local volume's serial). A directory's mtime
    changes only for entries added, removed or renamed directly in it, so every directory
    is still stat'ed; a reused directory has no file stats.

    ``should_stop`` is called before each directory, with none open; it may block, and
    once it returns True the walk ends with a partial result. ``interrupted`` (a pause or
    cancel was requested) is checked without blocking before each entry: the directory
    being read is closed and left out, so nothing is held open while ``should_stop``
    blocks. After a pause that directory is listed again checking only ``cancelled``, so
    a folder that takes longer to list than the gap between prompts still gets listed; a
    pause that lands during that second listing waits for it, as every listing did
    before the walk could stop part way.
    """
    files: list[str] = []
    listings: DirListings = {}
    stats: dict[str, os.stat_result] = {}
    reused_regular: set[str] = set()
    reused = 0
    next_cache: ListingCache | None = None if cache is None else {}
    complete = True
    # No isdir() precheck, so each directory costs exactly one stat: a root that is
    # missing or not a directory fails its stat or scandir below and yields nothing.
    seen_dirs: set[tuple[int, int]] = set()
    # (dir, its ancestors' ids, (parent's device, parent trusted) unless asked afresh)
    stack: list[tuple[str, frozenset[tuple[int, int]], tuple[int, bool] | None]] = [
        (os.path.abspath(base_dir), frozenset(), None)
    ]
    relist: str | None = None
    while stack:
        if should_stop is not None and should_stop():
            complete = False
            break
        yield_gil(run=RESCAN_YIELD_RUN)
        dirpath, ancestors, parent = stack.pop()
        listed_at_ns = time.time_ns()
        try:
            st = os.stat(dirpath)
        except OSError:
            continue
        dir_id = (st.st_dev, st.st_ino)
        if dir_id in (ancestors if follow_aliases else seen_dirs):
            continue
        is_trusted = False
        if next_cache is not None:
            if parent is not None and parent[0] == st.st_dev:
                is_trusted = parent[1]
            else:
                is_trusted = trusted is not None and trusted(dirpath)
        cached = cache.get(dirpath) if is_trusted and cache is not None else None
        name_stats: dict[str, os.stat_result] = {}
        was_reused = _reusable(cached, st)
        if was_reused:
            reused += 1
        else:
            try:
                listing = _list_visible_entries(
                    dirpath, cancelled if dirpath == relist else interrupted, stat_files
                )
            except OSError:
                continue
            if listing is None:
                stack.append((dirpath, ancestors, parent))
                relist = dirpath
                continue
            entries, name_stats = listing
            cached = CachedListing(_listing_key(st), listed_at_ns, entries)
        assert cached is not None
        names, subdirs, linked, regular, link_stats = _replay_entries(dirpath, cached.entries)
        if stat_files:
            name_stats.update(link_stats)
        if is_trusted and next_cache is not None:
            next_cache[dirpath] = cached
        if not follow_aliases:
            seen_dirs.add(dir_id)
        listings[dirpath] = (names, subdirs)
        for name in names:
            yield_gil(run=RESCAN_YIELD_RUN)
            path = os.path.abspath(os.path.join(dirpath, name))
            files.append(path)
            if name in name_stats:
                stats[path] = name_stats[name]
            elif was_reused and name in regular:
                reused_regular.add(path)
        inner = ancestors | {dir_id} if follow_aliases else ancestors
        stack.extend(
            (os.path.join(dirpath, name), inner, None if name in linked else (st.st_dev, is_trusted))
            for name in reversed(subdirs)
        )
    return ListingWalk(
        files, listings, len(listings) - reused, stats, frozenset(reused_regular), reused, next_cache, complete
    )
