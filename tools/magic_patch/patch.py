"""Generate and apply a pack's distribution pair: `<Pack>.json` + `<Pack>.diff`.

A converted pack is not deployed as a folder. Its checked-in review artifact
is the DIFFERENCE between the pristine pack and its `v2/` tree, in two files
that together are sufficient to re-create `v2/` byte-for-byte on top of any
copy of the original. Deployment wraps exactly those two files in one zip.

  <Pack>.json   the manifest — for every file of the v2 tree, what to do and
                how to prove it was done right:
                  copy     v2 file is byte-identical to the pack's own file
                  convert  v2 file derives from it: SEED with the pack's
                           bytes, then apply this file's hunks from the .diff
                  add      v2 file has no counterpart: its hunks in the .diff
                           are a creation (a/dev/null)
                  delete   a pack file deliberately absent from v2/
                Every entry carries sha256 of what it expects and of what it
                must produce, so application is verified, not hoped.

                `v2/` is a COMPLETE pack, so a node module sits beside every
                sibling it imports and `__file__`-relative resources resolve
                exactly as they did upstream. The duplication costs nothing in
                the artifact: `copy` is one manifest line, not a file body.

  <Pack>.diff   ordinary unified diff, human-readable — the REVIEW artifact.
                Because `convert` seeds from the original, its hunks are the
                conversion and nothing else; a reviewer reads the boundary
                changes, not two interleaved copies of the pack.

Why a pair instead of one big diff: a plain tree diff must carry every
unchanged file as a full addition (fonts, vendored JS, untouched sources),
which buries the conversion and bloats the artifact by the size of the pack.

A binary file may be `copy` or `delete` only. A binary that genuinely changes
in conversion has no reviewable diff, which is a smell worth refusing until a
real case argues otherwise.
"""

from __future__ import annotations

import difflib
import hashlib
import io
import json
import re
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath

FORMAT = "comfy-pack-patch/1"
OPS = {"copy", "convert", "add", "delete"}
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
SHA256_RE = re.compile(r"[0-9a-f]{64}")
HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?: .*)?$")
MAX_BUNDLE_BYTES = 96 * 1024 * 1024
MAX_BUNDLE_MEMBER_BYTES = 64 * 1024 * 1024

#: Kept out of both trees entirely: derived or environment litter, never pack
#: content. `.git` appears when a pack folder is a checkout; caches when
#: anything imported it in place.
IGNORE_DIRS = {".git", "__pycache__", ".pytest_cache", "node_modules"}
IGNORE_FILES = {".DS_Store"}


class PackPatchError(Exception):
    pass


def _files(
    root: Path,
    *,
    installed_manifest_paths: set[str] | None = None,
) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        rel_text = rel.as_posix()
        if (
            installed_manifest_paths is not None
            and rel_text not in installed_manifest_paths
        ):
            continue
        if p.is_symlink():
            raise PackPatchError(f"symbolic link is not pack-patch content: {p}")
        if not p.is_file():
            continue
        if rel.parts[0] == "v2" and root.name != "v2":
            continue
        if any(part in IGNORE_DIRS for part in rel.parts):
            continue
        if rel.name in IGNORE_FILES:
            continue
        out[rel_text] = p
    return out


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _is_text(p: Path) -> bool:
    data = p.read_bytes()
    if b"\0" in data[:8192]:
        return False
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def _lines(p: Path) -> list[str]:
    try:
        return p.read_bytes().decode("utf-8").splitlines(keepends=True)
    except UnicodeDecodeError as exc:
        raise PackPatchError(f"text file is not UTF-8: {p}") from exc


def _mode(p: Path) -> int:
    return stat.S_IMODE(p.stat().st_mode) & 0o777


def _unified(a_lines, b_lines, a_label: str, b_label: str) -> str:
    """difflib's unified diff, with git's no-newline convention restored.

    difflib yields a file's final line verbatim — WITHOUT a newline if the
    file has none — and never emits git's ``\\ No newline at end of file``
    marker. Concatenating such parts glues the next file's ``---`` header
    onto the unterminated line. Every unterminated line therefore gets a
    newline plus the marker, which the applier strips back out, keeping the
    round trip byte-exact for files that do not end in a newline.
    """
    out = []
    for line in difflib.unified_diff(
        a_lines, b_lines, fromfile=a_label, tofile=b_label, n=3
    ):
        out.append(line)
        if not line.endswith("\n"):
            out.append("\n\\ No newline at end of file\n")
    return "".join(out)


def pack_key(pack: Path) -> str:
    """The pack's identity: ``x`` + short git commit sha of the snapshot.

    In the db layout the key IS the snapshot folder's name —
    ``<Pack-Name>/<xsha>/`` — so identity is read off the path and the tree
    itself stays exactly upstream's bytes. ``packs.json`` beside the pack
    folders records provenance (upstream URL, full commit), not identity.
    """
    name = Path(pack).resolve().name
    if re.fullmatch(r"x[0-9a-f]{7,12}", name):
        return name
    raise PackPatchError(
        f"pack folder {name!r} is not a snapshot key — expected the db "
        f"layout <slug>/x<short-commit-sha>/"
    )


def _pack_folder(snapshot: Path) -> Path:
    pack_dirs = [
        directory
        for directory in sorted(Path(snapshot).iterdir())
        if directory.is_dir() and not directory.name.startswith(".")
    ]
    if len(pack_dirs) != 1:
        raise PackPatchError(
            f"{snapshot}: a snapshot holds exactly one pack folder, "
            f"found {[directory.name for directory in pack_dirs]}"
        )
    return pack_dirs[0]


def generate(pack: Path) -> tuple[dict, str]:
    """Build the (manifest, diff-text) pair for ``pack``.

    The v2 tree is authoritative for what ships; the original tree is
    authoritative for what it is a change AGAINST.
    """
    snapshot = Path(pack)
    pack_dir = _pack_folder(snapshot)
    v2 = pack_dir / "v2"
    if not v2.is_dir():
        raise PackPatchError(f"{pack_dir} has no v2/ tree")

    # Paths are relative to the pack folder on both sides. `_files` skips the
    # `v2` child, so the pristine side is the pack as upstream shipped it.
    orig = _files(pack_dir)
    conv = _files(v2)

    entries: list[dict] = []
    diff_parts: list[str] = []

    for rel in sorted(set(orig) | set(conv)):
        o, c = orig.get(rel), conv.get(rel)
        if c is None:
            entries.append({"path": rel, "op": "delete", "base_sha256": _sha(o)})
            continue
        if o is None:
            if not _is_text(c):
                raise PackPatchError(
                    f"binary file added in v2 with no original: {rel} — "
                    f"a patch pair cannot review this; ship it in the base "
                    f"pack or argue the case"
                )
            entries.append(
                {"path": rel, "op": "add", "v2_sha256": _sha(c), "mode": _mode(c)}
            )
            diff_parts.append(_unified([], _lines(c), "/dev/null", f"b/v2/{rel}"))
            continue

        base_sha, v2_sha = _sha(o), _sha(c)
        if base_sha == v2_sha:
            entries.append(
                {"path": rel, "op": "copy", "base_sha256": base_sha, "mode": _mode(c)}
            )
            continue
        if not (_is_text(o) and _is_text(c)):
            raise PackPatchError(f"binary file differs between trees: {rel}")
        entries.append(
            {
                "path": rel,
                "op": "convert",
                "base_sha256": base_sha,
                "v2_sha256": v2_sha,
                "mode": _mode(c),
            }
        )
        diff_parts.append(_unified(_lines(o), _lines(c), f"a/{rel}", f"b/v2/{rel}"))

    manifest = {
        "format": FORMAT,
        "pack": Path(pack).resolve().parent.name,
        "key": pack_key(pack),
        "files": entries,
        "counts": {
            op: sum(1 for e in entries if e["op"] == op)
            for op in ("copy", "convert", "add", "delete")
        },
    }
    return manifest, "".join(diff_parts)


def _safe_relpath(value) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\0" in value:
        raise PackPatchError(f"unsafe manifest path {value!r}")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or PureWindowsPath(value).drive
        or path.as_posix() != value
        or ".." in path.parts
        or path.parts[0] == "v2"
    ):
        raise PackPatchError(f"unsafe manifest path {value!r}")
    return value


def _validate_manifest(
    snapshot: Path,
    pack: Path,
    manifest,
    *,
    allow_runtime_products: bool = False,
) -> tuple[list[dict], dict[str, Path]]:
    if not isinstance(manifest, dict):
        raise PackPatchError("manifest must be an object")
    if manifest.get("format") != FORMAT:
        raise PackPatchError(f"unknown manifest format {manifest.get('format')!r}")
    expected_pack = snapshot.resolve().parent.name
    if manifest.get("pack") != expected_pack:
        raise PackPatchError(
            f"manifest pack {manifest.get('pack')!r} does not match {expected_pack!r}"
        )
    expected_key = pack_key(snapshot)
    if manifest.get("key") != expected_key:
        raise PackPatchError(
            f"manifest key {manifest.get('key')!r} does not match {expected_key!r}"
        )

    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise PackPatchError("manifest files must be a list")
    seen: set[str] = set()
    seen_casefolded: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise PackPatchError("manifest file entries must be objects")
        rel = _safe_relpath(entry.get("path"))
        folded = rel.casefold()
        if rel in seen or folded in seen_casefolded:
            raise PackPatchError(f"duplicate manifest path {rel!r}")
        seen.add(rel)
        seen_casefolded.add(folded)

        op = entry.get("op")
        if op not in OPS:
            raise PackPatchError(f"unknown op {op!r} for {rel}")
        required = {"path", "op"}
        if op in {"copy", "convert", "delete"}:
            required.add("base_sha256")
        if op in {"add", "convert"}:
            required.add("v2_sha256")
        if op != "delete":
            required.add("mode")
        if set(entry) != required:
            raise PackPatchError(
                f"manifest fields for {rel} are {sorted(entry)}, expected {sorted(required)}"
            )
        for field in ("base_sha256", "v2_sha256"):
            if field in entry and not (
                isinstance(entry[field], str) and SHA256_RE.fullmatch(entry[field])
            ):
                raise PackPatchError(f"invalid {field} for {rel}")
        if "mode" in entry and (
            isinstance(entry["mode"], bool)
            or not isinstance(entry["mode"], int)
            or not 0 <= entry["mode"] <= 0o777
        ):
            raise PackPatchError(f"invalid mode for {rel}")

    declared_base = {
        entry["path"]
        for entry in entries
        if entry["op"] in {"copy", "convert", "delete"}
    }
    base_files = _files(
        pack,
        installed_manifest_paths=(declared_base if allow_runtime_products else None),
    )
    actual_base = set(base_files)
    if declared_base != actual_base:
        missing = sorted(actual_base - declared_base)
        extra = sorted(declared_base - actual_base)
        raise PackPatchError(
            f"manifest base file set differs: missing={missing}, extra={extra}"
        )

    expected_counts = {
        op: sum(entry["op"] == op for entry in entries) for op in sorted(OPS)
    }
    if manifest.get("counts") != expected_counts:
        raise PackPatchError(
            f"manifest counts {manifest.get('counts')!r} do not match {expected_counts!r}"
        )
    return entries, base_files


def _parse_diff(diff_text: str) -> dict[str, tuple[str, list[str]]]:
    """Split one concatenated unified diff into per-target hunk blocks,
    keyed by the pack-relative path (the ``b/v2/`` prefix stripped).

    Framing counts hunk spans from the ``@@`` headers rather than pattern-
    matching ``---``/``+++`` anywhere: a removed content line beginning
    ``-- x`` renders as ``--- x`` and would otherwise read as a file
    boundary in the middle of a hunk. The header declares exactly how many
    body lines follow; trusting it is both simpler and unspoofable.
    """
    lines = diff_text.splitlines(keepends=True)
    blocks: dict[str, tuple[str, list[str]]] = {}
    i = 0
    while i < len(lines):
        if not lines[i].startswith("--- "):
            raise PackPatchError(f"expected a file header, got {lines[i]!r}")
        source = lines[i][4:].rstrip("\r\n")
        if i + 1 >= len(lines) or not lines[i + 1].startswith("+++ "):
            raise PackPatchError(f"file header without target at line {i + 1}")
        target = lines[i + 1][4:].rstrip("\r\n")
        if not target.startswith("b/v2/"):
            raise PackPatchError(f"unexpected diff target {target!r}")
        rel = _safe_relpath(target[len("b/v2/") :])
        if rel in blocks:
            raise PackPatchError(f"duplicate diff target {rel!r}")
        block: list[str] = []
        i += 2
        while i < len(lines) and lines[i].startswith("@@"):
            header = lines[i]
            match = HUNK_RE.fullmatch(header.rstrip("\r\n"))
            if match is None:
                raise PackPatchError(f"{rel}: malformed hunk header {header!r}")
            old_len = int(match.group(2)) if match.group(2) is not None else 1
            new_len = int(match.group(4)) if match.group(4) is not None else 1
            block.append(header)
            i += 1
            remaining_old, remaining_new = old_len, new_len
            while (remaining_old > 0 or remaining_new > 0) and i < len(lines):
                body = lines[i]
                tag = body[0] if body else " "
                if tag == " ":
                    remaining_old -= 1
                    remaining_new -= 1
                elif tag == "-":
                    remaining_old -= 1
                elif tag == "+":
                    remaining_new -= 1
                elif tag == "\\":
                    pass
                else:
                    raise PackPatchError(f"{target}: malformed hunk body {body!r}")
                block.append(body)
                i += 1
            if remaining_old != 0 or remaining_new != 0:
                raise PackPatchError(f"{rel}: truncated hunk body")
            if i < len(lines) and lines[i].startswith("\\"):
                block.append(lines[i])
                i += 1
        if not block:
            raise PackPatchError(f"{rel}: diff header has no hunks")
        blocks[rel] = (source, block)
    return blocks


def _apply_hunks(base_lines: list[str], hunks: list[str], rel: str) -> list[str]:
    # Git's no-newline convention: a "\ No newline at end of file" marker means
    # the line BEFORE it has no terminator — the newline on that line exists
    # only for the diff's own framing. Resolve markers up front so the loop
    # below never sees them and body lines compare byte-exactly.
    resolved: list[str] = []
    for h in hunks:
        if h.startswith("\\"):
            if not resolved:
                raise PackPatchError(f"{rel}: dangling no-newline marker")
            resolved[-1] = resolved[-1].rstrip("\n")
        else:
            resolved.append(h)
    hunks = resolved

    out: list[str] = []
    pos = 0
    i = 0
    while i < len(hunks):
        line = hunks[i]
        if not line.startswith("@@"):
            raise PackPatchError(f"{rel}: malformed hunk header {line!r}")
        header = line.split("@@")[1].strip()
        old_span = header.split(" ")[0]
        start = int(old_span.lstrip("-").split(",")[0])
        # A zero-length old span addresses the line AFTER which to insert.
        old_len = int(old_span.split(",")[1]) if "," in old_span else 1
        anchor = start - 1 if old_len else start
        if anchor < pos:
            raise PackPatchError(f"{rel}: overlapping hunks")
        out.extend(base_lines[pos:anchor])
        pos = anchor
        i += 1
        while i < len(hunks) and not hunks[i].startswith("@@"):
            h = hunks[i]
            tag, body = h[0], h[1:]
            if tag == " ":
                if pos >= len(base_lines) or base_lines[pos] != body:
                    raise PackPatchError(f"{rel}: context mismatch at line {pos + 1}")
                out.append(body)
                pos += 1
            elif tag == "-":
                if pos >= len(base_lines) or base_lines[pos] != body:
                    raise PackPatchError(f"{rel}: removal mismatch at line {pos + 1}")
                pos += 1
            elif tag == "+":
                out.append(body)
            else:
                raise PackPatchError(f"{rel}: malformed hunk line {h!r}")
            i += 1
    out.extend(base_lines[pos:])
    return out


def apply(pack: Path, manifest: dict, diff_text: str) -> None:
    """Re-create ``pack/v2`` from a pristine pack plus the pair.

    Refuses to run against a base that does not match the manifest's hashes —
    a patch applied to the wrong snapshot must fail before it writes anything,
    not produce a plausible near-miss.
    """
    snapshot = Path(pack)
    pack = _pack_folder(snapshot)
    entries, base_files = _validate_manifest(snapshot, pack, manifest)
    hunks_by_path = _parse_diff(diff_text)

    expected_diff_paths = {
        entry["path"]
        for entry in entries
        if entry["op"] == "convert"
        or (entry["op"] == "add" and entry["v2_sha256"] != EMPTY_SHA256)
    }
    if set(hunks_by_path) != expected_diff_paths:
        missing = sorted(expected_diff_paths - set(hunks_by_path))
        extra = sorted(set(hunks_by_path) - expected_diff_paths)
        raise PackPatchError(
            f"diff target set differs: missing={missing}, extra={extra}"
        )
    by_path = {entry["path"]: entry for entry in entries}
    for rel, (source, _) in hunks_by_path.items():
        expected_source = "/dev/null" if by_path[rel]["op"] == "add" else f"a/{rel}"
        if source != expected_source:
            raise PackPatchError(
                f"diff source for {rel} is {source!r}, expected {expected_source!r}"
            )

    # Verify the whole base first; write nothing until it all checks out.
    for e in entries:
        if e["op"] in ("copy", "convert", "delete"):
            base = base_files[e["path"]]
            if _sha(base) != e["base_sha256"]:
                raise PackPatchError(
                    f"base file does not match the manifest: {e['path']} — "
                    f"this pair was generated against a different snapshot"
                )

    v2 = pack / "v2"
    if v2.exists() or v2.is_symlink():
        raise PackPatchError(f"refusing to replace existing conversion: {v2}")
    staging = Path(tempfile.mkdtemp(prefix=".v2-build-", dir=snapshot))
    try:
        for e in entries:
            rel, op = e["path"], e["op"]
            target = staging / rel
            if op == "delete":
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if op == "copy":
                shutil.copy2(base_files[rel], target)
            elif op == "add":
                hunks = hunks_by_path.get(rel, ("", []))[1]
                produced = "".join(_apply_hunks([], hunks, rel))
                target.write_bytes(produced.encode("utf-8"))
            elif op == "convert":
                base_lines = _lines(base_files[rel])
                produced = "".join(_apply_hunks(base_lines, hunks_by_path[rel][1], rel))
                target.write_bytes(produced.encode("utf-8"))
            else:
                raise PackPatchError(f"unknown op {op!r} for {rel}")
            target.chmod(e["mode"])
            if "v2_sha256" in e and _sha(target) != e["v2_sha256"]:
                raise PackPatchError(
                    f"applied result does not match the manifest: {rel}"
                )
        staging.replace(v2)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _artifact_stem(manifest: dict) -> str:
    pack = manifest.get("pack") if isinstance(manifest, dict) else None
    key = manifest.get("key") if isinstance(manifest, dict) else None
    if (
        not isinstance(pack, str)
        or not pack
        or pack in {".", ".."}
        or "/" in pack
        or "\\" in pack
    ):
        raise PackPatchError(f"invalid artifact pack name {pack!r}")
    if not isinstance(key, str) or re.fullmatch(r"x[0-9a-f]{7,12}", key) is None:
        raise PackPatchError(f"invalid artifact key {key!r}")
    return f"{pack}-{key}"


def _zip_member(name: str, data: bytes) -> tuple[zipfile.ZipInfo, bytes]:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    return info, data


def bundle(manifest: dict, diff_text: str) -> bytes:
    """Build the deterministic deployment zip containing only the pair."""
    if not isinstance(diff_text, str):
        raise PackPatchError("diff must be text")
    stem = _artifact_stem(manifest)
    manifest_bytes = (json.dumps(manifest, indent=1) + "\n").encode("utf-8")
    diff_bytes = diff_text.encode("utf-8")
    if max(len(manifest_bytes), len(diff_bytes)) > MAX_BUNDLE_MEMBER_BYTES:
        raise PackPatchError("patch-pair member exceeds the deployment size limit")
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as archive:
        for info, data in (
            _zip_member(f"{stem}.json", manifest_bytes),
            _zip_member(f"{stem}.diff", diff_bytes),
        ):
            archive.writestr(info, data)
    return out.getvalue()


def _read_bundle(artifact: bytes) -> tuple[dict, str]:
    if not isinstance(artifact, bytes):
        raise PackPatchError("deployment artifact must be bytes")
    if len(artifact) > MAX_BUNDLE_BYTES:
        raise PackPatchError("deployment artifact exceeds the size limit")
    try:
        with zipfile.ZipFile(io.BytesIO(artifact)) as archive:
            infos = archive.infolist()
            if len(infos) != 2 or len({info.filename for info in infos}) != 2:
                raise PackPatchError("deployment zip must contain exactly two files")
            for info in infos:
                if (
                    info.is_dir()
                    or "/" in info.filename
                    or "\\" in info.filename
                    or info.flag_bits & 1
                ):
                    raise PackPatchError(
                        f"invalid deployment zip member {info.filename!r}"
                    )
                if (
                    info.file_size > MAX_BUNDLE_MEMBER_BYTES
                    or info.compress_size > MAX_BUNDLE_MEMBER_BYTES
                ):
                    raise PackPatchError(
                        f"deployment zip member is too large: {info.filename}"
                    )
            names = {info.filename for info in infos}
            json_names = [name for name in names if name.endswith(".json")]
            diff_names = [name for name in names if name.endswith(".diff")]
            if len(json_names) != 1 or len(diff_names) != 1:
                raise PackPatchError(
                    "deployment zip must contain one .json and one .diff"
                )
            manifest = json.loads(archive.read(json_names[0]).decode("utf-8"))
            expected_stem = _artifact_stem(manifest)
            expected_names = {f"{expected_stem}.json", f"{expected_stem}.diff"}
            if names != expected_names:
                raise PackPatchError(
                    f"deployment zip names {sorted(names)} do not match the manifest"
                )
            diff_text = archive.read(diff_names[0]).decode("utf-8")
    except PackPatchError:
        raise
    except (
        json.JSONDecodeError,
        UnicodeDecodeError,
        zipfile.BadZipFile,
        RuntimeError,
    ) as exc:
        raise PackPatchError(f"invalid deployment zip: {exc}") from exc
    return manifest, diff_text


def apply_bundle(pack: Path, artifact: bytes | Path) -> None:
    """Verify and apply a deployment zip to a pristine pack snapshot."""
    data = Path(artifact).read_bytes() if isinstance(artifact, Path) else artifact
    manifest, diff_text = _read_bundle(data)
    apply(pack, manifest, diff_text)


def validate_bundle(pack: Path, artifact: bytes | Path) -> None:
    """Verify that an existing materialized conversion matches its bundle."""
    data = Path(artifact).read_bytes() if isinstance(artifact, Path) else artifact
    manifest, diff_text = _read_bundle(data)
    snapshot = Path(pack)
    pack_root = _pack_folder(snapshot)
    entries, base_files = _validate_manifest(
        snapshot,
        pack_root,
        manifest,
        allow_runtime_products=True,
    )
    hunks_by_path = _parse_diff(diff_text)
    expected_diff_paths = {
        entry["path"]
        for entry in entries
        if entry["op"] == "convert"
        or (entry["op"] == "add" and entry["v2_sha256"] != EMPTY_SHA256)
    }
    if set(hunks_by_path) != expected_diff_paths:
        missing = sorted(expected_diff_paths - set(hunks_by_path))
        extra = sorted(set(hunks_by_path) - expected_diff_paths)
        raise PackPatchError(
            f"diff target set differs: missing={missing}, extra={extra}"
        )
    by_path = {entry["path"]: entry for entry in entries}
    for rel, (source, _) in hunks_by_path.items():
        expected_source = "/dev/null" if by_path[rel]["op"] == "add" else f"a/{rel}"
        if source != expected_source:
            raise PackPatchError(
                f"diff source for {rel} is {source!r}, expected {expected_source!r}"
            )
    for entry in entries:
        if entry["op"] in {"copy", "convert", "delete"}:
            if _sha(base_files[entry["path"]]) != entry["base_sha256"]:
                raise PackPatchError(
                    f"base file does not match the manifest: {entry['path']}"
                )

    v2 = pack_root / "v2"
    if not v2.is_dir() or v2.is_symlink():
        raise PackPatchError(f"materialized pack has no regular v2 tree: {v2}")
    converted = _files(v2)
    expected_paths = {entry["path"] for entry in entries if entry["op"] != "delete"}
    if set(converted) != expected_paths:
        missing = sorted(expected_paths - set(converted))
        extra = sorted(set(converted) - expected_paths)
        raise PackPatchError(
            f"materialized file set differs: missing={missing}, extra={extra}"
        )
    for entry in entries:
        rel = entry["path"]
        if entry["op"] == "delete":
            continue
        target = converted[rel]
        expected_sha = entry.get("v2_sha256", entry.get("base_sha256"))
        if _sha(target) != expected_sha or _mode(target) != entry["mode"]:
            raise PackPatchError(f"materialized file differs from manifest: {rel}")
        if entry["op"] not in {"add", "convert"}:
            continue
        hunks = hunks_by_path.get(rel, ("", []))[1]
        base_lines = [] if entry["op"] == "add" else _lines(base_files[rel])
        produced = "".join(_apply_hunks(base_lines, hunks, rel)).encode("utf-8")
        if produced != target.read_bytes():
            raise PackPatchError(f"diff does not reproduce materialized file: {rel}")


def validate_tree(actual: Path | str, expected: Path | str) -> None:
    """Require two pack trees to have identical files, bytes, and modes."""
    actual_root = Path(actual).resolve()
    expected_root = Path(expected).resolve()
    if not actual_root.is_dir() or not expected_root.is_dir():
        raise PackPatchError("validation diff requires two pack directories")
    actual_files = _files(actual_root)
    expected_files = _files(expected_root)
    actual_paths = set(actual_files)
    expected_paths = set(expected_files)
    missing = sorted(expected_paths - actual_paths)
    extra = sorted(actual_paths - expected_paths)
    shared = sorted(actual_paths & expected_paths)
    changed = [
        path
        for path in shared
        if _sha(actual_files[path]) != _sha(expected_files[path])
    ]
    modes = [
        path
        for path in shared
        if _mode(actual_files[path]) != _mode(expected_files[path])
    ]
    if missing or extra or changed or modes:
        raise PackPatchError(
            "materialized conversion differs from debug reference: "
            f"missing={missing}, extra={extra}, changed={changed}, modes={modes}"
        )


def main(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Generate or apply a pack's .json/.diff pair"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("pack", type=Path)
    g.add_argument("out_dir", type=Path)
    a = sub.add_parser("apply")
    a.add_argument("pack", type=Path)
    a.add_argument(
        "pair_prefix", type=Path, help="path up to the extension: <dir>/<Pack-Name>"
    )
    b = sub.add_parser("bundle")
    b.add_argument(
        "pair_prefix", type=Path, help="path up to the extension: <dir>/<Pack-Name>"
    )
    b.add_argument("output", type=Path, nargs="?")
    z = sub.add_parser("apply-zip")
    z.add_argument("pack", type=Path)
    z.add_argument("artifact", type=Path)
    args = ap.parse_args(argv)

    if args.cmd == "generate":
        manifest, diff_text = generate(args.pack)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        base = args.out_dir / f"{manifest['pack']}-{manifest['key']}"
        base.with_suffix(".json").write_bytes(
            (json.dumps(manifest, indent=1) + "\n").encode("utf-8")
        )
        base.with_suffix(".diff").write_bytes(diff_text.encode("utf-8"))
        c = manifest["counts"]
        print(
            f"{manifest['pack']} {manifest['key']}: "
            f"{c['convert']} converted, {c['add']} added, "
            f"{c['copy']} copied, {c['delete']} deleted -> "
            f"{base.with_suffix('.json').name}, {base.with_suffix('.diff').name}"
        )
        return 0

    if args.cmd == "bundle":
        manifest = json.loads(
            args.pair_prefix.with_suffix(".json").read_bytes().decode("utf-8")
        )
        diff_text = args.pair_prefix.with_suffix(".diff").read_bytes().decode("utf-8")
        output = args.output or args.pair_prefix.with_suffix(".zip")
        output.write_bytes(bundle(manifest, diff_text))
        print(f"wrote {output}")
        return 0

    if args.cmd == "apply-zip":
        apply_bundle(args.pack, args.artifact)
        print(f"re-created {_pack_folder(args.pack) / 'v2'} from {args.artifact}")
        return 0

    manifest = json.loads(
        args.pair_prefix.with_suffix(".json").read_bytes().decode("utf-8")
    )
    diff_text = args.pair_prefix.with_suffix(".diff").read_bytes().decode("utf-8")
    apply(args.pack, manifest, diff_text)
    print(f"re-created {_pack_folder(args.pack) / 'v2'} from the pair")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
