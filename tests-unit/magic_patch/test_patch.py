"""The pack distribution pair: generate ↔ apply must be a byte-exact loop.

The pair (`<Pack>-<key>.json` + `.diff`) is the ONLY thing a converted pack
ships as, so the property under test is total: applying the pair to a pristine
copy of the original re-creates the v2 tree byte for byte — and applying it to
anything that is NOT the pristine original refuses before writing a single
file. A patch that half-applies to the wrong snapshot would produce a
plausible near-miss of a security boundary, which is the worst artifact this
system could emit.

Run:  <venv-python> -m pytest backend/tests/test_packpatch.py -q
"""

from __future__ import annotations

import json
import io
import os
import pathlib
import shutil
import stat
import sys
import zipfile

import pytest

BACKEND = pathlib.Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
for path in (str(REPO), str(BACKEND)):
    if path not in sys.path:
        sys.path.insert(0, path)

from tools.magic_patch import patch as packpatch  # noqa: E402


def _mkpack(root: pathlib.Path) -> pathlib.Path:
    """A miniature pack exercising every op the format defines.

    v2/ is a COMPLETE clone of the pack plus the conversion's changes — that
    is what applying the patch to v1 produces, and it is why a node module in
    v2/ sits beside every sibling and resource it had upstream.
    """
    snapshot = root / "Mini-Pack" / "xabc1234"
    pack = snapshot / "Mini-Pack-HEAD"
    (pack / "nodes").mkdir(parents=True)
    (pack / "fonts").mkdir()

    (pack / "nodes" / "a.py").write_text(
        "import folder_paths\n\ndef run():\n    return folder_paths.x\n"
    )
    (pack / "nodes" / "same.py").write_text("VALUE = 1\n")
    (pack / "nodes" / "gone.py").write_text("legacy = True\n")
    (pack / "fonts" / "f.bin").write_bytes(b"\x00\x01\x02binary")

    v2 = pack / "v2"
    (v2 / "nodes").mkdir(parents=True)
    (v2 / "fonts").mkdir()
    # convert: a.py edited at the boundary
    (v2 / "nodes" / "a.py").write_text(
        "from comfy_api.latest import sdk\n\ndef run():\n    return sdk.x\n"
    )
    # add: no counterpart in the pack — and no trailing newline, the case
    # difflib mishandles silently (it emits no marker and no terminator)
    (v2 / "nodes" / "helper.py").write_text("SHARED = True")
    # copy: cloned unchanged, text and binary alike
    (v2 / "nodes" / "same.py").write_text("VALUE = 1\n")
    (v2 / "fonts" / "f.bin").write_bytes(b"\x00\x01\x02binary")
    # delete: gone.py deliberately has no v2 counterpart
    return snapshot


def test_round_trip_is_byte_exact(tmp_path):
    pack = _mkpack(tmp_path)
    manifest, diff_text = packpatch.generate(pack)

    assert manifest["key"] == "xabc1234"
    ops = {e["path"]: e["op"] for e in manifest["files"]}
    assert ops == {
        "nodes/a.py": "convert",
        "nodes/helper.py": "add",
        "nodes/same.py": "copy",
        "nodes/gone.py": "delete",
        "fonts/f.bin": "copy",
    }

    # Fresh pristine copy, no v2 — the distribution scenario.
    fresh = tmp_path / "fresh" / "Mini-Pack" / "xabc1234"
    shutil.copytree(pack, fresh, ignore=shutil.ignore_patterns("v2"))
    packpatch.apply(fresh, manifest, diff_text)

    original_v2 = sorted(
        (p.relative_to(pack / "Mini-Pack-HEAD" / "v2").as_posix(), p.read_bytes())
        for p in (pack / "Mini-Pack-HEAD" / "v2").rglob("*")
        if p.is_file()
    )
    produced_v2 = sorted(
        (p.relative_to(fresh / "Mini-Pack-HEAD" / "v2").as_posix(), p.read_bytes())
        for p in (fresh / "Mini-Pack-HEAD" / "v2").rglob("*")
        if p.is_file()
    )
    assert produced_v2 == original_v2


def test_wrong_snapshot_refused_before_any_write(tmp_path):
    pack = _mkpack(tmp_path)
    manifest, diff_text = packpatch.generate(pack)

    fresh = tmp_path / "fresh" / "Mini-Pack" / "xabc1234"
    shutil.copytree(pack, fresh, ignore=shutil.ignore_patterns("v2"))
    (fresh / "Mini-Pack-HEAD" / "nodes" / "a.py").write_text("tampered\n")

    with pytest.raises(packpatch.PackPatchError, match="different snapshot"):
        packpatch.apply(fresh, manifest, diff_text)
    assert not (fresh / "Mini-Pack-HEAD" / "v2").exists(), (
        "a refused apply must not leave a partial v2 tree behind"
    )


def test_changed_binary_is_refused_at_generation(tmp_path):
    pack = _mkpack(tmp_path)
    binary = pack / "Mini-Pack-HEAD" / "v2" / "fonts" / "f.bin"
    binary.write_bytes(b"\x00\x01\x03different")
    with pytest.raises(packpatch.PackPatchError, match="binary"):
        packpatch.generate(pack)


def test_identical_file_is_a_manifest_copy_not_diff_content(tmp_path):
    pack = _mkpack(tmp_path)
    manifest, diff_text = packpatch.generate(pack)
    entry = next(e for e in manifest["files"] if e["path"] == "nodes/same.py")
    assert entry["op"] == "copy"
    assert "nodes/same.py" not in diff_text


def test_folder_without_snapshot_key_is_refused(tmp_path):
    pack = tmp_path / "Some-Pack"
    (pack / "Mini-Pack-HEAD" / "v2").mkdir(parents=True)
    with pytest.raises(packpatch.PackPatchError, match="snapshot key"):
        packpatch.generate(pack)


def test_failed_application_leaves_no_partial_v2_tree(tmp_path):
    pack = _mkpack(tmp_path)
    manifest, diff_text = packpatch.generate(pack)
    helper = next(e for e in manifest["files"] if e["path"] == "nodes/helper.py")
    helper["v2_sha256"] = "0" * 64

    fresh = tmp_path / "fresh" / "Mini-Pack" / "xabc1234"
    shutil.copytree(pack, fresh, ignore=shutil.ignore_patterns("v2"))

    with pytest.raises(packpatch.PackPatchError, match="applied result"):
        packpatch.apply(fresh, manifest, diff_text)
    assert not (fresh / "Mini-Pack-HEAD" / "v2").exists()


def test_manifest_identity_must_match_destination_snapshot(tmp_path):
    pack = _mkpack(tmp_path)
    manifest, diff_text = packpatch.generate(pack)
    manifest["key"] = "xdeadbee"
    fresh = tmp_path / "fresh" / "Mini-Pack" / "xabc1234"
    shutil.copytree(pack, fresh, ignore=shutil.ignore_patterns("v2"))

    with pytest.raises(packpatch.PackPatchError, match="manifest key"):
        packpatch.apply(fresh, manifest, diff_text)
    assert not (fresh / "Mini-Pack-HEAD" / "v2").exists()


def test_manifest_cannot_omit_a_base_file(tmp_path):
    pack = _mkpack(tmp_path)
    manifest, diff_text = packpatch.generate(pack)
    manifest["files"] = [
        entry for entry in manifest["files"] if entry["path"] != "nodes/same.py"
    ]
    fresh = tmp_path / "fresh" / "Mini-Pack" / "xabc1234"
    shutil.copytree(pack, fresh, ignore=shutil.ignore_patterns("v2"))

    with pytest.raises(packpatch.PackPatchError, match="base file set"):
        packpatch.apply(fresh, manifest, diff_text)
    assert not (fresh / "Mini-Pack-HEAD" / "v2").exists()


def test_manifest_paths_cannot_escape_the_pack(tmp_path):
    pack = _mkpack(tmp_path)
    manifest, diff_text = packpatch.generate(pack)
    helper = next(
        entry for entry in manifest["files"] if entry["path"] == "nodes/helper.py"
    )
    helper["path"] = "../../escaped.py"
    diff_text = diff_text.replace(
        "+++ b/v2/nodes/helper.py", "+++ b/v2/../../escaped.py"
    )
    fresh = tmp_path / "fresh" / "Mini-Pack" / "xabc1234"
    shutil.copytree(pack, fresh, ignore=shutil.ignore_patterns("v2"))

    with pytest.raises(packpatch.PackPatchError, match="unsafe manifest path"):
        packpatch.apply(fresh, manifest, diff_text)
    assert not (fresh / "escaped.py").exists()
    assert not (fresh / "Mini-Pack-HEAD" / "v2").exists()


def test_duplicate_manifest_paths_are_refused(tmp_path):
    pack = _mkpack(tmp_path)
    manifest, diff_text = packpatch.generate(pack)
    manifest["files"].append(dict(manifest["files"][0]))
    fresh = tmp_path / "fresh" / "Mini-Pack" / "xabc1234"
    shutil.copytree(pack, fresh, ignore=shutil.ignore_patterns("v2"))

    with pytest.raises(packpatch.PackPatchError, match="duplicate manifest path"):
        packpatch.apply(fresh, manifest, diff_text)


def test_diff_targets_and_source_labels_are_bound_to_manifest(tmp_path):
    pack = _mkpack(tmp_path)
    manifest, diff_text = packpatch.generate(pack)
    fresh = tmp_path / "fresh" / "Mini-Pack" / "xabc1234"
    shutil.copytree(pack, fresh, ignore=shutil.ignore_patterns("v2"))

    extra = "--- /dev/null\n+++ b/v2/nodes/unlisted.py\n@@ -0,0 +1 @@\n+x\n"
    with pytest.raises(packpatch.PackPatchError, match="diff target set"):
        packpatch.apply(fresh, manifest, diff_text + extra)

    wrong_source = diff_text.replace("--- a/nodes/a.py", "--- a/nodes/not-a.py", 1)
    with pytest.raises(packpatch.PackPatchError, match="diff source"):
        packpatch.apply(fresh, manifest, wrong_source)


def test_symlinks_are_not_pack_patch_content(tmp_path):
    pack = _mkpack(tmp_path)
    external = tmp_path / "external.py"
    external.write_text("outside = True\n")
    os.symlink(external, pack / "Mini-Pack-HEAD" / "nodes" / "link.py")
    with pytest.raises(packpatch.PackPatchError, match="symbolic link"):
        packpatch.generate(pack)


def test_added_file_mode_round_trips(tmp_path):
    pack = _mkpack(tmp_path)
    helper = pack / "Mini-Pack-HEAD" / "v2" / "nodes" / "helper.py"
    helper.chmod(0o755)
    manifest, diff_text = packpatch.generate(pack)
    fresh = tmp_path / "fresh" / "Mini-Pack" / "xabc1234"
    shutil.copytree(pack, fresh, ignore=shutil.ignore_patterns("v2"))

    packpatch.apply(fresh, manifest, diff_text)
    made = fresh / "Mini-Pack-HEAD" / "v2" / "nodes" / "helper.py"
    assert stat.S_IMODE(made.stat().st_mode) == 0o755


def test_deployment_zip_contains_only_the_pair_and_applies(tmp_path):
    pack = _mkpack(tmp_path)
    manifest, diff_text = packpatch.generate(pack)
    artifact = packpatch.bundle(manifest, diff_text)
    assert artifact == packpatch.bundle(manifest, diff_text)

    with zipfile.ZipFile(io.BytesIO(artifact)) as archive:
        assert archive.namelist() == [
            "Mini-Pack-xabc1234.json",
            "Mini-Pack-xabc1234.diff",
        ]

    fresh = tmp_path / "fresh" / "Mini-Pack" / "xabc1234"
    shutil.copytree(pack, fresh, ignore=shutil.ignore_patterns("v2"))
    packpatch.apply_bundle(fresh, artifact)

    want = {
        path.relative_to(pack / "Mini-Pack-HEAD" / "v2").as_posix(): path.read_bytes()
        for path in (pack / "Mini-Pack-HEAD" / "v2").rglob("*")
        if path.is_file()
    }
    made = fresh / "Mini-Pack-HEAD" / "v2"
    got = {
        path.relative_to(made).as_posix(): path.read_bytes()
        for path in made.rglob("*")
        if path.is_file()
    }
    assert got == want


def test_deployment_zip_rejects_extra_members(tmp_path):
    pack = _mkpack(tmp_path)
    manifest, diff_text = packpatch.generate(pack)
    artifact = io.BytesIO(packpatch.bundle(manifest, diff_text))
    rewritten = io.BytesIO()
    with zipfile.ZipFile(artifact) as source, zipfile.ZipFile(rewritten, "w") as target:
        for info in source.infolist():
            target.writestr(info.filename, source.read(info))
        target.writestr("extra.py", "unexpected")

    fresh = tmp_path / "fresh" / "Mini-Pack" / "xabc1234"
    shutil.copytree(pack, fresh, ignore=shutil.ignore_patterns("v2"))
    with pytest.raises(packpatch.PackPatchError, match="exactly two files"):
        packpatch.apply_bundle(fresh, rewritten.getvalue())
    assert not (fresh / "Mini-Pack-HEAD" / "v2").exists()


def test_apply_zip_cli_reports_the_created_pack_path(tmp_path, capsys):
    pack = _mkpack(tmp_path)
    manifest, diff_text = packpatch.generate(pack)
    artifact = tmp_path / "Mini-Pack-xabc1234.zip"
    artifact.write_bytes(packpatch.bundle(manifest, diff_text))
    fresh = tmp_path / "fresh" / "Mini-Pack" / "xabc1234"
    shutil.copytree(pack, fresh, ignore=shutil.ignore_patterns("v2"))

    assert packpatch.main(["apply-zip", str(fresh), str(artifact)]) == 0
    assert str(fresh / "Mini-Pack-HEAD" / "v2") in capsys.readouterr().out


def test_real_pack_pair_round_trips(tmp_path):
    """The actual KJNodes pack, snapshotted, through the full loop.

    Snapshotted first because the live tree is being translated file by file —
    the property must hold for whatever state the tree is in, so the test
    freezes one state and proves the loop on it.
    """
    src = REPO / "pack-db" / "packs" / "comfyui-kjnodes" / "x3f20054"
    if not src.is_dir():
        pytest.skip("KJNodes pack not present")
    snap = tmp_path / "db" / "comfyui-kjnodes" / "x3f20054"
    shutil.copytree(
        src, snap, ignore=shutil.ignore_patterns("__pycache__", ".DS_Store")
    )

    manifest, diff_text = packpatch.generate(snap)
    assert manifest["key"] == "x3f20054"
    assert manifest["counts"]["convert"] >= 27

    fresh = tmp_path / "fresh" / "comfyui-kjnodes" / "x3f20054"
    shutil.copytree(snap, fresh, ignore=shutil.ignore_patterns("v2"))
    packpatch.apply(fresh, manifest, diff_text)

    want = {
        p.relative_to(snap / "ComfyUI-KJNodes-HEAD" / "v2").as_posix(): p.read_bytes()
        for p in (snap / "ComfyUI-KJNodes-HEAD" / "v2").rglob("*")
        if p.is_file()
        if "__pycache__" not in p.parts and p.name != ".DS_Store"
    }
    made = fresh / "ComfyUI-KJNodes-HEAD" / "v2"
    got = {
        p.relative_to(made).as_posix(): p.read_bytes()
        for p in made.rglob("*")
        if p.is_file()
    }
    assert got == want


def test_checked_in_real_pack_pair_is_fresh():
    snapshot = REPO / "pack-db" / "packs" / "comfyui-kjnodes" / "x3f20054"
    pair = (
        REPO
        / "pack-db"
        / "patches"
        / "comfyui-kjnodes"
        / "x3f20054"
        / "comfyui-kjnodes-x3f20054"
    )
    if not snapshot.is_dir():
        pytest.skip("KJNodes pack not present")

    manifest, diff_text = packpatch.generate(snapshot)
    assert json.loads(pair.with_suffix(".json").read_text()) == manifest
    assert pair.with_suffix(".diff").read_text() == diff_text
