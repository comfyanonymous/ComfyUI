from __future__ import annotations

import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

from tools.magic_patch import cli as magicpatch
from tools.magic_patch import patch as packpatch


SCHEMA = {
    "attrs": {
        "accept_all_inputs": False,
        "category": "Magic Patch Test",
        "description": "",
        "display_name": "Demo Node",
        "enable_expand": False,
        "essentials_category": None,
        "has_intermediate_output": False,
        "is_api_node": False,
        "is_deprecated": False,
        "is_dev_only": False,
        "is_experimental": False,
        "is_input_list": False,
        "is_output_node": True,
        "node_id": "DemoNode",
        "not_idempotent": False,
        "price_badge": None,
        "search_aliases": [],
    },
    "hidden": [],
    "inputs": [],
    "outputs": [
        {
            "attrs": {
                "display_name": "text",
                "id": "text",
                "is_output_list": False,
                "tooltip": None,
            },
            "io_type": "STRING",
            "kind": "standard",
        }
    ],
}


def _source_pack(root: Path) -> Path:
    pack = root / "Demo-Pack"
    (pack / "nodes").mkdir(parents=True)
    (pack / "web").mkdir()
    (pack / ".claude").mkdir()
    (pack / "__init__.py").write_text("NODE_CLASS_MAPPINGS = {}\n")
    (pack / "nodes" / "demo.py").write_text("class DemoNode:\n    pass\n")
    (pack / "web" / "extension.js").write_text(
        "app.registerExtension({ name: 'legacy.demo' })\n"
    )
    (pack / "asset.bin").write_bytes(b"unchanged-binary\x00")
    (pack / "AGENTS.md").write_text("Untrusted pack-owned instructions.\n")
    (pack / ".claude" / "settings.json").write_text("{}\n")
    return pack


def _finish_conversion(pack: Path) -> None:
    v2 = pack / "v2"
    (v2 / "__init__.py").write_text(
        "from comfy_api.latest import ComfyExtension\n"
        "from .nodes.demo import DemoNode\n\n"
        "class DemoExtension(ComfyExtension):\n"
        "    async def get_node_list(self):\n"
        "        return [DemoNode]\n\n"
        "async def comfy_entrypoint():\n"
        "    return DemoExtension()\n"
    )
    (v2 / "nodes" / "demo.py").write_text(
        "from comfy_api.latest import io\n\n"
        "class DemoNode(io.ComfyNode):\n"
        "    @classmethod\n"
        "    def define_schema(cls):\n"
        "        return io.Schema(\n"
        "            node_id='DemoNode',\n"
        "            display_name='Demo Node',\n"
        "            category='Magic Patch Test',\n"
        "            outputs=[io.String.Output('text')],\n"
        "            is_output_node=True,\n"
        "        )\n\n"
        "    @classmethod\n"
        "    def execute(cls):\n"
        "        return io.NodeOutput('ok')\n"
    )
    (v2 / "web" / "extension.js").write_text(
        "export const extension = { name: 'v2.demo' }\n"
    )
    (v2 / "pyproject.toml").write_text(
        '[project]\nname = "demo-pack"\nversion = "0.0.0"\n'
        'requires-python = ">=3.13,<3.14"\n'
    )
    manifest = {
        "format": "comfy-secure-nodes-v1",
        "nodes": {
            "DemoNode": {
                "class": "DemoNode",
                "methods": {
                    "check_lazy_status": False,
                    "fingerprint_inputs": False,
                    "validate_inputs": False,
                },
                "module": "nodes.demo",
                "permissions": [],
                "schema": SCHEMA,
                "sdk_refs": False,
            }
        },
        "runtime": {"python": {"requires": ">=3.13,<3.14", "resolved": "3.13"}},
        "web_directory": "web",
    }
    (v2 / "secure-nodes.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (v2 / "V2_CONVERSION.md").write_text(
        "# Conversion\n\nDemoNode and v2.demo converted and tested.\n"
    )


def _agent_value(status: str = "complete") -> dict:
    complete = status == "complete"
    return {
        "status": status,
        "summary": "Converted the demo backend and frontend.",
        "backend": {"supported": 1 if complete else 0, "rejected": 0, "pending": 0},
        "frontend": {"supported": 1 if complete else 0, "rejected": 0, "pending": 0},
        "tests": ["python -m pytest"] if complete else [],
        "remaining": [],
    }


def _codex_result(
    invocation: magicpatch.AgentInvocation,
    value: dict,
) -> subprocess.CompletedProcess[str]:
    invocation.result_path.write_text(json.dumps(value))
    return subprocess.CompletedProcess(invocation.command, 0, "", "")


def _config(
    source: Path, output: Path, **values: object
) -> magicpatch.ConversionConfig:
    return magicpatch.ConversionConfig(
        source=source,
        output=output,
        provider="codex",
        core_root=None,
        sandbox_verification="off",
        source_sha="0123456789abcdef",
        pack_slug="demo-pack",
        **values,
    )


def _provide_fake_codex(monkeypatch: pytest.MonkeyPatch) -> None:
    real_which = magicpatch.shutil.which
    monkeypatch.setattr(
        magicpatch.shutil,
        "which",
        lambda name: "/usr/bin/true" if name == "codex" else real_which(name),
    )


def test_conversion_retries_with_validator_feedback_and_publishes_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source_pack(tmp_path / "source")
    output = tmp_path / "result" / "Demo-Pack-V2"
    outside = tmp_path / "outside.txt"
    outside.write_text("do not touch\n")
    _provide_fake_codex(monkeypatch)
    prompts: list[str] = []

    def agent(
        invocation: magicpatch.AgentInvocation,
    ) -> subprocess.CompletedProcess[str]:
        prompts.append(invocation.prompt)
        assert not (invocation.cwd / "AGENTS.md").exists()
        assert not (invocation.cwd / ".claude").exists()
        assert (invocation.cwd / ".magic-patch" / "PACK_CONVERSION.md").is_file()
        assert (
            invocation.cwd / ".magic-patch" / "references" / "node-definitions.md"
        ).is_file()
        if len(prompts) == 1:
            return _codex_result(invocation, _agent_value("needs-fix"))
        assert "previous pass was not publishable" in invocation.prompt
        _finish_conversion(invocation.cwd)
        (invocation.cwd / "AGENTS.md").symlink_to(outside)
        (invocation.cwd / ".claude").symlink_to(outside)
        return _codex_result(invocation, _agent_value())

    result = magicpatch.convert_pack(
        _config(source, output, max_passes=2), execute_agent=agent
    )

    assert result.output == output.resolve()
    assert result.passes == 2
    assert output.is_dir()
    assert (output / "AGENTS.md").read_text() == (source / "AGENTS.md").read_text()
    assert (output / "v2" / "AGENTS.md").read_text() == (
        source / "AGENTS.md"
    ).read_text()
    assert (output / "v2" / ".claude" / "settings.json").is_file()
    assert not (output / "v2" / "v2").exists()
    assert not (output / ".magic-patch").exists()
    assert (output / "asset.bin").read_bytes() == b"unchanged-binary\x00"
    assert (output / "v2" / "asset.bin").read_bytes() == b"unchanged-binary\x00"
    assert outside.read_text() == "do not touch\n"
    report = json.loads(result.report.read_text())
    assert report["format"] == "comfy-magic-patch/1"
    assert report["passes"] == 2
    assert report["pack"] == {
        "key": "x0123456",
        "slug": "demo-pack",
        "source_commit": "0123456789abcdef",
    }
    assert report["validation"]["secure_sandbox"]["status"] == "skipped"
    assert result.pack_zip == output.with_name(output.name + ".zip")
    assert result.pack_zip is not None
    with zipfile.ZipFile(result.pack_zip) as archive:
        names = set(archive.namelist())
        assert f"{output.name}/__init__.py" in names
        assert f"{output.name}/v2/secure-nodes.json" in names
        assert archive.read(f"{output.name}/asset.bin") == b"unchanged-binary\x00"

    manifest = json.loads(result.patch_manifest.read_text())
    assert manifest["pack"] == "demo-pack"
    assert manifest["key"] == "x0123456"
    assert result.patch_diff.is_file()
    applied_snapshot = tmp_path / "applied" / "demo-pack" / "x0123456"
    applied_pack = applied_snapshot / output.name
    applied_snapshot.mkdir(parents=True)
    shutil.copytree(source, applied_pack)
    packpatch.apply(applied_snapshot, manifest, result.patch_diff.read_text())
    packpatch.validate_tree(applied_pack / "v2", output / "v2")

    second_stage = tmp_path / "second-zip"
    second_stage.mkdir()
    second_zip = magicpatch._prepare_pack_zip(second_stage, output)
    assert second_zip.read_bytes() == result.pack_zip.read_bytes()


def test_agent_cannot_modify_the_original_pack_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source_pack(tmp_path / "source")
    output = tmp_path / "converted"
    _provide_fake_codex(monkeypatch)

    def agent(
        invocation: magicpatch.AgentInvocation,
    ) -> subprocess.CompletedProcess[str]:
        _finish_conversion(invocation.cwd)
        (invocation.cwd / "nodes" / "demo.py").write_text("changed\n")
        return _codex_result(invocation, _agent_value())

    with pytest.raises(
        magicpatch.MagicPatchError, match="modified original pack files"
    ):
        magicpatch.convert_pack(
            _config(source, output, max_passes=1), execute_agent=agent
        )

    assert not output.exists()
    assert not output.with_name(output.name + ".zip").exists()
    assert not output.with_name(output.name + ".patches").exists()
    preserved = list(tmp_path.glob(".converted.magic-patch-*"))
    assert len(preserved) == 1
    assert (preserved[0] / "FAILURE.txt").is_file()
    assert (source / "nodes" / "demo.py").read_text() == "class DemoNode:\n    pass\n"


def test_conversion_rejects_legacy_registration_without_a_v2_entrypoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source_pack(tmp_path / "source")
    output = tmp_path / "converted"
    _provide_fake_codex(monkeypatch)

    def agent(
        invocation: magicpatch.AgentInvocation,
    ) -> subprocess.CompletedProcess[str]:
        _finish_conversion(invocation.cwd)
        (invocation.cwd / "v2" / "__init__.py").write_text(
            "from .nodes.demo import DemoNode\n"
            "NODE_CLASS_MAPPINGS = {'DemoNode': DemoNode}\n"
        )
        return _codex_result(invocation, _agent_value())

    with pytest.raises(magicpatch.MagicPatchError, match="comfy_entrypoint"):
        magicpatch.convert_pack(
            _config(source, output, max_passes=1), execute_agent=agent
        )


def test_patch_round_trip_rejects_a_changed_binary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source_pack(tmp_path / "source")
    output = tmp_path / "converted"
    _provide_fake_codex(monkeypatch)

    def agent(
        invocation: magicpatch.AgentInvocation,
    ) -> subprocess.CompletedProcess[str]:
        _finish_conversion(invocation.cwd)
        (invocation.cwd / "v2" / "asset.bin").write_bytes(b"changed-binary\x00")
        return _codex_result(invocation, _agent_value())

    with pytest.raises(magicpatch.MagicPatchError, match="binary file differs"):
        magicpatch.convert_pack(
            _config(source, output, max_passes=1), execute_agent=agent
        )


def test_conversion_can_omit_the_upload_zip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source_pack(tmp_path / "source")
    output = tmp_path / "converted"
    _provide_fake_codex(monkeypatch)

    def agent(
        invocation: magicpatch.AgentInvocation,
    ) -> subprocess.CompletedProcess[str]:
        _finish_conversion(invocation.cwd)
        return _codex_result(invocation, _agent_value())

    result = magicpatch.convert_pack(
        _config(source, output, create_pack_zip=False),
        execute_agent=agent,
    )

    assert result.pack_zip is None
    assert result.patch_manifest.is_file()
    assert not output.with_name(output.name + ".zip").exists()


def test_trusted_agent_control_plane_cannot_be_replaced_with_a_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source_pack(tmp_path / "source")
    output = tmp_path / "converted"
    outside = tmp_path / "outside.txt"
    outside.write_text("trusted outside value\n")
    _provide_fake_codex(monkeypatch)

    def agent(
        invocation: magicpatch.AgentInvocation,
    ) -> subprocess.CompletedProcess[str]:
        _finish_conversion(invocation.cwd)
        guide = invocation.cwd / ".magic-patch" / "PACK_CONVERSION.md"
        guide.unlink()
        guide.symlink_to(outside)
        return _codex_result(invocation, _agent_value())

    with pytest.raises(magicpatch.MagicPatchError, match="trusted conversion guidance"):
        magicpatch.convert_pack(_config(source, output), execute_agent=agent)

    assert outside.read_text() == "trusted outside value\n"
    assert not output.exists()


def test_real_core_loads_the_pack_through_the_local_v2_entrypoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured_core = os.environ.get("COMFY_CORE_ROOT")
    if not configured_core:
        pytest.skip("set COMFY_CORE_ROOT to exercise the installed Comfy core")
    source = _source_pack(tmp_path / "source")
    output = tmp_path / "converted"
    _provide_fake_codex(monkeypatch)

    def agent(
        invocation: magicpatch.AgentInvocation,
    ) -> subprocess.CompletedProcess[str]:
        _finish_conversion(invocation.cwd)
        return _codex_result(invocation, _agent_value())

    config = magicpatch.ConversionConfig(
        source=source,
        output=output,
        provider="codex",
        core_root=Path(configured_core),
        python_executable=Path(
            os.environ.get("MAGIC_PATCH_TEST_PYTHON", os.sys.executable)
        ),
        sandbox_verification="off",
        source_sha="0123456789abcdef",
        pack_slug="demo-pack",
    )
    result = magicpatch.convert_pack(config, execute_agent=agent)
    assert result.output == output.resolve()


def test_existing_output_and_symlinked_input_are_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source_pack(tmp_path / "source")
    output = tmp_path / "converted"
    output.mkdir()
    _provide_fake_codex(monkeypatch)
    with pytest.raises(magicpatch.MagicPatchError, match="output already exists"):
        magicpatch.convert_pack(_config(source, output))

    output.rmdir()
    (source / "linked").symlink_to(source / "nodes" / "demo.py")
    with pytest.raises(magicpatch.MagicPatchError, match="symbolic link"):
        magicpatch.convert_pack(_config(source, output))


def test_patch_identity_requires_git_root_or_explicit_commit(tmp_path: Path) -> None:
    source = _source_pack(tmp_path / "source")
    config = magicpatch.ConversionConfig(
        source=source,
        output=tmp_path / "converted",
        provider="codex",
        core_root=None,
    )

    with pytest.raises(magicpatch.MagicPatchError, match="pass --source-sha"):
        magicpatch._pack_identity(config, source)


def test_patch_identity_is_derived_from_a_git_pack_root(tmp_path: Path) -> None:
    source = _source_pack(tmp_path)
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    subprocess.run(["git", "add", "."], cwd=source, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Magic Patch Test",
            "-c",
            "user.email=magic-patch@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=source,
        check=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    config = magicpatch.ConversionConfig(
        source=source,
        output=tmp_path / "converted",
        provider="codex",
        core_root=None,
    )

    identity = magicpatch._pack_identity(config, source)

    assert identity.slug == "demo-pack"
    assert identity.key == f"x{commit[:7]}"
    assert identity.commit == commit


def test_artifact_publication_rolls_back_as_a_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = tmp_path / "stage"
    source_pack = stage / "pack"
    source_patch = stage / "patch"
    source_pack.mkdir(parents=True)
    source_patch.mkdir()
    (source_pack / "pack.txt").write_text("pack\n")
    (source_patch / "patch.json").write_text("{}\n")
    source_zip = stage / "pack.zip"
    source_report = stage / "report.json"
    source_zip.write_bytes(b"zip")
    source_report.write_text("{}\n")
    output = tmp_path / "published-pack"
    patch_output = tmp_path / "published-patch"
    pack_zip = tmp_path / "published.zip"
    report = tmp_path / "published.json"
    real_replace = magicpatch.os.replace
    calls = 0

    def fail_once(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("publication interrupted")
        real_replace(source, destination)

    monkeypatch.setattr(magicpatch.os, "replace", fail_once)

    with pytest.raises(OSError, match="publication interrupted"):
        magicpatch._publish_artifacts(
            [
                (source_pack, output, True),
                (source_patch, patch_output, True),
                (source_zip, pack_zip, False),
                (source_report, report, False),
            ]
        )

    assert source_pack.is_dir()
    assert source_patch.is_dir()
    assert source_zip.is_file()
    assert source_report.is_file()
    assert not output.exists()
    assert not patch_output.exists()
    assert not pack_zip.exists()
    assert not report.exists()


def test_provider_invocations_use_noninteractive_restricted_modes(
    tmp_path: Path,
) -> None:
    schema = tmp_path / "schema.json"
    result = tmp_path / "result.json"
    codex = magicpatch._invocation(
        "codex", tmp_path, "prompt", result, schema, model="model-a", max_turns=7
    )
    assert codex.command[:2] == ("codex", "exec")
    assert "--sandbox" in codex.command
    assert "workspace-write" in codex.command
    assert "--ephemeral" in codex.command
    assert "--ignore-user-config" in codex.command
    assert "--output-schema" in codex.command

    claude = magicpatch._invocation(
        "claude", tmp_path, "prompt", result, schema, model="model-b", max_turns=7
    )
    assert claude.command[0] == "claude"
    assert "--print" in claude.command
    assert "--restricted" in claude.command
    assert "--safe-mode" in claude.command
    assert "--no-session-persistence" in claude.command
    assert "--max-turns" in claude.command


def test_claude_structured_result_is_parsed_from_json_envelope(tmp_path: Path) -> None:
    invocation = magicpatch.AgentInvocation(
        "claude", ("claude",), "prompt", tmp_path, tmp_path / "unused.json"
    )
    completed = subprocess.CompletedProcess(
        invocation.command,
        0,
        json.dumps({"structured_output": _agent_value()}),
        "",
    )
    result = magicpatch._parse_agent_output(invocation, completed)
    assert result.status == "complete"
    assert result.backend_supported == 1


@pytest.mark.parametrize("direct_push", [True, False])
def test_create_pull_request_uses_disposable_clone_and_formatted_body(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    direct_push: bool,
) -> None:
    source = _source_pack(tmp_path / "source")
    output = tmp_path / "converted"
    _provide_fake_codex(monkeypatch)

    def agent(
        invocation: magicpatch.AgentInvocation,
    ) -> subprocess.CompletedProcess[str]:
        _finish_conversion(invocation.cwd)
        return _codex_result(invocation, _agent_value())

    result = magicpatch.convert_pack(_config(source, output), execute_agent=agent)
    observed: dict[str, str] = {}

    def runner(
        command: list[str] | tuple[str, ...], *, cwd: Path
    ) -> subprocess.CompletedProcess[str]:
        args = list(command)
        stdout = ""
        returncode = 0
        if args[:3] == ["git", "rev-parse", "--show-toplevel"]:
            returncode = 1
        elif args[:4] == ["gh", "repo", "view", "Comfy-Org/demo-pack"]:
            stdout = "main\n"
        elif args[:3] == ["gh", "repo", "clone"]:
            clone = Path(args[4])
            clone.mkdir(parents=True)
        elif args[:3] == ["git", "status", "--porcelain"]:
            stdout = "A  v2/secure-nodes.json\n"
        elif args[:4] == ["git", "config", "--get", "user.name"]:
            stdout = "Contributor\n"
        elif args[:4] == ["git", "config", "--get", "user.email"]:
            stdout = "contributor@example.com\n"
        elif args[:3] == ["git", "push", "origin"] and not direct_push:
            returncode = 1
        elif args[:3] == ["gh", "api", "user"]:
            stdout = "contributor\n"
        elif args[:3] == ["gh", "pr", "create"]:
            body = Path(args[args.index("--body-file") + 1])
            observed["body"] = body.read_text()
            observed["copied"] = str((cwd / "v2" / "secure-nodes.json").is_file())
            observed["head"] = args[args.index("--head") + 1]
            stdout = "https://github.com/Comfy-Org/demo-pack/pull/42\n"
        return subprocess.CompletedProcess(args, returncode, stdout, "")

    config = _config(
        source,
        output,
        create_pr=True,
        pr_repo="Comfy-Org/demo-pack",
        pr_branch="magic-patch/test",
    )
    url = magicpatch.create_pull_request(config, result, run_command=runner)

    assert url == "https://github.com/Comfy-Org/demo-pack/pull/42"
    assert observed["copied"] == "True"
    assert "Backend nodes supported: 1" in observed["body"]
    assert "JSON/diff patch pair recreated" in observed["body"]
    expected_head = (
        "magic-patch/test" if direct_push else "contributor:magic-patch/test"
    )
    assert observed["head"] == expected_head
    assert not (source / "v2").exists()
