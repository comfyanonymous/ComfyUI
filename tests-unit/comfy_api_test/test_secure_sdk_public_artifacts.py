"""Opaque public model acquisition; all HTTP responses are inert fixtures."""
import asyncio
from dataclasses import replace
import hashlib
import io
import json
import os
import threading

import pytest

import folder_paths
from comfy_api.latest import _public_artifacts as artifacts
from comfy_api.latest import _sdk, sdk


BODY = b"opaque fixture bytes; never deserialize this file"


def declaration(**changes):
    values = dict(repo_id="owner/model", release_id=12, release_tag="v1.2",
                  asset_id=34, filename="model.torchscript", folder="nlf",
                  sha256=hashlib.sha256(BODY).hexdigest(), size=len(BODY))
    values.update(changes)
    return sdk.GitHubReleaseArtifact(**values)


class Response(io.BytesIO):
    status = 200

    def __init__(self, data, url, headers=None):
        super().__init__(data)
        self.url = url
        self.headers = headers or {}

    def geturl(self):
        return self.url


@pytest.fixture
def source(monkeypatch, tmp_path):
    root = tmp_path / "nlf"
    monkeypatch.setitem(folder_paths.folder_names_and_paths, "nlf", ([str(root)], {".torchscript"}))
    monkeypatch.setenv("COMFY_SECURE_PUBLIC_ARTIFACT_MAX", "1024")
    monkeypatch.setenv("COMFY_SECURE_PUBLIC_ARTIFACT_SECONDS", "30")
    monkeypatch.setenv("COMFY_SECURE_HF_WEIGHT_MAX", "1024")
    calls = []
    config = {"body": BODY, "metadata": None, "headers": {}}

    class Opener:
        def open(self, request, timeout):
            calls.append((request.full_url, dict(request.header_items()), timeout))
            artifact = config.get("artifact", declaration())
            if request.full_url.endswith(f"/releases/{artifact.release_id}"):
                metadata = config["metadata"] or {
                    "id": artifact.release_id, "tag_name": artifact.release_tag,
                    "assets": [{"id": artifact.asset_id, "name": artifact.filename,
                                "size": artifact.size, "browser_download_url": artifact.source_url}],
                }
                return Response(json.dumps(metadata).encode(), request.full_url)
            return Response(config["body"], request.full_url, config["headers"])

    monkeypatch.setattr(artifacts.urllib.request, "build_opener", lambda *handlers: Opener())
    return root, calls, config


@pytest.mark.parametrize("changes", [
    {"repo_id": "https://github.com/a/b"}, {"repo_id": "a/../b"},
    {"repo_id": "a/b?token=secret"}, {"release_tag": "../../other"},
    {"release_tag": "v1?token=secret"}, {"release_id": True}, {"asset_id": 0},
    {"filename": "../model.torchscript"}, {"filename": "model.pt"},
    {"filename": "C:\\model.torchscript"}, {"folder": "checkpoints"},
    {"sha256": None}, {"sha256": "0" * 63}, {"size": -1},
    {"size": True}, {"on_demand": 1},
])
def test_invalid_declaration_refuses_before_io(changes):
    with pytest.raises((TypeError, ValueError)):
        declaration(**changes)


def test_public_declaration_and_distinct_permission():
    artifact = declaration()
    assert artifact.on_demand is True
    assert artifact.catalogue_name.endswith("/" + artifact.sha256 + "/model.torchscript")
    plan = _sdk.ExecutionPlan("p", "1", "Model", required_artifacts=(artifact,))
    assert plan.required_artifacts == (artifact,)
    assert plan.permissions == ("models.download_artifacts",)
    assert not plan.required_weights
    hf = sdk.HuggingFaceWeight("owner/model", "weights.safetensors", "loras")
    legacy_plan = _sdk.ExecutionPlan("p", "2", "Weights", required_weights=(hf,))
    assert legacy_plan.permissions == ("models.download",)
    assert legacy_plan.required_artifacts == ()


def test_success_is_opaque_exact_and_reused_without_network(source, monkeypatch):
    root, calls, config = source
    def forbidden(*args, **kwargs):
        pytest.fail("host tensor/JIT parser must not see opaque artifacts")
    monkeypatch.setattr(_sdk._InProcessModels, "_verify_weight_file", forbidden)
    artifact = declaration()
    plan = _sdk.ExecutionPlan("public-artifact", "1", "Model", required_artifacts=(artifact,))
    models = _sdk.InProcessCtxProvider().build(plan).models
    logical = asyncio.run(models.download_public_artifact(artifact))
    assert logical == artifact.catalogue_name
    assert (root / logical).read_bytes() == BODY
    assert len(calls) == 2
    assert all("Authorization" not in headers and "Cookie" not in headers for _,headers,_ in calls)
    assert all(0 < timeout <= 10 for _,_,timeout in calls)
    calls.clear()
    assert artifacts.download_github_release_artifact(artifact) == logical
    assert calls == []


@pytest.mark.parametrize("change", ["tag", "release", "asset", "name", "size", "url"])
def test_metadata_identity_mismatch_never_publishes(source, change):
    root, calls, config = source
    artifact = declaration()
    asset = {"id": artifact.asset_id, "name": artifact.filename,
             "size": artifact.size, "browser_download_url": artifact.source_url}
    metadata = {"id": artifact.release_id, "tag_name": artifact.release_tag, "assets": [asset]}
    if change == "tag": metadata["tag_name"] = "other"
    elif change == "release": metadata["id"] += 1
    elif change == "asset": asset["id"] += 1
    elif change == "name": asset["name"] = "other.torchscript"
    elif change == "size": asset["size"] += 1
    else: asset["browser_download_url"] = "https://example.com/model.torchscript"
    config["metadata"] = metadata
    with pytest.raises(ValueError): artifacts.download_github_release_artifact(artifact)
    assert len(calls) == 1
    assert not root.exists()


@pytest.mark.parametrize("payload", [BODY[:-1], BODY + b"x", b"x" * len(BODY)])
def test_bad_payload_cleans_temporary_and_does_not_publish(source, payload):
    root, calls, config = source
    config["body"] = payload
    artifact = declaration()
    with pytest.raises(ValueError): artifacts.download_github_release_artifact(artifact)
    assert not (root / artifact.catalogue_name).exists()
    assert not list(root.rglob(".public-artifact-*"))


@pytest.mark.parametrize("headers", [{"Content-Length": "1"}, {"Content-Length": "garbage"}, {"Content-Encoding": "gzip"}])
def test_unexpected_delivery_headers_refuse(source, headers):
    root, calls, config = source
    config["headers"] = headers
    with pytest.raises(ValueError): artifacts.download_github_release_artifact(declaration())
    assert not any(path.is_file() for path in root.rglob("*"))


def test_byte_ceiling_refuses_before_metadata_or_paths(source, monkeypatch):
    root, calls, config = source
    monkeypatch.setenv("COMFY_SECURE_PUBLIC_ARTIFACT_MAX", str(len(BODY)-1))
    with pytest.raises(ValueError, match="byte limit"):
        artifacts.download_github_release_artifact(declaration())
    assert calls == [] and not root.exists()


@pytest.mark.parametrize("url", ["http://github.com/a", "https://127.0.0.1/a", "https://github.com.evil.test/a", "https://github.com@evil.test/a", "https://user:secret@github.com/a", "https://github.com:444/a", "file:///tmp/model"])
def test_redirect_refused_before_following(url):
    handler = artifacts._PublicGitHubRedirects()
    request = artifacts.urllib.request.Request("https://github.com/a/b")
    with pytest.raises(ValueError): handler.redirect_request(request, None, 302, "redirect", {}, url)


def test_existing_corrupt_file_is_refused_without_network_or_overwrite(source):
    root, calls, config = source
    artifact = declaration(); path = root / artifact.catalogue_name
    path.parent.mkdir(parents=True); path.write_bytes(b"x" * len(BODY))
    with pytest.raises(ValueError, match="sha256"):
        artifacts.download_github_release_artifact(artifact)
    assert calls == [] and path.read_bytes() == b"x" * len(BODY)


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink and FIFO semantics")
def test_linked_destination_and_nonregular_cache_refuse(source, tmp_path):
    root, calls, config = source
    artifact = declaration(); path = root / artifact.catalogue_name
    path.parent.mkdir(parents=True)
    external = tmp_path / "external"; external.write_bytes(BODY)
    path.symlink_to(external)
    with pytest.raises(ValueError, match="symlinks"):
        artifacts.download_github_release_artifact(artifact)
    path.unlink(); os.mkfifo(path)
    with pytest.raises(ValueError, match="regular"):
        artifacts.download_github_release_artifact(artifact)
    assert calls == []


def test_cancellation_removes_private_partial_and_releases_lock(source, monkeypatch):
    root, calls, config = source
    cancelled = threading.Event()
    original = Response.read1
    def read1(self, size):
        data = original(self, size)
        if not self.url.endswith("/releases/12"): cancelled.set()
        return data
    monkeypatch.setattr(Response, "read1", read1)
    with pytest.raises(InterruptedError):
        artifacts.download_github_release_artifact(declaration(), cancelled)
    assert not any(path.is_file() for path in root.rglob("*"))
    assert artifacts._INSTALL_LOCK.acquire(blocking=False)
    artifacts._INSTALL_LOCK.release()


def test_expired_absolute_deadline_cannot_be_renewed():
    deadline = artifacts._Acquisition(0, threading.Event())
    with pytest.raises(TimeoutError): artifacts._remaining(deadline)


def test_slow_delivery_does_not_renew_absolute_deadline(source, monkeypatch):
    root, calls, config = source
    now = [0.0]
    monkeypatch.setattr(artifacts.time, "monotonic", lambda: now[0])
    original = Response.read1
    def slow(self, size):
        if not self.url.endswith("/releases/12"): now[0] += 31
        return original(self, size)
    monkeypatch.setattr(Response, "read1", slow)
    with pytest.raises(TimeoutError): artifacts.download_github_release_artifact(declaration())
    assert not any(path.is_file() for path in root.rglob("*"))


def test_public_coroutine_cancellation_waits_for_partial_cleanup(source, monkeypatch):
    root, calls, config = source
    entered = threading.Event(); release = threading.Event()
    original = Response.read1
    def waiting(self, size):
        if not self.url.endswith("/releases/12"):
            entered.set()
            if not release.wait(3): raise TimeoutError("test fixture not released")
        return original(self, size)
    monkeypatch.setattr(Response, "read1", waiting)
    async def run():
        models = _sdk.InProcessCtxProvider().build(_sdk.ExecutionPlan("p", "1", "Model")).models
        task = asyncio.create_task(models.download_public_artifact(declaration()))
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            task.cancel(); release.set()
            with pytest.raises(asyncio.CancelledError): await asyncio.wait_for(task, 3)
        finally:
            release.set()
    asyncio.run(run())
    assert not any(path.is_file() for path in root.rglob("*"))


def test_different_releases_are_immutable_distinct_catalogue_entries(source):
    root, calls, config = source
    first = declaration(); second = replace(first, release_id=13, release_tag="v2", asset_id=35)
    for artifact in (first, second):
        config["artifact"] = artifact
        assert artifacts.download_github_release_artifact(artifact) == artifact.catalogue_name
    assert first.catalogue_name != second.catalogue_name
    assert (root/first.catalogue_name).read_bytes() == (root/second.catalogue_name).read_bytes() == BODY


def test_hf_weight_route_keeps_opaque_extension_refusal():
    with pytest.raises(ValueError, match="limited to"):
        sdk.HuggingFaceWeight("owner/model", "model.torchscript", "nlf")


def test_opaque_asset_stays_readable_but_host_tensor_loader_refuses(source, monkeypatch):
    root, calls, config = source
    artifact = declaration()
    logical = artifacts.download_github_release_artifact(artifact)
    def forbidden(*args, **kwargs): pytest.fail("host tensor parser was called")
    monkeypatch.setattr("comfy.utils.load_torch_file", forbidden)
    async def run():
        refs = _sdk.InProcessRefResolver()
        ctx = _sdk.InProcessCtxProvider().build(_sdk.ExecutionPlan("p", "1", "Model"))
        with _sdk.bind_runtime(refs, ctx, _sdk.InProcessOps()):
            asset = await ctx.assets.resolve("nlf", logical)
            assert await ctx.assets.size(asset) == len(BODY)
            assert await ctx.assets.digest(asset) == artifact.sha256
            assert await ctx.assets.read_range(asset, offset=0, length=len(BODY)) == BODY
            with pytest.raises(ValueError, match="isolated guest"):
                await ctx.assets.load_state_dict(asset)
    asyncio.run(run())
