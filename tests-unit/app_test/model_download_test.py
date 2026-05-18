import os

import pytest

import folder_paths
from app.model_download import (
    ModelDownloadError,
    ModelDownloadRequest,
    is_allowed_model_download_url,
    normalize_model_relative_path,
    parse_model_download_request,
    resolve_model_download_destination,
)


def test_parse_model_download_request_allows_huggingface_model_url():
    request = parse_model_download_request({
        "name": "nested/model.safetensors",
        "url": "https://huggingface.co/org/repo/resolve/main/model.safetensors",
        "directory": "checkpoints",
    })

    assert request == ModelDownloadRequest(
        name="nested/model.safetensors",
        url="https://huggingface.co/org/repo/resolve/main/model.safetensors",
        directory="checkpoints",
    )


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8000/model.safetensors",
        "http://huggingface.co/org/repo/resolve/main/model.safetensors",
        "https://example.com/model.safetensors",
    ],
)
def test_download_url_allowlist_rejects_untrusted_or_plain_http_urls(url):
    assert is_allowed_model_download_url(url) is False


@pytest.mark.parametrize(
    "name",
    [
        "../model.safetensors",
        "nested/../../model.safetensors",
        "/absolute/model.safetensors",
        "model.safetensors\x00",
    ],
)
def test_normalize_model_relative_path_rejects_unsafe_paths(name):
    with pytest.raises(ModelDownloadError):
        normalize_model_relative_path(name)


def test_parse_model_download_request_rejects_unsupported_extensions():
    with pytest.raises(ModelDownloadError):
        parse_model_download_request({
            "name": "model.gguf",
            "url": "https://huggingface.co/org/repo/resolve/main/model.gguf",
            "directory": "checkpoints",
        })


def test_resolve_model_download_destination_uses_configured_model_folder(tmp_path, monkeypatch):
    model_root = tmp_path / "models" / "checkpoints"
    monkeypatch.setattr(folder_paths, "folder_names_and_paths", {
        "checkpoints": ([str(model_root)], {".safetensors"}),
    })

    destination = resolve_model_download_destination(ModelDownloadRequest(
        name="sub/model.safetensors",
        url="https://huggingface.co/org/repo/resolve/main/model.safetensors",
        directory="checkpoints",
    ))

    assert destination.directory == "checkpoints"
    assert destination.relative_path == "sub/model.safetensors"
    assert destination.full_path == os.path.join(str(model_root), "sub", "model.safetensors")
    assert destination.already_exists is False


def test_resolve_model_download_destination_reuses_existing_model(tmp_path, monkeypatch):
    model_root = tmp_path / "models" / "checkpoints"
    model_root.mkdir(parents=True)
    existing = model_root / "model.safetensors"
    existing.write_bytes(b"model")
    monkeypatch.setattr(folder_paths, "folder_names_and_paths", {
        "checkpoints": ([str(model_root)], {".safetensors"}),
    })

    destination = resolve_model_download_destination(ModelDownloadRequest(
        name="model.safetensors",
        url="https://huggingface.co/org/repo/resolve/main/model.safetensors",
        directory="checkpoints",
    ))

    assert destination.full_path == str(existing)
    assert destination.already_exists is True


@pytest.mark.parametrize("directory", ["configs", "custom_nodes", "unknown"])
def test_resolve_model_download_destination_rejects_blocked_or_unknown_directories(tmp_path, monkeypatch, directory):
    monkeypatch.setattr(folder_paths, "folder_names_and_paths", {
        "configs": ([str(tmp_path / "configs")], {".yaml"}),
        "custom_nodes": ([str(tmp_path / "custom_nodes")], set()),
    })

    with pytest.raises(ModelDownloadError):
        resolve_model_download_destination(ModelDownloadRequest(
            name="model.safetensors",
            url="https://huggingface.co/org/repo/resolve/main/model.safetensors",
            directory=directory,
        ))
