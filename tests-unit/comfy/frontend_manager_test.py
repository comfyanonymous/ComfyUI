import pytest
import os
import argparse
from unittest.mock import patch

from app.frontend_management import (
    FrontEndProvider,
    Release,
    FrontendManager,
)


@pytest.fixture
def mock_releases():
    return [
        Release(
            id=1,
            tag_name="1.0.0",
            name="Release 1.0.0",
            prerelease=False,
            created_at="2022-01-01T00:00:00Z",
            published_at="2022-01-01T00:00:00Z",
            body="Release notes for 1.0.0",
            assets=[{"name": "dist.zip", "url": "https://example.com/dist.zip"}],
        ),
        Release(
            id=2,
            tag_name="2.0.0",
            name="Release 2.0.0",
            prerelease=False,
            created_at="2022-02-01T00:00:00Z",
            published_at="2022-02-01T00:00:00Z",
            body="Release notes for 2.0.0",
            assets=[{"name": "dist.zip", "url": "https://example.com/dist.zip"}],
        ),
    ]


@pytest.fixture
def mock_provider(mock_releases):
    provider = FrontEndProvider(
        name="test",
        owner="test-owner",
        repo="test-repo",
    )
    provider.all_releases = mock_releases
    provider.latest_release = mock_releases[1]
    FrontendManager.PROVIDERS = [provider]
    return provider


def test_get_release(mock_provider, mock_releases):
    version = "1.0.0"
    release = mock_provider.get_release(version)
    assert release == mock_releases[0]


def test_get_release_latest(mock_provider, mock_releases):
    version = "latest"
    release = mock_provider.get_release(version)
    assert release == mock_releases[1]


def test_get_release_invalid_version(mock_provider):
    version = "invalid"
    with pytest.raises(ValueError):
        mock_provider.get_release(version)


def test_init_frontend_default():
    version_string = FrontendManager.DEFAULT_VERSION_STRING
    frontend_path = FrontendManager.init_frontend(version_string)
    assert frontend_path == FrontendManager.DEFAULT_FRONTEND_PATH


def test_init_frontend_provider_version(mock_provider, mock_releases):
    version_string = f"{mock_provider.name}@1.0.0"
    with patch("app.frontend_management.download_release_asset_zip") as mock_download:
        with patch("os.makedirs") as mock_makedirs:
            frontend_path = FrontendManager.init_frontend(version_string)
            assert frontend_path == os.path.join(
                FrontendManager.CUSTOM_FRONTENDS_ROOT, mock_provider.name, "1.0.0"
            )
            mock_makedirs.assert_called_once_with(frontend_path, exist_ok=True)
            mock_download.assert_called_once_with(
                mock_releases[0], destination_path=frontend_path
            )


def test_init_frontend_provider_latest(mock_provider, mock_releases):
    version_string = f"{mock_provider.name}@latest"
    with patch("app.frontend_management.download_release_asset_zip") as mock_download:
        with patch("os.makedirs") as mock_makedirs:
            frontend_path = FrontendManager.init_frontend(version_string)
            assert frontend_path == os.path.join(
                FrontendManager.CUSTOM_FRONTENDS_ROOT, mock_provider.name, "2.0.0"
            )
            mock_makedirs.assert_called_once_with(frontend_path, exist_ok=True)
            mock_download.assert_called_once_with(
                mock_releases[1], destination_path=frontend_path
            )

def test_init_frontend_invalid_version():
    version_string = "test@1.100.99"
    with pytest.raises(ValueError):
        FrontendManager.init_frontend(version_string)


def test_init_frontend_invalid_provider():
    version_string = "invalid@latest"
    with pytest.raises(argparse.ArgumentTypeError):
        FrontendManager.init_frontend(version_string)


def test_parse_version_string():
    version_string = "test@1.0.0"
    provider, version = FrontendManager.parse_version_string(version_string)
    assert provider == "test"
    assert version == "1.0.0"


def test_parse_version_string_invalid():
    version_string = "invalid"
    with pytest.raises(argparse.ArgumentTypeError):
        FrontendManager.parse_version_string(version_string)
