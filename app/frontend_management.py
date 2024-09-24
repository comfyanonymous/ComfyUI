from __future__ import annotations
import argparse
import logging
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TypedDict, Optional

import requests
from typing_extensions import NotRequired
from comfy.cli_args import DEFAULT_VERSION_STRING


REQUEST_TIMEOUT = 10  # seconds


class Asset(TypedDict):
    url: str


class Release(TypedDict):
    id: int
    tag_name: str
    name: str
    prerelease: bool
    created_at: str
    published_at: str
    body: str
    assets: NotRequired[list[Asset]]


@dataclass
class FrontEndProvider:
    owner: str
    repo: str

    @property
    def folder_name(self) -> str:
        return f"{self.owner}_{self.repo}"

    @property
    def release_url(self) -> str:
        return f"https://api.github.com/repos/{self.owner}/{self.repo}/releases"

    @cached_property
    def all_releases(self) -> list[Release]:
        releases = []
        api_url = self.release_url
        while api_url:
            response = requests.get(api_url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()  # Raises an HTTPError if the response was an error
            releases.extend(response.json())
            # GitHub uses the Link header to provide pagination links. Check if it exists and update api_url accordingly.
            if "next" in response.links:
                api_url = response.links["next"]["url"]
            else:
                api_url = None
        return releases

    @cached_property
    def latest_release(self) -> Release:
        latest_release_url = f"{self.release_url}/latest"
        response = requests.get(latest_release_url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()  # Raises an HTTPError if the response was an error
        return response.json()

    def get_release(self, version: str) -> Release:
        if version == "latest":
            return self.latest_release
        else:
            for release in self.all_releases:
                if release["tag_name"] in [version, f"v{version}"]:
                    return release
            raise ValueError(f"Version {version} not found in releases")


def download_release_asset_zip(release: Release, destination_path: str) -> None:
    """Download dist.zip from github release."""
    asset_url = None
    for asset in release.get("assets", []):
        if asset["name"] == "dist.zip":
            asset_url = asset["url"]
            break

    if not asset_url:
        raise ValueError("dist.zip not found in the release assets")

    # Use a temporary file to download the zip content
    with tempfile.TemporaryFile() as tmp_file:
        headers = {"Accept": "application/octet-stream"}
        response = requests.get(
            asset_url, headers=headers, allow_redirects=True, timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()  # Ensure we got a successful response

        # Write the content to the temporary file
        tmp_file.write(response.content)

        # Go back to the beginning of the temporary file
        tmp_file.seek(0)

        # Extract the zip file content to the destination path
        with zipfile.ZipFile(tmp_file, "r") as zip_ref:
            zip_ref.extractall(destination_path)


class FrontendManager:
    DEFAULT_FRONTEND_PATH = str(Path(__file__).parents[1] / "web")
    CUSTOM_FRONTENDS_ROOT = str(Path(__file__).parents[1] / "web_custom_versions")

    @classmethod
    def parse_version_string(cls, value: str) -> tuple[str, str, str]:
        """
        Args:
            value (str): The version string to parse.

        Returns:
            tuple[str, str]: A tuple containing provider name and version.

        Raises:
            argparse.ArgumentTypeError: If the version string is invalid.
        """
        VERSION_PATTERN = r"^([a-zA-Z0-9][a-zA-Z0-9-]{0,38})/([a-zA-Z0-9_.-]+)@(v?\d+\.\d+\.\d+|latest)$"
        match_result = re.match(VERSION_PATTERN, value)
        if match_result is None:
            raise argparse.ArgumentTypeError(f"Invalid version string: {value}")

        return match_result.group(1), match_result.group(2), match_result.group(3)

    @classmethod
    def init_frontend_unsafe(cls, version_string: str, provider: Optional[FrontEndProvider] = None) -> str:
        """
        Initializes the frontend for the specified version.

        Args:
            version_string (str): The version string.
            provider (FrontEndProvider, optional): The provider to use. Defaults to None.

        Returns:
            str: The path to the initialized frontend.

        Raises:
            Exception: If there is an error during the initialization process.
            main error source might be request timeout or invalid URL.
        """
        if version_string == DEFAULT_VERSION_STRING:
            return cls.DEFAULT_FRONTEND_PATH

        repo_owner, repo_name, version = cls.parse_version_string(version_string)
        provider = provider or FrontEndProvider(repo_owner, repo_name)
        release = provider.get_release(version)

        semantic_version = release["tag_name"].lstrip("v")
        web_root = str(
            Path(cls.CUSTOM_FRONTENDS_ROOT) / provider.folder_name / semantic_version
        )
        if not os.path.exists(web_root):
            try:
                os.makedirs(web_root, exist_ok=True)
                logging.info(
                    "Downloading frontend(%s) version(%s) to (%s)",
                    provider.folder_name,
                    semantic_version,
                    web_root,
                )
                logging.debug(release)
                download_release_asset_zip(release, destination_path=web_root)
            finally:
                # Clean up the directory if it is empty, i.e. the download failed
                if not os.listdir(web_root):
                    os.rmdir(web_root)

        return web_root

    @classmethod
    def init_frontend(cls, version_string: str) -> str:
        """
        Initializes the frontend with the specified version string.

        Args:
            version_string (str): The version string to initialize the frontend with.

        Returns:
            str: The path of the initialized frontend.
        """
        try:
            return cls.init_frontend_unsafe(version_string)
        except Exception as e:
            logging.error("Failed to initialize frontend: %s", e)
            logging.info("Falling back to the default frontend.")
            return cls.DEFAULT_FRONTEND_PATH
