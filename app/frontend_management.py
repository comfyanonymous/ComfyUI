import argparse
import os
import re
import tempfile
import zipfile
import logging
from functools import cached_property
from typing import TypedDict
from dataclasses import dataclass
from typing_extensions import NotRequired
from pathlib import Path

import requests


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
    name: str
    owner: str
    repo: str
    # Semantic version number, e.g. 1.0.0
    stable_version: str

    @property
    def release_url(self) -> str:
        return f"https://api.github.com/repos/{self.owner}/{self.repo}/releases"

    @cached_property
    def all_releases(self) -> list[Release]:
        releases = []
        api_url = self.release_url
        while api_url:
            response = requests.get(api_url)
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
        response = requests.get(latest_release_url)
        response.raise_for_status()  # Raises an HTTPError if the response was an error
        return response.json()

    @cached_property
    def stable_release(self) -> Release:
        for release in self.all_releases:
            if release["tag_name"] in (self.stable_version, f"v{self.stable_version}"):
                return release
        raise ValueError(f"Stable version {self.stable_version} not found in releases")

    def get_release(self, version: str) -> Release:
        if version == "latest":
            return self.latest_release
        elif version == "stable":
            return self.stable_release
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
        response = requests.get(asset_url)
        response.raise_for_status()  # Ensure we got a successful response

        # Write the content to the temporary file
        tmp_file.write(response.content)

        # Go back to the beginning of the temporary file
        tmp_file.seek(0)

        # Extract the zip file content to the destination path
        with zipfile.ZipFile(tmp_file, "r") as zip_ref:
            zip_ref.extractall(destination_path)


class FrontendManager:
    DEFAULT_FRONTEND_PATH = str(Path(__file__).parent / "web")
    CUSTOM_FRONTENDS_ROOT = str(Path(__file__).parent / "web_custom_versions")

    PROVIDERS = [
        FrontEndProvider(
            name="main",
            owner="Comfy-Org",
            repo="ComfyUI_frontend",
            stable_version="1.0.0",
        ),
        FrontEndProvider(
            name="legacy",
            owner="Comfy-Org",
            repo="ComfyUI_frontend_legacy",
            stable_version="1.0.0",
        ),
    ]

    @classmethod
    def parse_version_string(cls, value: str) -> tuple[str, str]:
        """
        Args:
            value (str): The version string to parse.

        Returns:
            tuple[str, str]: A tuple containing provider name and version.

        Raises:
            argparse.ArgumentTypeError: If the version string is invalid.
        """
        VERSION_PATTERN = (
            r"^("
            + "|".join([provider.name for provider in cls.PROVIDERS])
            + r")@(\d+\.\d+\.\d+|latest|stable)$"
        )
        match_result = re.match(VERSION_PATTERN, value)
        if match_result is None:
            raise argparse.ArgumentTypeError(f"Invalid version string: {value}")

        return match_result.group(1), match_result.group(2)

    @classmethod
    def add_argument(cls, parser: argparse.ArgumentParser):
        help_string = f"""
        The version string should be in the format of:
        [provider]@[version]
        where provider is one of: {", ".join([provider.name for provider in cls.PROVIDERS])}
        and version is one of: a valid version number, latest, stable
        """

        parser.add_argument(
            "--front-end-version",
            type=str,
            default="main@stable",
            help=help_string,
        )

    @classmethod
    def init_frontend(cls, version_string: str) -> str:
        """
        Initializes the frontend for the specified version.

        Args:
            version_string (str): The version string.

        Returns:
            str: The path to the initialized frontend.

        Raises:
            ValueError: If the provider name is not found in the list of providers.
        """
        if version_string == "main@stable":
            return cls.DEFAULT_FRONTEND_PATH

        provider_name, version = cls.parse_version_string(version_string)
        provider = next(
            provider for provider in cls.PROVIDERS if provider.name == provider_name
        )
        release = provider.get_release(version)

        semantic_version = release["tag_name"].lstrip("v")
        web_root = str(Path(cls.CUSTOM_FRONTENDS_ROOT) / provider.name / semantic_version)
        if not os.path.exists(web_root):
            os.makedirs(web_root, exist_ok=True)
            logging.info(f"Downloading {provider.name} frontend version {semantic_version}")
            download_release_asset_zip(release, destination_path=web_root)
        return web_root
