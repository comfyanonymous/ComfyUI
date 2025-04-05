from __future__ import annotations
import argparse
import logging
import os
import re
import sys
import tempfile
import zipfile
import importlib
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TypedDict, Optional
from importlib.metadata import version

import requests
from typing_extensions import NotRequired

from comfy.cli_args import DEFAULT_VERSION_STRING
import app.logger

# The path to the requirements.txt file
req_path = Path(__file__).parents[1] / "requirements.txt"


def frontend_install_warning_message():
    """The warning message to display when the frontend version is not up to date."""

    extra = ""
    if sys.flags.no_user_site:
        extra = "-s "
    return f"""
Please install the updated requirements.txt file by running:
{sys.executable} {extra}-m pip install -r {req_path}

This error is happening because the ComfyUI frontend is no longer shipped as part of the main repo but as a pip package instead.

If you are on the portable package you can run: update\\update_comfyui.bat to solve this problem
""".strip()


def check_frontend_version():
    """Check if the frontend version is up to date."""

    def parse_version(version: str) -> tuple[int, int, int]:
        return tuple(map(int, version.split(".")))

    try:
        frontend_version_str = version("comfyui-frontend-package")
        frontend_version = parse_version(frontend_version_str)
        with open(req_path, "r", encoding="utf-8") as f:
            required_frontend = parse_version(f.readline().split("=")[-1])
        if frontend_version < required_frontend:
            app.logger.log_startup_warning(
                f"""
________________________________________________________________________
WARNING WARNING WARNING WARNING WARNING

Installed frontend version {".".join(map(str, frontend_version))} is lower than the recommended version {".".join(map(str, required_frontend))}.

{frontend_install_warning_message()}
________________________________________________________________________
""".strip()
            )
        else:
            logging.info("ComfyUI frontend version: {}".format(frontend_version_str))
    except Exception as e:
        logging.error(f"Failed to check frontend version: {e}")


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
    CUSTOM_FRONTENDS_ROOT = str(Path(__file__).parents[1] / "web_custom_versions")

    @classmethod
    def default_frontend_path(cls) -> str:
        try:
            import comfyui_frontend_package

            return str(importlib.resources.files(comfyui_frontend_package) / "static")
        except ImportError:
            logging.error(
                f"""
********** ERROR ***********

comfyui-frontend-package is not installed.

{frontend_install_warning_message()}

********** ERROR ***********
""".strip()
            )
            sys.exit(-1)

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
    def init_frontend_unsafe(
        cls, version_string: str, provider: Optional[FrontEndProvider] = None
    ) -> str:
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
            check_frontend_version()
            return cls.default_frontend_path()

        repo_owner, repo_name, version = cls.parse_version_string(version_string)

        if version.startswith("v"):
            expected_path = str(
                Path(cls.CUSTOM_FRONTENDS_ROOT)
                / f"{repo_owner}_{repo_name}"
                / version.lstrip("v")
            )
            if os.path.exists(expected_path):
                logging.info(
                    f"Using existing copy of specific frontend version tag: {repo_owner}/{repo_name}@{version}"
                )
                return expected_path

        logging.info(
            f"Initializing frontend: {repo_owner}/{repo_name}@{version}, requesting version details from GitHub..."
        )

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
            check_frontend_version()
            return cls.default_frontend_path()
