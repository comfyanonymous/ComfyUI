# Docker buildfile for the ComfyUI image, with support for hardware
# acceleration, file ownership synchronization, custom nodes, and custom node
# managers.

# While Python 3.13 is well supported by ComfyUI, some older custom node packs
# may not work correctly with this version, which is why we're staying on Python
# 3.12 for now.
#
# Users are free to try different base Python image tags (e.g., 3.13, alpine,
# *-slim), but for maintainability, only one base version is officially
# supported at a time.
FROM python:3.12.12-trixie

# Install cmake, which is an indirect installation dependencies
RUN apt-get update && apt-get install -y --no-install-recommends cmake

# Create a regular user whose UID and GID will match the host user's at runtime.
# Also create a home directory for this user (-m), as some common Python tools
# (such as uv) interact with the user’s home directory.
RUN useradd -m comfyui

# Install ComfyUI under /comfyui and set folder ownership to the comfyui user.
# With the legacy Docker builder (DOCKER_BUILDKIT=0), WORKDIR always creates missing
# directories as root (even if a different USER is active). To ensure the comfyui user
# can write inside, ownership must be fixed manually.
WORKDIR /comfyui
RUN chown comfyui:comfyui .

# Install ComfyUI as ComfyUI
USER comfyui

# Set up a Python virtual environment and configure it as the default Python.
#
# Reasons for using a virtual environment:
# - Some custom nodes use third-party tools like uv, which do not support
#   user-level installations.
# - Custom node managers may install or update dependencies as the regular user,
#   so a global installation is not an option.
# This leaves virtual environments as the only viable choice.
RUN python -m venv .venv
ENV PATH="/comfyui/.venv/bin:$PATH"

# Install ComfyUI's Python dependencies. Although dependency keeping is also
# performed at startup, building ComfyUI's base dependencies into the image
# significantly speeds up each containers' first run.
#
# Since this step takes a long time to complete, it's performed early to take
# advantage of Docker's build cache, thereby accelerating subsequent builds.
COPY requirements.txt manager_requirements.txt ./
RUN pip install --no-cache-dir --disable-pip-version-check \
    -r requirements.txt

# Install ComfyUI
COPY . .

# Purely declarative: inform Docker and image users that this image is designed
# to listen on port 8188 for the web GUI.
EXPOSE 8188

# Declare persistent volumes. We assign one volume per data directory to match
# ComfyUI’s natural file layout and to let users selectively choose which
# directories they want to mount.
VOLUME /comfyui/.venv
VOLUME /comfyui/custom_nodes
VOLUME /comfyui/input
VOLUME /comfyui/models
VOLUME /comfyui/output
VOLUME /comfyui/temp
VOLUME /comfyui/user
VOLUME /home/comfyui

# Switch back to root to run the entrypoint and to install additional system
# dependencies
USER root

# Configure entrypoint
RUN chmod +x entrypoint.sh
ENTRYPOINT [ "./entrypoint.sh" ]
CMD [ "python", "./main.py" ]

# Install additional system dependencies
ARG APT_EXTRA_PACKAGES
RUN apt-get install -y --no-install-recommends $APT_EXTRA_PACKAGES \
	&& apt-get clean                                               \
	&& rm -rf /var/lib/apt/lists/*
