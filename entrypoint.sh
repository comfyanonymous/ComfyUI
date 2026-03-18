#!/bin/sh

# Entrypoint script for the ComfyUI Docker image.

set -e

user="comfyui"
user_group="$user"

# Allow users to specify a UID and GID matching their own, so files created
# inside the container retain the same numeric ownership when mounted on the
# host.
if [ -n "$UID" ] && [ -n "$GID" ]; then
    echo "[entrypoint] Setting user UID and GID..."
    usermod  -u "$UID" "$user" > /dev/null
    groupmod -g "$GID" "$user_group"
else
    echo "[entrypoint] Missing UID or GID environment variables; keeping default values."
fi

# Changing a user's UID and GID revokes that user's access to files owned by the
# original UID/GID. To preserve access to runtime data, the ownership of those
# directories must be updated recursively so that their numeric owner matches
# the user's new UID and GID.
echo "[entrypoint] Changing directory ownership..."
chown -R "$user:$user_group" \
    /comfyui                 \
    /home/comfyui

# To use CUDA and other NVIDIA features, regular users must belong to the group
# that owns the /dev/nvidia* device files -- typically the video group.
#
# Known issue: Because these device files are mounted from the host system,
# there's no guarantee that the device's group ID will match the intended group
# inside the container. For example, the video group might be mapped to GID 27
# on the host, which corresponds to the sudo group in the python:3.12 image.
# This shouldn't cause major problems, and given the lack of a universal
# standard for system GIDs, there isn't much we can realistically change to
# address this issue.
echo "[entrypoint] Adding user to GPU device groups..."
for dev in /dev/nvidia*; do
    [ -e "$dev" ] || continue
    group=$(ls -ld "$dev" | awk '{print $4}')
    usermod -aG "$group" "$user"
done

# Install or update the Python dependencies defined by ComfyUI (or any installed
# custom node) and also install any user-defined dependencies specified in
# PIP_EXTRA_PACKAGES.
echo "[entrypoint] Updating Python dependencies..."
su -c "
   pip install                                   \\
        --no-cache-dir                           \\
        --disable-pip-version-check              \\
        --extra-index-url '$PIP_EXTRA_INDEX_URL' \\
        -r requirements.txt                      \\
        $(find custom_nodes -mindepth 2 -maxdepth 2 -type f -name requirements.txt -printf "-r '%p' ") \\
        $PIP_EXTRA_PACKAGES
" comfyui \
    || echo "[entrypoint] Failed to install dependencies, starting anyway" >&2

# Run command as comfyui
echo "[entrypoint] Running command"
exec su -c "$*" comfyui
