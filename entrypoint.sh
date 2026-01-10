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


echo "[entrypoint] Changing directory ownership..."
chown -R "$user:$user_group" \
    /data                    \
    /comfyui                 \
    /home/comfyui

# Add the user to the groups owning /dev/nvidia* devices to ensure CUDA access.
# Typically, these devices belong to a single "video" group, but to be safe, we
# add the user to each device's group individually.
echo "[entrypoint] Adding user to GPU device groups..."
for dev in /dev/nvidia*; do
    # Known issue: There is no universal standard for group IDs across Linux
    # systems, so this may add the user to unexpected groups. For example, the
    # 'video' group on some systems uses GID 27, which corresponds to 'sudo' in
    # the python:3.12 image. This should not cause serious problems.
    group=$(ls -ld "$dev" | awk '{print $4}')
    usermod -aG "$group" "$user"
done

# Install or update the Python dependencies defined by ComfyUI (or any installed
# custom node) and also install any user-defined dependencies specified in
# PIP_EXTRA_PACKAGES.
echo "[entrypoint] Updating Python dependencies..."
su -c "
   pip install                      \\
        --no-cache-dir              \\
        --disable-pip-version-check \\
        -r requirements.txt         \\
        $(find custom_nodes -mindepth 2 -maxdepth 2 -type f -name requirements.txt -printf "-r '%p' ") \\
        $PIP_EXTRA_PACKAGES
" comfyui \
    || echo "[entrypoint] Failed to install dependencies, starting anyway" >&2

# Run command as comfyui
echo "[entrypoint] Running command"
exec su -c "$*" comfyui
