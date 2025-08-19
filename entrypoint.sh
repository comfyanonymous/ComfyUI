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


# Install packages listed in ./requirements.txt, requirement files under
# ./custom_nodes, and any specified in PIP_EXTRA_PACKAGES. Also store a hash of
# all dependencies to detect when new or updated packages need to be installed.
packages_hash_file="/home/comfyui/pkghash"

packages_comfyui=$(cat requirements.txt)
packages_custom=$(find custom_nodes -name requirements.txt -exec cat {} \;)
packages_extras=$(echo "$PIP_EXTRA_PACKAGES" | tr ' ' '\n')

current_hash=$(
    {
        echo "$packages_comfyui"
        echo "$packages_custom"
        echo "$packages_extras"
    } | sort | sha256sum | awk '{print $1}'
)

if [ ! -f "$packages_hash_file" ] || [ "$current_hash" != "$(cat $packages_hash_file)" ]; then
    echo "[entrypoint] Installing new python dependencies, this might take a while..."
    reqs="-r requirements.txt"
    for req in custom_nodes/*/requirements.txt; do
        [ -f "$req" ] && reqs="$reqs -r $req"
    done

    su -c "pip install -q --disable-pip-version-check --no-cache-dir $reqs $PIP_EXTRA_PACKAGES" comfyui
    echo "$current_hash" > "$packages_hash_file"
else
    echo "[entrypoint] Requirements unchanged, skipping install"
fi


# Run command as comfyui
echo "[entrypoint] Running command"
exec su -c "$*" comfyui
