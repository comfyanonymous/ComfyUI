#!/bin/bash
# comfy-node-install - Install ComfyUI custom nodes from GitHub repositories
# Usage: comfy-node-install <repo-url> [repo-url2 ...]

set -e

COMFYUI_DIR="${COMFYUI_DIR:-/app/ComfyUI}"
CUSTOM_NODES_DIR="${COMFYUI_DIR}/custom_nodes"

# Ensure custom_nodes directory exists
mkdir -p "${CUSTOM_NODES_DIR}"

install_node() {
    local repo_url="$1"
    if [ -z "$repo_url" ]; then
        echo "Error: Repository URL is required"
        return 1
    fi

    # Extract repository name from URL
    local repo_name=$(basename "${repo_url}" .git)
    
    # Handle full GitHub URLs or just repo paths
    if [[ "$repo_url" != http* ]]; then
        repo_url="https://github.com/${repo_url}"
    fi

    local target_dir="${CUSTOM_NODES_DIR}/${repo_name}"

    echo "Installing custom node: ${repo_name} from ${repo_url}"

    # Remove existing installation if it exists
    if [ -d "${target_dir}" ]; then
        echo "  Removing existing installation..."
        rm -rf "${target_dir}"
    fi

    # Clone the repository
    if [ -n "${GIT_LFS_SKIP_SMUDGE}" ]; then
        GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 "${repo_url}" "${target_dir}"
    else
        git clone --depth 1 "${repo_url}" "${target_dir}"
    fi

    echo "  Successfully installed ${repo_name}"
}

# Install all provided repositories
for repo_url in "$@"; do
    install_node "${repo_url}"
done

