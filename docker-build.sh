#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status.

# Define Python Version and Venv Path (must match Dockerfile ENV)
PYTHON_VERSION="3.12"
VENV_PATH="/app/venv"
VENV_PYTHON="${VENV_PATH}/bin/python" # Used for constructing PyTorch install args

# Function to display version information
display_version_info() {
    echo "==========================================================="
    echo " PyTorch Version Selection:"
    echo "-----------------------------------------------------------"
    echo " Stable Version:"
    echo "  - Thoroughly tested, recommended for general use."
    echo "  - Pros: Reliability, fewer bugs."
    echo "  - Cons: May lack the absolute latest features."
    echo "-----------------------------------------------------------"
    echo " Latest Version (Nightly/Pre-release):"
    echo "  - Includes newest features and optimizations."
    echo "  - Pros: Cutting-edge capabilities."
    echo "  - Cons: Potentially less stable, may have bugs."
    echo "==========================================================="
}

# Function to ask user for GPU type
ask_gpu_type() {
    echo "Select GPU Type:"
    select gpu_choice in "NVIDIA" "AMD" "Cancel"; do
        case $gpu_choice in
            NVIDIA)
                gpu="NVIDIA"
                echo "Selected: NVIDIA"
                break
                ;;
            AMD)
                gpu="AMD"
                echo "Selected: AMD"
                break
                ;;
            Cancel)
                echo "Build cancelled."
                exit 0
                ;;
            *)
                echo "Invalid option $REPLY. Please choose 1, 2, or 3."
                ;;
        esac
    done
}

# Function to ask user for version preference
ask_version() {
    echo "Select PyTorch Version:"
    select version_choice in "Stable" "Latest" "Cancel"; do
        case $version_choice in
            Stable)
                version="Stable"
                echo "Selected: Stable"
                break
                ;;
            Latest)
                version="Latest"
                echo "Selected: Latest"
                break
                ;;
            Cancel)
                echo "Build cancelled."
                exit 0
                ;;
            *)
                echo "Invalid option $REPLY. Please choose 1, 2, or 3."
                ;;
        esac
    done
}

# --- Main Script Logic ---
display_version_info
ask_gpu_type
ask_version

# --- Determine Build Arguments based on Input ---
echo "Configuring build arguments..."

# --- Initialize new ARGs ---
TORCH_PRE_FLAG=""
TORCH_INDEX_URL=""
TORCH_EXTRA_INDEX_URL="" # Initialize as empty

if [[ "$gpu" == "NVIDIA" ]]; then
    if [[ "$version" == "Stable" ]]; then
        BASE_IMAGE_TAG="nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04"
        TORCH_EXTRA_INDEX_URL="--extra-index-url https://download.pytorch.org/whl/cu126"
    else # Latest
        BASE_IMAGE_TAG="nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04"
        TORCH_PRE_FLAG="--pre"
        TORCH_INDEX_URL="--index-url https://download.pytorch.org/whl/nightly/cu128"
    fi

elif [[ "$gpu" == "AMD" ]]; then
     if [[ "$version" == "Stable" ]]; then
        BASE_IMAGE_TAG="rocm/dev-ubuntu-24.04:6.2.4-complete"
        TORCH_INDEX_URL="--index-url https://download.pytorch.org/whl/rocm6.2"
    else # Latest
        BASE_IMAGE_TAG="rocm/dev-ubuntu-24.04:6.3.4-complete"
        TORCH_PRE_FLAG="--pre"
        TORCH_INDEX_URL="--index-url https://download.pytorch.org/whl/nightly/rocm6.3"
    fi
else
    echo "Error: Invalid GPU type configured after selection."
    exit 1
fi

# --- Construct and Run the Docker Build Command ---
IMAGE_NAME="comfyui-red-image:${gpu,,}-${version,,}"

echo "-----------------------------------------------------------"
echo "Starting Docker build..."
echo "  Image Tag: ${IMAGE_NAME}"
echo "  Base Image: ${BASE_IMAGE_TAG}"
echo "  PyTorch Pre Flag: '${TORCH_PRE_FLAG}'"
echo "  PyTorch Index URL: '${TORCH_INDEX_URL}'"
echo "  PyTorch Extra Index URL: '${TORCH_EXTRA_INDEX_URL}'"
echo "-----------------------------------------------------------"

# Build the image using the SEPARATE Docker build arguments
# REMOVED the interrupting comments from within this command block
docker build \
    --no-cache \
    --build-arg BASE_IMAGE_TAG="${BASE_IMAGE_TAG}" \
    --build-arg TORCH_PRE_FLAG="${TORCH_PRE_FLAG}" \
    --build-arg TORCH_INDEX_URL="${TORCH_INDEX_URL}" \
    --build-arg TORCH_EXTRA_INDEX_URL="${TORCH_EXTRA_INDEX_URL}" \
    -t "${IMAGE_NAME}" \
    -f Dockerfile .

BUILD_STATUS=$?

# --- Report Build Status ---
echo "-----------------------------------------------------------"
if [ $BUILD_STATUS -eq 0 ]; then
    echo "Docker build successful!"
    echo "Image created: ${IMAGE_NAME}"
    echo ""
    echo "To run the container using Docker Compose (assuming docker-compose.yml is configured):"
    echo "  docker-compose up -d"
    echo ""
    echo "To stop the container:"
    echo "  docker-compose down"
else
    echo "Docker build failed with status: ${BUILD_STATUS}"
fi
echo "==========================================================="

exit $BUILD_STATUS