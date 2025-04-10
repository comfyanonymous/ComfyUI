# --- Base Image Selection ---
    ARG BASE_IMAGE_TAG="nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04"
    ARG TORCH_PRE_FLAG=""
    ARG TORCH_INDEX_URL=""
    ARG TORCH_EXTRA_INDEX_URL=""
    
    # Use the ARG for the base image build stage
    FROM ${BASE_IMAGE_TAG} AS base
    # --- First stage ends here ---
    
    # --- Start the final stage from the base ---
    FROM base
    
    # Environment variables (Keep as is)
    ENV DEBIAN_FRONTEND=noninteractive
    ENV UV_INSTALL_DIR="/root/.local/bin"
    ENV UV_EXE="/root/.local/bin/uv"
    ENV PYTHON_VERSION="3.12"
    ENV VENV_PATH="/app/venv"
    ENV VENV_PYTHON="${VENV_PATH}/bin/python"
    ENV PATH="${VENV_PATH}/bin:${UV_INSTALL_DIR}:${PATH}"
    
    # --- Layer 1: Install OS Dependencies & Python --- (Keep as is)
    RUN apt-get update \
        && apt-get install -y --no-install-recommends \
            git curl "python${PYTHON_VERSION}" "python${PYTHON_VERSION}-dev" "python${PYTHON_VERSION}-venv" \
            wget ffmpeg libsm6 libxext6 libgl1 grep \
        && apt-get clean && rm -rf /var/lib/apt/lists/*
    
    # --- Layer 2: Install and Verify UV --- (Keep as is)
    RUN echo "Installing uv..." \
        && curl -LsSf https://astral.sh/uv/install.sh | sh \
        && echo "Verifying uv installation..." \
        && ${UV_EXE} --version
    
    WORKDIR /app
    
    # --- Layer 3: Create Virtual Environment & Ensure Core Tools ---
    RUN echo "Creating virtual environment with uv..." \
        && ${UV_EXE} venv ${VENV_PATH} --python "python${PYTHON_VERSION}" \
        && echo "Ensuring pip and wheel are installed/updated in venv..." \
    # Explicitly install/upgrade pip and wheel using uv right after venv creation
        && ${UV_EXE} pip install -p ${VENV_PYTHON} --upgrade pip wheel \
        && echo "Verifying pip exists in venv:" \
        && ${VENV_PYTHON} -m pip --version

# --- RE-DECLARE ARGs HERE ---
    ARG TORCH_PRE_FLAG
    ARG TORCH_INDEX_URL
    ARG TORCH_EXTRA_INDEX_URL

    # --- Layer 4: PyTorch Installation ---
    RUN echo "--- Executing PyTorch Install Step ---" \
        && echo "  ARG TORCH_PRE_FLAG='${TORCH_PRE_FLAG}'" \
        && echo "  ARG TORCH_INDEX_URL='${TORCH_INDEX_URL}'" \
        && echo "  ARG TORCH_EXTRA_INDEX_URL='${TORCH_EXTRA_INDEX_URL}'" \
        && echo "  Now running uv pip install..." \
        && ${UV_EXE} pip install \
            --upgrade \
            -p ${VENV_PYTHON} \
        # --- REMOVED INNER QUOTES from expansions ---
            ${TORCH_PRE_FLAG:+$TORCH_PRE_FLAG} \
            ${TORCH_INDEX_URL:+$TORCH_INDEX_URL} \
            ${TORCH_EXTRA_INDEX_URL:+$TORCH_EXTRA_INDEX_URL} \
        # --- END REMOVED INNER QUOTES ---
            torch torchvision torchaudio
    
# --- Layer 5: ComfyUI Setup (Clone & Requirements) ---
    RUN echo "Cloning ComfyUI..." \
        && git clone https://github.com/RedsAnalysis/ComfyUI.git /app/comfyui \
        && echo "Filtering requirements.txt to remove potential torch conflicts..." \
        && grep -vE '^torch(vision|audio)?(=|<|>)?' /app/comfyui/requirements.txt > /app/comfyui/requirements.filtered.txt \
        && REQS_FILE="/app/comfyui/requirements.filtered.txt" \
        && echo "Installing ComfyUI base requirements from ${REQS_FILE}..." \
    # Explicitly add torchsde here along with requirements file
        && ${UV_EXE} pip install \
            -p ${VENV_PYTHON} \
            pyyaml \
            torchsde \
            -r ${REQS_FILE}
    
    # --- Layer 6: ComfyUI-Manager Setup --- (Keep as is)
    RUN echo "Cloning ComfyUI-Manager..." \
        && git clone https://github.com/ltdrdata/ComfyUI-Manager.git /app/comfyui/custom_nodes/ComfyUI-Manager \
        && MANAGER_REQS="/app/comfyui/custom_nodes/ComfyUI-Manager/requirements.txt" \
        && if [ -f "${MANAGER_REQS}" ]; then \
             echo "Installing ComfyUI-Manager requirements..."; \
             ${UV_EXE} pip install -p ${VENV_PYTHON} -r ${MANAGER_REQS}; \
           else \
             echo "ComfyUI-Manager requirements.txt not found."; \
           fi
    
    # --- Final Setup --- (Keep as is)
    EXPOSE 8188
    HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
      CMD curl --fail http://localhost:8188/ || exit 1
    CMD ["python", "/app/comfyui/main.py", "--listen", "0.0.0.0", "--port", "8188"]