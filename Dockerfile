# syntax=docker/dockerfile:1.4

ARG BASE_IMAGE="python:3.11-slim-bookworm"

FROM ${BASE_IMAGE}

ARG PYTORCH_INSTALL_ARGS=""
ARG EXTRA_ARGS=""
ARG USERNAME="comfyui"
ARG USER_UID=1000
ARG USER_GID=${USER_UID}

RUN \
	--mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
	--mount=target=/var/cache/apt,type=cache,sharing=locked \
	set -eux; \
		apt-get update; \
		apt-get install -y --no-install-recommends \
			git \
			git-lfs \
 			rsync

RUN set -eux; \
	groupadd --gid ${USER_GID} ${USERNAME}; \
	useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME}

# run instructions as user
USER ${USER_UID}:${USER_GID}

WORKDIR /app

ENV PIP_CACHE_DIR="/cache/pip"
ENV VIRTUAL_ENV=/app/venv
ENV VIRTUAL_ENV_CUSTOM=/app/custom_venv
ENV TRANSFORMERS_CACHE="/app/.cache/transformers"

# create cache directory
RUN mkdir -p ${TRANSFORMERS_CACHE}

# create virtual environment to manage packages
RUN python -m venv ${VIRTUAL_ENV}

# run python from venv
ENV PATH="${VIRTUAL_ENV_CUSTOM}/bin:${VIRTUAL_ENV}/bin:${PATH}"

RUN --mount=type=cache,target=/cache/,uid=${USER_UID},gid=${USER_GID} \
	pip install torch torchvision torchaudio ${PYTORCH_INSTALL_ARGS}

# copy requirements files first so packages can be cached separately
COPY --chown=${USER_UID}:${USER_GID} requirements.txt .
RUN --mount=type=cache,target=/cache/,uid=${USER_UID},gid=${USER_GID} \
	pip install -r requirements.txt

COPY --chown=${USER_UID}:${USER_GID} . .

# default environment variables
ENV COMFYUI_ADDRESS=0.0.0.0
ENV COMFYUI_PORT=8188
ENV COMFYUI_EXTRA_BUILD_ARGS="${EXTRA_ARGS}"
ENV COMFYUI_EXTRA_ARGS=""
# default start command
CMD if [ -d "${VIRTUAL_ENV_CUSTOM}" ]; then rsync -aP "${VIRTUAL_ENV}/" "${VIRTUAL_ENV_CUSTOM}/"; fi;\
  python -u main.py --listen ${COMFYUI_ADDRESS} --port ${COMFYUI_PORT} ${COMFYUI_EXTRA_BUILD_ARGS} ${COMFYUI_EXTRA_ARGS}
