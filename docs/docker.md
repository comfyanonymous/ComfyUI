# Docker Compose

This repository includes a `docker-compose.yml` file to simplify running ComfyUI with Docker.

## Docker Volumes vs. Local Directories

By default, the `docker-compose.yml` file uses a Docker-managed volume named `workspace_data`. This volume stores all of ComfyUI's data, including models, inputs, and outputs. This is the most straightforward way to get started, but it can be less convenient if you want to manage these files directly from your host machine.

For more direct control, you can configure Docker Compose to use local directories (bind mounts) instead. This maps folders on your host machine directly into the container.

To switch to using local directories, edit `docker-compose.yml`:

1.  In both the `backend` and `frontend` services, replace `- workspace_data:/workspace` with the specific local directories you want to mount. For example:

    ```yaml
    services:
      backend:
        volumes:
          # - workspace_data:/workspace # Comment out or remove this line
          - ./models:/workspace/models
          - ./custom_nodes:/workspace/custom_nodes
          - ./output:/workspace/output
          - ./input:/workspace/input
      ...
      frontend:
        volumes:
          # - workspace_data:/workspace # Comment out or remove this line
          - ./models:/workspace/models
          - ./custom_nodes:/workspace/custom_nodes
          - ./output:/workspace/output
          - ./input:/workspace/input
    ```

2.  At the bottom of the file, remove or comment out the `workspace_data: {}` definition under `volumes`.

    ```yaml
    volumes:
      # workspace_data: {} # Comment out or remove this line
    ```

    Before running `docker compose up`, make sure the local directories (`./models`, `./custom_nodes`, etc.) exist in the same directory as your `docker-compose.yml` file.

The example `docker-compose` file contains other configuration settings. You can also use bind-mount volumes in the `volumes` key of the whole compose file. Read it carefully.

## Running with Docker Compose

### Linux

Before you begin, you must have the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for Docker installed to enable GPU acceleration.

### Starting the Stack

```shell
docker compose up
```

# Containers

On NVIDIA:

```agsl
docker pull ghcr.io/hiddenswitch/comfyui:latest
```

To run:

**Windows, `cmd`**
```shell
docker run -p "8188:8188" -v %cd%:/workspace -w "/workspace" --rm -it --gpus=all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 ghcr.io/hiddenswitch/comfyui:latest
```

**Linux**:
```shell
docker run -p "8188:8188" -v $(pwd):/workspace -w "/workspace" --rm -it --gpus=all --ipc=host ghcr.io/hiddenswitch/comfyui:latest
```
