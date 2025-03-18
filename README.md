## Prerequisites For Docker
 1. NVIDIA drivers installed on your host system.
 2. Docker installed on your host system.
 3. NVIDIA Container Toolkit (nvidia-docker2) OR WSL2 installed so that you can use GPU inside
 Docker.

### Run dockerfile
` docker run --gpus all \-it \-p 8188:8188 \--name comfyui-instance \
 comfyui-gpu`
### Run docker-compose
`docker compose up -d`