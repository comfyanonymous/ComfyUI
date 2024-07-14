# Use `aria2` as downloader

Two environment variables are needed to use `aria2` as the downloader.

```bash
export COMFYUI_MANAGER_ARIA2_SERVER=http://127.0.0.1:6800
export COMFYUI_MANAGER_ARIA2_SECRET=__YOU_MUST_CHANGE_IT__
```

An example `docker-compose.yml`

```yaml
services:

  aria2:
    container_name: aria2
    image: p3terx/aria2-pro
    environment:
      - PUID=1000
      - PGID=1000
      - UMASK_SET=022
      - RPC_SECRET=__YOU_MUST_CHANGE_IT__
      - RPC_PORT=5080
      - DISK_CACHE=64M
      - IPV6_MODE=false
      - UPDATE_TRACKERS=false
      - CUSTOM_TRACKER_URL=
    volumes:
      - ./config:/config
      - ./downloads:/downloads
      - ~/ComfyUI/models:/models
      - ~/ComfyUI/custom_nodes:/custom_nodes
    ports:
      - 6800:6800
    restart: unless-stopped
    logging:
      driver: json-file
      options:
        max-size: 1m
```
