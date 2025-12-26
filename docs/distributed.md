# Distributed, Multi-Process and Multi-GPU Comfy

This package supports multi-processing across machines using RabbitMQ. This means you can launch multiple ComfyUI backend workers and queue prompts against them from multiple frontends.

## Getting Started

ComfyUI has two roles: `worker` and `frontend`. An unlimited number of workers can consume and execute workflows (prompts) in parallel; and an unlimited number of frontends can submit jobs. All of the frontends' API calls will operate transparently against your collection of workers, including progress notifications from the websocket.

To share work among multiple workers and frontends, ComfyUI uses RabbitMQ or any AMQP-compatible message queue like SQS or Kafka.

### Example with RabbitMQ and File Share

On a machine in your local network, install **Docker** and run RabbitMQ:

```shell
docker run -it --rm --name rabbitmq -p 5672:5672 rabbitmq:latest
```

Find the machine's main LAN IP address:

**Windows (PowerShell)**:

```pwsh
Get-NetIPConfiguration | Where-Object { $_.InterfaceAlias -like '*Ethernet*' -and $_.IPv4DefaultGateway -ne $null } | ForEach-Object { $_.IPv4Address.IPAddress }
```

**Linux**

```shell
ip -4 addr show $(ip route show default | awk '/default/ {print $5}') | grep -oP 'inet \K[\d.]+'
```

**macOS**

```shell
ifconfig $(route get default | grep interface | awk '{print $2}') | awk '/inet / {print $2; exit}'
```

On my machine, this prints `10.1.0.100`, which is a local LAN IP that other hosts on my network can reach.

On this machine, you can also set up a file share for models, outputs and inputs.

Once you have installed this Python package following the installation steps, you can start a worker using:

**Starting a Worker:**

```shell
# you must replace the IP address with the one you printed above
comfyui-worker --distributed-queue-connection-uri="amqp://guest:guest@10.1.0.100"
```

All the normal command line arguments are supported. This means you can use `--cwd` to point to a file share containing the `models/` directory:

```shell
comfyui-worker --cwd //10.1.0.100/shared/workspace --distributed-queue-connection-uri="amqp://guest:guest@10.1.0.100"
```

**Starting a Frontend:**

```shell
comfyui --listen --distributed-queue-connection-uri="amqp://guest:guest@10.1.0.100" --distributed-queue-frontend
```

However, the frontend will **not** be able to find the output images or models to show the client by default. You must specify a place where the frontend can find the **same** outputs and models that are available to the backends:

```shell
comfyui --cwd //10.1.0.100/shared/workspace --listen --distributed-queue-connection-uri="amqp://guest:guest@10.1.0.100" --distributed-queue-frontend
```

You can carefully mount network directories into `outputs/` and `inputs/` such that they are shared among workers and frontends; you can store the `models/` on each machine, or serve them over a file share too.

### Operating

The frontend expects to find the referenced output images in its `--output-directory` or in the default `outputs/` under `--cwd` (aka the "workspace").

This means that workers and frontends do **not** have to have the same argument to `--cwd`. The paths that are passed to the **frontend**, such as the `inputs/` and `outputs/` directories, must have the **same contents** as the paths passed as those directories to the workers.

Since reading models like large checkpoints over the network can be slow, you can use `--extra-model-paths-config` to specify additional model paths. Or, you can use `--cwd some/path`, where `some/path` is a local directory, and, and mount `some/path/outputs` to a network directory.

Known models listed in [**model_downloader.py**](../comfy/model_downloader.py) are downloaded using `huggingface_hub` with the default `cache_dir`. This means you can mount a read-write-many volume, like an SMB share, into the default cache directory. Read more about this [here](https://huggingface.co/docs/huggingface_hub/en/guides/download).
