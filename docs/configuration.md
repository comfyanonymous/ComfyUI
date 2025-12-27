# Configuration

This project supports configuration with command line arguments, the environment and a configuration file.

## Configuration File

First, run `comfyui --help` for all supported configuration and arguments.

Args that start with `--` can also be set in a config file (`config.yaml`, `config.ini`, `config.conf` or `config.json` or specified via `-c`). Config file syntax allows: `key=value`, `flag=true`, `stuff=[a,b,c]` (for details, see syntax [here](https://goo.gl/R74nmi)). In general, command-line values override environment variables which override config file values which override defaults.

## Extra Model Paths

Copy [docs/examples/configuration/extra_model_paths.yaml](examples/configuration/extra_model_paths.yaml) to your working directory, and modify the folder paths to match your folder structure.

You can pass additional extra model path configurations with one or more copies of `--extra-model-paths-config=some_configuration.yaml`.

### Command Line Arguments

```
usage: comfyui [-h] [-c CONFIG_FILE]
               [--write-out-config-file CONFIG_OUTPUT_PATH] [-w CWD]
               [--base-paths BASE_PATHS [BASE_PATHS ...]] [-H [IP]]
               [--port PORT] [--enable-cors-header [ORIGIN]]
               [--max-upload-size MAX_UPLOAD_SIZE]
               [--base-directory BASE_DIRECTORY]
               [--extra-model-paths-config PATH [PATH ...]]
               [--output-directory OUTPUT_DIRECTORY]
               [--temp-directory TEMP_DIRECTORY]
               [--input-directory INPUT_DIRECTORY] [--auto-launch]
               [--disable-auto-launch] [--cuda-device DEVICE_ID]
               [--default-device DEFAULT_DEVICE_ID]
               [--cuda-malloc | --disable-cuda-malloc]
               [--force-fp32 | --force-fp16 | --force-bf16]
               [--fp32-unet | --fp64-unet | --bf16-unet | --fp16-unet | --fp8_e4m3fn-unet | --fp8_e5m2-unet | --fp8_e8m0fnu-unet]
               [--fp16-vae | --fp32-vae | --bf16-vae] [--cpu-vae]
               [--fp8_e4m3fn-text-enc | --fp8_e5m2-text-enc | --fp16-text-enc | --fp32-text-enc | --bf16-text-enc]
               [--directml [DIRECTML_DEVICE]]
               [--oneapi-device-selector SELECTOR_STRING]
               [--disable-ipex-optimize] [--supports-fp8-compute]
               [--preview-method [none,auto,latent2rgb,taesd]]
               [--preview-size PREVIEW_SIZE]
               [--cache-classic | --cache-lru CACHE_LRU | --cache-none | --cache-ram [CACHE_RAM]]
               [--use-split-cross-attention | --use-quad-cross-attention | --use-pytorch-cross-attention | --use-sage-attention | --use-flash-attention]
               [--disable-xformers]
               [--force-upcast-attention | --dont-upcast-attention]
               [--enable-manager]
               [--disable-manager-ui | --enable-manager-legacy-ui]
               [--gpu-only | --highvram | --normalvram | --lowvram | --novram | --cpu]
               [--reserve-vram RESERVE_VRAM] [--async-offload [NUM_STREAMS]]
               [--disable-async-offload] [--force-non-blocking]
               [--default-hashing-function {md5,sha1,sha256,sha512}]
               [--disable-smart-memory] [--deterministic] [--fast [FAST ...]]
               [--disable-pinned-memory] [--mmap-torch-files] [--disable-mmap]
               [--dont-print-server] [--quick-test-for-ci]
               [--windows-standalone-build] [--disable-metadata]
               [--disable-all-custom-nodes]
               [--whitelist-custom-nodes WHITELIST_CUSTOM_NODES [WHITELIST_CUSTOM_NODES ...]]
               [--blacklist-custom-nodes BLACKLIST_CUSTOM_NODES [BLACKLIST_CUSTOM_NODES ...]]
               [--disable-api-nodes] [--enable-eval] [--multi-user]
               [--create-directories] [--log-stdout]
               [--plausible-analytics-base-url PLAUSIBLE_ANALYTICS_BASE_URL]
               [--plausible-analytics-domain PLAUSIBLE_ANALYTICS_DOMAIN]
               [--analytics-use-identity-provider]
               [--distributed-queue-connection-uri DISTRIBUTED_QUEUE_CONNECTION_URI]
               [--distributed-queue-worker] [--distributed-queue-frontend]
               [--distributed-queue-name DISTRIBUTED_QUEUE_NAME]
               [--external-address EXTERNAL_ADDRESS]
               [--logging-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
               [--disable-known-models] [--max-queue-size MAX_QUEUE_SIZE]
               [--otel-service-name OTEL_SERVICE_NAME]
               [--otel-service-version OTEL_SERVICE_VERSION]
               [--otel-exporter-otlp-endpoint OTEL_EXPORTER_OTLP_ENDPOINT]
               [--force-channels-last] [--force-hf-local-dir-mode]
               [--enable-video-to-image-fallback]
               [--front-end-version FRONT_END_VERSION]
               [--panic-when PANIC_WHEN [PANIC_WHEN ...]]
               [--front-end-root FRONT_END_ROOT]
               [--executor-factory EXECUTOR_FACTORY]
               [--openai-api-key OPENAI_API_KEY]
               [--ideogram-api-key IDEOGRAM_API_KEY]
               [--anthropic-api-key ANTHROPIC_API_KEY]
               [--user-directory USER_DIRECTORY]
               [--enable-compress-response-body]
               [--comfy-api-base COMFY_API_BASE]
               [--block-runtime-package-installation]
               [--database-url DATABASE_URL]
               [--workflows WORKFLOWS [WORKFLOWS ...]]
               [--disable-requests-caching]

options:
  -h, --help            show this help message and exit
  -c CONFIG_FILE, --config CONFIG_FILE
                        config file path
  --write-out-config-file CONFIG_OUTPUT_PATH
                        takes the current command line args and writes them
                        out to a config file at the given path, then exits
  -w CWD, --cwd CWD     Specify the working directory. If not set, this is the
                        current working directory. models/, input/, output/
                        and other directories will be located here by default.
                        [env var: COMFYUI_CWD]
  --base-paths BASE_PATHS [BASE_PATHS ...]
                        Additional base paths for custom nodes, models and
                        inputs. [env var: COMFYUI_BASE_PATHS]
  -H [IP], --listen [IP]
                        Specify the IP address to listen on (default:
                        127.0.0.1). You can give a list of ip addresses by
                        separating them with a comma like: 127.2.2.2,127.3.3.3
                        If --listen is provided without an argument, it
                        defaults to 0.0.0.0,:: (listens on all ipv4 and ipv6)
                        [env var: COMFYUI_LISTEN]
  --port PORT           Set the listen port. [env var: COMFYUI_PORT]
  --enable-cors-header [ORIGIN]
                        Enable CORS (Cross-Origin Resource Sharing) with
                        optional origin or allow all with default '*'. [env
                        var: COMFYUI_ENABLE_CORS_HEADER]
  --max-upload-size MAX_UPLOAD_SIZE
                        Set the maximum upload size in MB. [env var:
                        COMFYUI_MAX_UPLOAD_SIZE]
  --base-directory BASE_DIRECTORY
                        Set the ComfyUI base directory for models,
                        custom_nodes, input, output, temp, and user
                        directories. [env var: COMFYUI_BASE_DIRECTORY]
  --extra-model-paths-config PATH [PATH ...]
                        Load one or more extra_model_paths.yaml files. Can be
                        specified multiple times or as a comma-separated list.
                        [env var: COMFYUI_EXTRA_MODEL_PATHS_CONFIG]
  --output-directory OUTPUT_DIRECTORY
                        Set the ComfyUI output directory. Overrides --base-
                        directory. [env var: COMFYUI_OUTPUT_DIRECTORY]
  --temp-directory TEMP_DIRECTORY
                        Set the ComfyUI temp directory (default is in the
                        ComfyUI directory). Overrides --base-directory. [env
                        var: COMFYUI_TEMP_DIRECTORY]
  --input-directory INPUT_DIRECTORY
                        Set the ComfyUI input directory. Overrides --base-
                        directory. [env var: COMFYUI_INPUT_DIRECTORY]
  --auto-launch         Automatically launch ComfyUI in the default browser.
                        [env var: COMFYUI_AUTO_LAUNCH]
  --disable-auto-launch
                        Disable auto launching the browser. [env var:
                        COMFYUI_DISABLE_AUTO_LAUNCH]
  --cuda-device DEVICE_ID
                        Set the id of the cuda device this instance will use.
                        All other devices will not be visible. [env var:
                        COMFYUI_CUDA_DEVICE]
  --default-device DEFAULT_DEVICE_ID
                        Set the id of the default device, all other devices
                        will stay visible. [env var: COMFYUI_DEFAULT_DEVICE]
  --cuda-malloc         Enable cudaMallocAsync (enabled by default for torch
                        2.0 and up). [env var: COMFYUI_CUDA_MALLOC]
  --disable-cuda-malloc
                        Disable cudaMallocAsync. [env var:
                        COMFYUI_DISABLE_CUDA_MALLOC]
  --force-fp32          Force fp32 (If this makes your GPU work better please
                        report it). [env var: COMFYUI_FORCE_FP32]
  --force-fp16          Force fp16. [env var: COMFYUI_FORCE_FP16]
  --force-bf16          Force bf16. [env var: COMFYUI_FORCE_BF16]
  --fp32-unet           Run the diffusion model in fp32. [env var:
                        COMFYUI_FP32_UNET]
  --fp64-unet           Run the diffusion model in fp64. [env var:
                        COMFYUI_FP64_UNET]
  --bf16-unet           Run the diffusion model in bf16. [env var:
                        COMFYUI_BF16_UNET]
  --fp16-unet           Run the diffusion model in fp16 [env var:
                        COMFYUI_FP16_UNET]
  --fp8_e4m3fn-unet     Store unet weights in fp8_e4m3fn. [env var:
                        COMFYUI_FP8_E4M3FN_UNET]
  --fp8_e5m2-unet       Store unet weights in fp8_e5m2. [env var:
                        COMFYUI_FP8_E5M2_UNET]
  --fp8_e8m0fnu-unet    Store unet weights in fp8_e8m0fnu. [env var:
                        COMFYUI_FP8_E8M0FNU_UNET]
  --fp16-vae            Run the VAE in fp16, might cause black images. [env
                        var: COMFYUI_FP16_VAE]
  --fp32-vae            Run the VAE in full precision fp32. [env var:
                        COMFYUI_FP32_VAE]
  --bf16-vae            Run the VAE in bf16. [env var: COMFYUI_BF16_VAE]
  --cpu-vae             Run the VAE on the CPU. [env var: COMFYUI_CPU_VAE]
  --fp8_e4m3fn-text-enc
                        Store text encoder weights in fp8 (e4m3fn variant).
                        [env var: COMFYUI_FP8_E4M3FN_TEXT_ENC]
  --fp8_e5m2-text-enc   Store text encoder weights in fp8 (e5m2 variant). [env
                        var: COMFYUI_FP8_E5M2_TEXT_ENC]
  --fp16-text-enc       Store text encoder weights in fp16. [env var:
                        COMFYUI_FP16_TEXT_ENC]
  --fp32-text-enc       Store text encoder weights in fp32. [env var:
                        COMFYUI_FP32_TEXT_ENC]
  --bf16-text-enc       Store text encoder weights in bf16. [env var:
                        COMFYUI_BF16_TEXT_ENC]
  --directml [DIRECTML_DEVICE]
                        Use torch-directml. [env var: COMFYUI_DIRECTML]
  --oneapi-device-selector SELECTOR_STRING
                        Sets the oneAPI device(s) this instance will use. [env
                        var: COMFYUI_ONEAPI_DEVICE_SELECTOR]
  --disable-ipex-optimize
                        Disables ipex.optimize default when loading models
                        with Intel's Extension for Pytorch. [env var:
                        COMFYUI_DISABLE_IPEX_OPTIMIZE]
  --supports-fp8-compute
                        ComfyUI will act like if the device supports fp8
                        compute. [env var: COMFYUI_SUPPORTS_FP8_COMPUTE]
  --preview-method [none,auto,latent2rgb,taesd]
                        Default preview method for sampler nodes. [env var:
                        COMFYUI_PREVIEW_METHOD]
  --preview-size PREVIEW_SIZE
                        Sets the maximum preview size for sampler nodes. [env
                        var: COMFYUI_PREVIEW_SIZE]
  --cache-classic       WARNING: Unused. Use the old style (aggressive)
                        caching. [env var: COMFYUI_CACHE_CLASSIC]
  --cache-lru CACHE_LRU
                        Use LRU caching with a maximum of N node results
                        cached. May use more RAM/VRAM. [env var:
                        COMFYUI_CACHE_LRU]
  --cache-none          Reduced RAM/VRAM usage at the expense of executing
                        every node for each run. [env var: COMFYUI_CACHE_NONE]
  --cache-ram [CACHE_RAM]
                        Use RAM pressure caching with the specified headroom
                        threshold. If available RAM drops below the threhold
                        the cache remove large items to free RAM. Default 4GB
                        [env var: COMFYUI_CACHE_RAM]
  --use-split-cross-attention
                        Use the split cross attention optimization. Ignored
                        when xformers is used. [env var:
                        COMFYUI_USE_SPLIT_CROSS_ATTENTION]
  --use-quad-cross-attention
                        Use the sub-quadratic cross attention optimization .
                        Ignored when xformers is used. [env var:
                        COMFYUI_USE_QUAD_CROSS_ATTENTION]
  --use-pytorch-cross-attention
                        Use the new pytorch 2.0 cross attention function
                        (default). [env var:
                        COMFYUI_USE_PYTORCH_CROSS_ATTENTION]
  --use-sage-attention  Use sage attention. [env var:
                        COMFYUI_USE_SAGE_ATTENTION]
  --use-flash-attention
                        Use FlashAttention. [env var:
                        COMFYUI_USE_FLASH_ATTENTION]
  --disable-xformers    Disable xformers. [env var: COMFYUI_DISABLE_XFORMERS]
  --force-upcast-attention
                        Force enable attention upcasting, please report if it
                        fixes black images. [env var:
                        COMFYUI_FORCE_UPCAST_ATTENTION]
  --dont-upcast-attention
                        Disable all upcasting of attention. Should be
                        unnecessary except for debugging. [env var:
                        COMFYUI_DONT_UPCAST_ATTENTION]
  --enable-manager      Enable the ComfyUI-Manager feature. [env var:
                        COMFYUI_ENABLE_MANAGER]
  --disable-manager-ui  Disables only the ComfyUI-Manager UI and endpoints.
                        Scheduled installations and similar background tasks
                        will still operate. [env var:
                        COMFYUI_DISABLE_MANAGER_UI]
  --enable-manager-legacy-ui
                        Enables the legacy UI of ComfyUI-Manager [env var:
                        COMFYUI_ENABLE_MANAGER_LEGACY_UI]
  --gpu-only            Store and run everything (text encoders/CLIP models,
                        etc... on the GPU). [env var: COMFYUI_GPU_ONLY]
  --highvram            By default models will be unloaded to CPU memory after
                        being used. This option keeps them in GPU memory. [env
                        var: COMFYUI_HIGHVRAM]
  --normalvram          Used to force normal vram use if lowvram gets
                        automatically enabled. [env var: COMFYUI_NORMALVRAM]
  --lowvram             Split the unet in parts to use less vram. [env var:
                        COMFYUI_LOWVRAM]
  --novram              When lowvram isn't enough. [env var: COMFYUI_NOVRAM]
  --cpu                 To use the CPU for everything (slow). [env var:
                        COMFYUI_CPU]
  --reserve-vram RESERVE_VRAM
                        Set the amount of vram in GB you want to reserve for
                        use by your OS/other software. Defaults to 0.0, since
                        this isn't conceptually robust anyway. [env var:
                        COMFYUI_RESERVE_VRAM]
  --async-offload [NUM_STREAMS]
                        Use async weight offloading. An optional argument
                        controls the amount of offload streams. Default is 2.
                        Enabled by default on Nvidia. [env var:
                        COMFYUI_ASYNC_OFFLOAD]
  --disable-async-offload
                        Disable async weight offloading. [env var:
                        COMFYUI_DISABLE_ASYNC_OFFLOAD]
  --force-non-blocking  Force ComfyUI to use non-blocking operations for all
                        applicable tensors. This may improve performance on
                        some non-Nvidia systems but can cause issues with some
                        workflows. [env var: COMFYUI_FORCE_NON_BLOCKING]
  --default-hashing-function {md5,sha1,sha256,sha512}
                        Allows you to choose the hash function to use for
                        duplicate filename / contents comparison. Default is
                        sha256. [env var: COMFYUI_DEFAULT_HASHING_FUNCTION]
  --disable-smart-memory
                        Force ComfyUI to aggressively offload to regular ram
                        instead of keeping models in VRAM when it can. [env
                        var: COMFYUI_DISABLE_SMART_MEMORY]
  --deterministic       Make pytorch use slower deterministic algorithms when
                        it can. Note that this might not make images
                        deterministic in all cases. [env var:
                        COMFYUI_DETERMINISTIC]
  --fast [FAST ...]     Enable some untested and potentially quality
                        deteriorating optimizations. Pass a list specific
                        optimizations if you only want to enable specific
                        ones. Current valid optimizations: fp16_accumulation
                        fp8_matrix_mult cublas_ops autotune [env var:
                        COMFYUI_FAST]
  --disable-pinned-memory
                        Disable pinned memory use. [env var:
                        COMFYUI_DISABLE_PINNED_MEMORY]
  --mmap-torch-files    Use mmap when loading ckpt/pt files. [env var:
                        COMFYUI_MMAP_TORCH_FILES]
  --disable-mmap        Don't use mmap when loading safetensors. [env var:
                        COMFYUI_DISABLE_MMAP]
  --dont-print-server   Don't print server output. [env var:
                        COMFYUI_DONT_PRINT_SERVER]
  --quick-test-for-ci   Quick test for CI. Raises an error if nodes cannot be
                        imported, [env var: COMFYUI_QUICK_TEST_FOR_CI]
  --windows-standalone-build
                        Windows standalone build: Enable convenient things
                        that most people using the standalone windows build
                        will probably enjoy (like auto opening the page on
                        startup). [env var: COMFYUI_WINDOWS_STANDALONE_BUILD]
  --disable-metadata    Disable saving prompt metadata in files. [env var:
                        COMFYUI_DISABLE_METADATA]
  --disable-all-custom-nodes
                        Disable loading all custom nodes. [env var:
                        COMFYUI_DISABLE_ALL_CUSTOM_NODES]
  --whitelist-custom-nodes WHITELIST_CUSTOM_NODES [WHITELIST_CUSTOM_NODES ...]
                        Specify custom node folders to load even when
                        --disable-all-custom-nodes is enabled. [env var:
                        COMFYUI_WHITELIST_CUSTOM_NODES]
  --blacklist-custom-nodes BLACKLIST_CUSTOM_NODES [BLACKLIST_CUSTOM_NODES ...]
                        Specify custom node folders to never load. Accepts
                        shell-style globs. [env var:
                        COMFYUI_BLACKLIST_CUSTOM_NODES]
  --disable-api-nodes   Disable loading all api nodes. Also prevents the
                        frontend from communicating with the internet. [env
                        var: COMFYUI_DISABLE_API_NODES]
  --enable-eval         Enable nodes that can evaluate Python code in
                        workflows. [env var: COMFYUI_ENABLE_EVAL]
  --multi-user          Enables per-user storage. [env var:
                        COMFYUI_MULTI_USER]
  --create-directories  Creates the default models/, input/, output/ and temp/
                        directories, then exits. [env var:
                        COMFYUI_CREATE_DIRECTORIES]
  --log-stdout          Send normal process output to stdout instead of stderr
                        (default). [env var: COMFYUI_LOG_STDOUT]
  --plausible-analytics-base-url PLAUSIBLE_ANALYTICS_BASE_URL
                        Enables server-side analytics events sent to the
                        provided URL. [env var:
                        COMFYUI_PLAUSIBLE_ANALYTICS_BASE_URL]
  --plausible-analytics-domain PLAUSIBLE_ANALYTICS_DOMAIN
                        Specifies the domain name for analytics events. [env
                        var: COMFYUI_PLAUSIBLE_ANALYTICS_DOMAIN]
  --analytics-use-identity-provider
                        Uses platform identifiers for unique visitor
                        analytics. [env var:
                        COMFYUI_ANALYTICS_USE_IDENTITY_PROVIDER]
  --distributed-queue-connection-uri DISTRIBUTED_QUEUE_CONNECTION_URI
                        EXAMPLE: "amqp://guest:guest@127.0.0.1" - Servers and
                        clients will connect to this AMPQ URL to form a
                        distributed queue and exchange prompt execution
                        requests and progress updates. [env var:
                        COMFYUI_DISTRIBUTED_QUEUE_CONNECTION_URI]
  --distributed-queue-worker
                        Workers will pull requests off the AMQP URL. [env var:
                        COMFYUI_DISTRIBUTED_QUEUE_WORKER]
  --distributed-queue-frontend
                        Frontends will start the web UI and connect to the
                        provided AMQP URL to submit prompts. [env var:
                        COMFYUI_DISTRIBUTED_QUEUE_FRONTEND]
  --distributed-queue-name DISTRIBUTED_QUEUE_NAME
                        This name will be used by the frontends and workers to
                        exchange prompt requests and replies. Progress updates
                        will be prefixed by the queue name, followed by a '.',
                        then the user ID [env var:
                        COMFYUI_DISTRIBUTED_QUEUE_NAME]
  --external-address EXTERNAL_ADDRESS
                        Specifies a base URL for external addresses reported
                        by the API, such as for image paths. [env var:
                        COMFYUI_EXTERNAL_ADDRESS]
  --logging-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Set the logging level [env var: COMFYUI_LOGGING_LEVEL]
  --disable-known-models
                        Disables automatic downloads of known models and
                        prevents them from appearing in the UI. [env var:
                        COMFYUI_DISABLE_KNOWN_MODELS]
  --max-queue-size MAX_QUEUE_SIZE
                        The API will reject prompt requests if the queue's
                        size exceeds this value. [env var:
                        COMFYUI_MAX_QUEUE_SIZE]
  --otel-service-name OTEL_SERVICE_NAME
                        The name of the service or application that is
                        generating telemetry data. [env var:
                        OTEL_SERVICE_NAME]
  --otel-service-version OTEL_SERVICE_VERSION
                        The version of the service or application that is
                        generating telemetry data. [env var:
                        OTEL_SERVICE_VERSION]
  --otel-exporter-otlp-endpoint OTEL_EXPORTER_OTLP_ENDPOINT
                        A base endpoint URL for any signal type, with an
                        optionally-specified port number. Helpful for when
                        you're sending more than one signal to the same
                        endpoint and want one environment variable to control
                        the endpoint. [env var: OTEL_EXPORTER_OTLP_ENDPOINT]
  --force-channels-last
                        Force channels last format when inferencing the
                        models. [env var: COMFYUI_FORCE_CHANNELS_LAST]
  --force-hf-local-dir-mode
                        Download repos from huggingface.co to the
                        models/huggingface directory with the "local_dir"
                        argument instead of models/huggingface_cache with the
                        "cache_dir" argument, recreating the traditional file
                        structure. [env var: COMFYUI_FORCE_HF_LOCAL_DIR_MODE]
  --enable-video-to-image-fallback
                        Enable fallback to convert video frames to images for
                        models that do not natively support video inputs. [env
                        var: COMFYUI_ENABLE_VIDEO_TO_IMAGE_FALLBACK]
  --front-end-version FRONT_END_VERSION
                        Specifies the version of the frontend to be used. This
                        command needs internet connectivity to query and
                        download available frontend implementations from
                        GitHub releases. The version string should be in the
                        format of: [repoOwner]/[repoName]@[version] where
                        version is one of: "latest" or a valid version number
                        (e.g. "1.0.0") [env var: COMFYUI_FRONT_END_VERSION]
  --panic-when PANIC_WHEN [PANIC_WHEN ...]
                        List of fully qualified exception class names to panic
                        (sys.exit(1)) when a workflow raises it. Example:
                        --panic-when=torch.cuda.OutOfMemoryError. Can be
                        specified multiple times or as a comma-separated list.
                        [env var: COMFYUI_PANIC_WHEN]
  --front-end-root FRONT_END_ROOT
                        The local filesystem path to the directory where the
                        frontend is located. Overrides --front-end-version.
                        [env var: COMFYUI_FRONT_END_ROOT]
  --executor-factory EXECUTOR_FACTORY
                        When running ComfyUI as a distributed worker, this
                        specifies the kind of executor that should be used to
                        run the actual ComfyUI workflow worker. A
                        ThreadPoolExecutor is the default. A
                        ProcessPoolExecutor results in better memory
                        management, since the process will be closed and
                        large, contiguous blocks of CUDA memory can be freed.
                        [env var: COMFYUI_EXECUTOR_FACTORY]
  --openai-api-key OPENAI_API_KEY
                        Configures the OpenAI API Key for the OpenAI nodes.
                        Visit https://platform.openai.com/api-keys to create
                        this key. [env var: OPENAI_API_KEY]
  --ideogram-api-key IDEOGRAM_API_KEY
                        Configures the Ideogram API Key for the Ideogram
                        nodes. Visit https://ideogram.ai/manage-api to create
                        this key. [env var: IDEOGRAM_API_KEY]
  --anthropic-api-key ANTHROPIC_API_KEY
                        Configures the Anthropic API key for its nodes related
                        to Claude functionality. Visit
                        https://console.anthropic.com/settings/keys to create
                        this key. [env var: ANTHROPIC_API_KEY]
  --user-directory USER_DIRECTORY
                        Set the ComfyUI user directory with an absolute path.
                        Overrides --base-directory. [env var:
                        COMFYUI_USER_DIRECTORY]
  --enable-compress-response-body
                        Enable compressing response body. [env var:
                        COMFYUI_ENABLE_COMPRESS_RESPONSE_BODY]
  --comfy-api-base COMFY_API_BASE
                        Set the base URL for the ComfyUI API. (default:
                        https://api.comfy.org) [env var:
                        COMFYUI_COMFY_API_BASE]
  --block-runtime-package-installation
                        When set, custom nodes like ComfyUI Manager, Easy Use,
                        Nunchaku and others will not be able to use pip or uv
                        to install packages at runtime (experimental). [env
                        var: COMFYUI_BLOCK_RUNTIME_PACKAGE_INSTALLATION]
  --database-url DATABASE_URL
                        Specify the database URL, e.g. for an in-memory
                        database you can use 'sqlite:///:memory:'. [env var:
                        COMFYUI_DATABASE_URL]
  --workflows WORKFLOWS [WORKFLOWS ...]
                        Execute the API workflow(s) specified in the provided
                        files. For each workflow, its outputs will be printed
                        to a line to standard out. Application logging will be
                        redirected to standard error. Use `-` to signify
                        standard in. [env var: COMFYUI_WORKFLOWS]
  --disable-requests-caching
                        Disable requests caching (useful for testing) [env
                        var: COMFYUI_DISABLE_REQUESTS_CACHING]

Args that start with '--' can also be set in a config file (config.yaml or
config.json or config.cfg or config.ini or specified via -c). Config file
syntax allows: key=value, flag=true, stuff=[a,b,c] (for details, see syntax at
https://goo.gl/R74nmi). In general, command-line values override environment
variables which override config file values which override defaults.

```
