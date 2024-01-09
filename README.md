# ComfyTS

## The most powerful and modular stable diffusion GUI and backend.

![ComfyUI Screenshot](comfyui_screenshot.png)

ComfyTS ("Comfy-The-Sequel" or "Comfy-TypeScript") is a fork of ComfyUI. Project goals:

- Fix issues with ComfyUI
- Adapt ComfyUI to work in a serverless, multi-user environment more easily
- Maintain compatability with the existing ComfyUI ecosystem of custom-nodes and workflows

### Docker Instructions:

- Start your docker daemon, then in the root folder run the build command:

  `docker build -t voidtech0/comfy-ts:0.1.0 .`

Note that the docker-build does not copy the models in the docker-image (that would be stupid). Instead, it expects to load the models from an NFS-drive mounted to the container on startup.

### Docker To Do:

- Make sure filesystem cache and sym-links are working.
- Do we really need the extra_model_paths?
- We probably won't need sym-links and extra-model paths anymore to be honest; we can build those into comfy-ts directly.
- Stop custom nodes from downloading external files and doing pip-install at runtime (on startup). We should ensure that's all done at build-time.

### General To Do:

- Add ComfyUI manager into this repo by default
- Add a startup flag to turn off the ComfyUI manager and other settings. (This is for when running ComfyTS in a cloud environment, where users downloading custom-nodes would be inappropriate.)
- Add a startup flag to switch between using ComfyUI-local or using the void-tech API
