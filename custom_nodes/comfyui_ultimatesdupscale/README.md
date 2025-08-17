# ComfyUI_UltimateSDUpscale

 [ComfyUI](https://github.com/comfyanonymous/ComfyUI) nodes for the [Ultimate Stable Diffusion Upscale script by Coyote-A](https://github.com/Coyote-A/ultimate-upscale-for-automatic1111). This is a wrapper for the script used in the A1111 extension.

## Installation

Enter the following command from the commandline starting in ComfyUI/custom_nodes/
```
git clone https://github.com/ssitu/ComfyUI_UltimateSDUpscale --recursive
```

## Usage

Nodes can be found in the node menu under `image/upscaling`:

|Node|Description|
| --- | --- |
| Ultimate SD Upscale | The primary node that has the most of the inputs as the original extension script. |
| Ultimate SD Upscale <br>(No Upscale) | Same as the primary node, but without the upscale inputs and assumes that the input image is already upscaled. Use this if you already have an upscaled image or just want to do the tiled sampling. |
| Ultimate SD Upscale <br>(Custom Sample) | Same as the primary node, but has additional inputs for a custom sampler and custom sigmas. Both must be provided if one is used. If neither is provided, the widgets (the settings below the input slots) for the sampler and step/denoise settings will be used, like in the base USDU node. |

---

Details about most of the parameters can be found [here](https://github.com/Coyote-A/ultimate-upscale-for-automatic1111/wiki/FAQ#parameters-descriptions).

Parameters not found in the original repository:

* `upscale_by` The number to multiply the width and height of the image by. If you want to specify an exact width and height, use the "No Upscale" version of the node and perform the upscaling separately (e.g., ImageUpscaleWithModel -> ImageScale -> UltimateSDUpscaleNoUpscale).
* `force_uniform_tiles` If enabled, tiles that would be cut off by the edges of the image will expand the tile using the rest of the image to keep the same tile size determined by `tile_width` and `tile_height`, which is what the A1111 Web UI does. If disabled, the minimal size for tiles will be used, which may make the sampling faster but may cause artifacts due to irregular tile sizes.

## Examples

#### Using the ControlNet tile model:

![image](https://github.com/ssitu/ComfyUI_UltimateSDUpscale/assets/57548627/64f8d3b2-10ae-45ee-9f8a-40b798a51655)
