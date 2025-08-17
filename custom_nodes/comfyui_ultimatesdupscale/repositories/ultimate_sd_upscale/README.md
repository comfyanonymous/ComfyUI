# Ultimate SD Upscale extension for [AUTOMATIC1111 Stable Diffusion web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
Now you have the opportunity to use a large denoise (0.3-0.5) and not spawn many artifacts. Works on any video card, since you can use a 512x512 tile size and the image will converge.

News channel: https://t.me/usdunews

# Instructions
All instructions can be found on the project's [wiki](https://github.com/Coyote-A/ultimate-upscale-for-automatic1111/wiki).

# Refs

https://github.com/ssitu/ComfyUI_UltimateSDUpscale - Implementation for ComfyUI

# Examples
More on [wiki page](https://github.com/Coyote-A/ultimate-upscale-for-automatic1111/wiki/Examples)

<details> 
  <summary>E1</summary>
  Original image

  ![Original](https://i.imgur.com/J8mRYOD.png)

  2k upscaled. **Tile size**: 512, **Padding**: 32, **Mask blur**: 16, **Denoise**: 0.4
  ![2k upscale](https://i.imgur.com/0aKua4r.png)
</details>

<details> 
  <summary>E2</summary>
  Original image

  ![Original](https://i.imgur.com/aALNI2w.png)

  2k upscaled. **Tile size**: 768, **Padding**: 55, **Mask blur**: 20, **Denoise**: 0.35
  ![2k upscale](https://i.imgur.com/B5PHz0J.png)

  4k upscaled. **Tile size**: 768, **Padding**: 55, **Mask blur**: 20, **Denoise**: 0.35
  ![4k upscale](https://i.imgur.com/tIUQ7TJ.jpg)
</details>

<details> 
  <summary>E3</summary>
  Original image

  ![Original](https://i.imgur.com/AGtszA8.png)

  4k upscaled. **Tile size**: 768, **Padding**: 55, **Mask blur**: 20, **Denoise**: 0.4
  ![4k upscale](https://i.imgur.com/LCYLfCs.jpg)
</details>

# API Usage

```javascript
{
"script_name" : "ultimate sd upscale",
"script_args" : [
	null, // _ (not used)
	512, // tile_width
	512, // tile_height
	8, // mask_blur
	32, // padding
	64, // seams_fix_width
	0.35, // seams_fix_denoise
	32, // seams_fix_padding
	0, // upscaler_index
	true, // save_upscaled_image a.k.a Upscaled
	0, // redraw_mode
	false, // save_seams_fix_image a.k.a Seams fix
	8, // seams_fix_mask_blur
	0, // seams_fix_type
	0, // target_size_type
	2048, // custom_width
	2048, // custom_height
	2 // custom_scale
]
}
```
upscaler_index
| Value         |  |
|:-------------:| -----:|
| 0 | None |
| 1 | Lanczos |
| 2 | Nearest |
| 3 | ESRGAN_4x |
| 4 | LDSR |
| 5 | R-ESRGAN_4x+ |
| 6 | R-ESRGAN 4x+ Anime6B |
| 7 | ScuNET GAN |
| 8 | ScuNET PSNR |
| 9 | SwinIR 4x |

redraw_mode
| Value         |  |
|:-------------:| -----:|
| 0 | Linear |
| 1 | Chess |
| 2 | None |

seams_fix_mask_blur
| Value         |  |
|:-------------:| -----:|
| 0 | None |
| 1 | BAND_PASS |
| 2 | HALF_TILE |
| 3 | HALF_TILE_PLUS_INTERSECTIONS |

seams_fix_type
| Value         |  |
|:-------------:| -----:|
| 0 | None |
| 1 | Band pass |
| 2 | Half tile offset pass |
| 3 | Half tile offset pass + intersections |

seams_fix_type
| Value         |  |
|:-------------:| -----:|
| 0 | From img2img2 settings |
| 1 | Custom size |
| 2 | Scale from image size |

