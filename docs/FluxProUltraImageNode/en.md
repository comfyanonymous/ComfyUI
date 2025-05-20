![ComfyUI Native Flux 1.1 [pro] Ultra Image node](assets/flux-1-1-pro-ultra-image.jpg)

The Flux 1.1 [pro] Ultra Image node allows you to generate ultra-high-resolution images through text prompts, directly connecting to Black Forest Labs' latest image generation API.

This node supports two main usage modes:
1. **Text-to-Image**: Generate high-quality images from text prompts (when no image input is used)
2. **Image-to-Image**: Combine existing images with prompts to create new images that blend features from both (Remix mode)

This node supports Ultra mode through API calls, capable of generating images at 4 times the resolution of standard Flux 1.1 [pro] (up to 4MP), without sacrificing prompt adherence, and maintaining super-fast generation times of just 10 seconds. Compared to other high-resolution models, it's more than 2.5 times faster.

## Parameter Description

### Basic Parameters

| Parameter         | Type    | Default | Description                                                                                                                                                                                                                      |
| ----------------- | ------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| prompt            | String  | ""      | Text description for generating the image                                                                                                                                                                                        |
| prompt_upsampling | Boolean | False   | Whether to use prompt upsampling technique to enhance details. When enabled, automatically modifies prompts for more creative generation, but results become non-deterministic (same seed won't produce exactly the same result) |
| seed              | Integer | 0       | Random seed value, controls generation randomness                                                                                                                                                                                |
| aspect_ratio      | String  | "16:9"  | Width-to-height ratio of the image, must be between 1:4 and 4:1                                                                                                                                                                  |
| raw               | Boolean | False   | When set to True, generates less processed, more natural-looking images                                                                                                                                                          |

### Optional Parameters

| Parameter             | Type  | Default | Description                                                                                                                                               |
| --------------------- | ----- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| image_prompt          | Image | None    | Optional input, used for Image-to-Image (Remix) mode                                                                                                      |
| image_prompt_strength | Float | 0.1     | Active when `image_prompt` is input, adjusts the blend between prompt and image prompt. Higher values make output closer to input image, range is 0.0-1.0 |

### Output

| Output | Type  | Description                            |
| ------ | ----- | -------------------------------------- |
| IMAGE  | Image | Generated high-resolution image result |

## How It Works

Flux 1.1 [pro] Ultra mode uses optimized deep learning architecture and efficient GPU acceleration technology to achieve high-resolution image generation without sacrificing speed. When a request is sent to the API, the system parses the prompt, applies appropriate parameters, then computes the image in parallel, finally generating and returning the high-resolution result.

Compared to regular models, Ultra mode particularly focuses on detail preservation and consistency at large scales, ensuring impressive quality even at 4MP high resolution.