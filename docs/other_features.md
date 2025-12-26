# SVG Conversion and String Saving

ComfyUI LTS supports powerful SVG conversion capabilities using vtracer and Skia, along with enhanced string saving functionality. This allows for seamless conversion between raster images and SVG format, as well as flexible string saving options.

![SVG Conversion Example](assets/svg_01.png)

In this example, a raster image is converted to SVG, potentially modified, and then converted back to a raster image. The resulting image and SVG code can be saved.

You can try the [SVG Conversion Workflow](../tests/inference/workflows/svg-0.json) to explore these features.

# Ideogram

First class support for Ideogram, currently the best still images model.

Visit [API key management](https://ideogram.ai/manage-api) and set the environment variable `IDEOGRAM_API_KEY` to it.

The `IdeogramEdit` node expects the white areas of the mask to be kept, and the black areas of the mask to be inpainted.

Use the **Fit Image to Diffusion Size** with the **Ideogram** resolution set to correctly fit images for inpainting.
