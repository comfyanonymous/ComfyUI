[![Youtube Badge](https://img.shields.io/badge/Youtube-FF0000?style=for-the-badge&logo=Youtube&logoColor=white&link=https://www.youtube.com/watch?v=AccoxDZIg3Y&list=PL_Ej2RDzjQLGfEeizq4GISeY3FtVyFmGP)](https://www.youtube.com/watch?v=AccoxDZIg3Y&list=PL_Ej2RDzjQLGfEeizq4GISeY3FtVyFmGP)

# ComfyUI-Impact-Pack

**Custom node pack for ComfyUI**
This node pack helps to conveniently enhance images through Detector, Detailer, Upscaler, Pipe, and more.

NOTE: The UltralyticsDetectorProvider node is not part of the ComfyUI-Impact-Pack. To use the UltralyticsDetectorProvider node, please install the ComfyUI-Impact-Subpack separately.

## NOTICE 
* V8.19: legacy nodes (mmdet and etc.) are removed
* V8.18: Support [facebookresearch/sam2](https://github.com/facebookresearch/sam2) models
* V8.0: The `Impact Subpack` is no longer installed automatically. To use `UltralyticsDetectorProvider` nodes, please install the `Impact Subpack` separately.
* V7.6: Automatic installation is no longer supported. Please install using ComfyUI-Manager, or manually install requirements.txt and run install.py to complete the installation.
* V7.0: Supports Switch based on Execution Model Inversion.
* V6.0: Supports FLUX.1 model in Impact KSampler, Detailers, PreviewBridgeLatent
* V5.0: It is no longer compatible with versions of ComfyUI before 2024.04.08. 
* V4.87.4: Update to a version of ComfyUI after 2024.04.08 for proper functionality.
* V4.85: Incompatible with the outdated **ComfyUI IPAdapter Plus**. (A version dated March 24th or later is required.)
* V4.77: Compatibility patch applied. Requires ComfyUI version (Oct. 8th) or later.
* V4.73.3: ControlNetApply (SEGS) supports AnimateDiff
* V4.20.1: Due to the feature update in `RegionalSampler`, the parameter order has changed, causing malfunctions in previously created `RegionalSamplers`. Please adjust the parameters accordingly.
* V4.12: `MASKS` is changed to `MASK`.
* V4.7.2 isn't compatible with old version of `ControlNet Auxiliary Preprocessor`. If you will use `MediaPipe FaceMesh to SEGS` update to latest version(Sep. 17th).  
* Selection weight syntax is changed(: -> ::) since V3.16. ([tutorial](https://github.com/ltdrdata/ComfyUI-extension-tutorials/blob/Main/ComfyUI-Impact-Pack/tutorial/ImpactWildcardProcessor.md))
* Starting from V3.6, requires latest version(Aug 8, 9ccc965) of ComfyUI.
* **In versions below V3.3.1, there was an issue with the image quality generated after using the UltralyticsDetectorProvider. Please make sure to upgrade to a newer version.**
* Starting from V3.0, nodes related to `mmdet` are optional nodes that are activated only based on the configuration settings.
  - Through ComfyUI-Impact-Subpack, you can utilize UltralyticsDetectorProvider to access various detection models.
* Between versions 2.22 and 2.21, there is partial compatibility loss regarding the Detailer workflow. If you continue to use the existing workflow, errors may occur during execution. An additional output called "enhanced_alpha_list" has been added to Detailer-related nodes.
* The permission error related to cv2 that occurred during the installation of Impact Pack has been patched in version 2.21.4. However, please note that the latest versions of ComfyUI and ComfyUI-Manager are required.
* The "PreviewBridge" feature may not function correctly on ComfyUI versions released before July 1, 2023.
* Attempting to load the "ComfyUI-Impact-Pack" on ComfyUI versions released before June 27, 2023, will result in a failure.
* With the addition of wildcard support in FaceDetailer, the structure of DETAILER_PIPE-related nodes and Detailer nodes has changed. There may be malfunctions when using the existing workflow.


## How To Install

### **Recommended**
* Install via [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager).

### **Manual**
* Navigate to `ComfyUI/custom_nodes` in your terminal (cmd).
* Clone the repository under the `custom_nodes` directory using the following command:
  ```
  git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack comfyui-impact-pack
  cd comfyui-impact-pack
  ```
* Install dependencies in your Python environment.
    * For Windows Portable, run the following command inside `ComfyUI\custom_nodes\comfyui-impact-pack`:
        ```
        ..\..\..\python_embeded\python.exe -m pip install -r requirements.txt
        ```
    * If using venv or conda, activate your Python environment first, then run:
        ```
        pip install -r requirements.txt
        ```

### Companion Pack
* If you need the `Ultralytics Detector Provider` to use various YOLO detection models, you should also install [ComfyUI-Impact-Subpack](https://github.com/ltdrdata/ComfyUI-Impact-Subpack).


## Custom Nodes
### [Detector nodes](https://github.com/ltdrdata/ComfyUI-extension-tutorials/blob/Main/ComfyUI-Impact-Pack/tutorial/detectors.md)
  * `SAMLoader (Impact)` - Loads the SAM model.
  * `ONNXDetectorProvider` - Loads the ONNX model to provide BBOX_DETECTOR.
  * `CLIPSegDetectorProvider` - Wrapper for CLIPSeg to provide BBOX_DETECTOR.
    * You need to install the ComfyUI-CLIPSeg node extension.
  * `SEGM Detector (combined)` - Detects segmentation and returns a mask from the input image.
  * `BBOX Detector (combined)` - Detects bounding boxes and returns a mask from the input image.
  * `SAMDetector (combined)` - Utilizes the SAM technology to extract the segment at the location indicated by the input SEGS on the input image and outputs it as a unified mask.
  * `SAMDetector (Segmented)` - It is similar to `SAMDetector (combined)`, but it separates and outputs the detected segments. Multiple segments can be found for the same detected area, and currently, a policy is in place to group them arbitrarily in sets of three. This aspect is expected to be improved in the future.
    * As a result, it outputs the `combined_mask`, which is a unified mask, and `batch_masks`, which are multiple masks grouped together in batch form.
    * While `batch_masks` may not be completely separated, it provides functionality to perform some level of segmentation.
  * `Simple Detector (SEGS)` - Operating primarily with `BBOX_DETECTOR`, and with the additional provision of `SAM_MODEL` or `SEGM_DETECTOR`, this node internally generates improved SEGS through mask operations on both *bbox* and *silhouette*. It serves as a convenient tool to simplify a somewhat intricate workflow.
  * `Simple Detector for Video (SEGS)` – Performs detection on videos composed of image frames. Instead of using a single mask, it performs detection individually on each image frame and generates a SEGS object with a batch of masks. 
  * `SAM2 Video Detector (SEGS)` – Similar to `Simple Detector for Video (SEGS)`, but utilizes SAM2’s video tracking technology to generate a SEGS object with a batch of masks. 
      * To use this node, you must select a SAM2 model in the SAMLoader.


### ControlNet, IPAdapter
  * `ControlNetApply (SEGS)` - To apply ControlNet in SEGS, you need to use the Preprocessor Provider node from the Inspire Pack to utilize this node.
    * `segs_preprocessor` and `control_image` can be selectively applied. If a `control_image` is given, `segs_preprocessor` will be ignored.
    * If set to `control_image`, you can preview the cropped cnet image through `SEGSPreview (CNET Image)`. Images generated by `segs_preprocessor` should be verified through the `cnet_images` output of each Detailer.
    * The `segs_preprocessor` operates by applying preprocessing on-the-fly based on the cropped image during the detailing process, while `control_image` will be cropped and used as input to `ControlNetApply (SEGS)`.
  * `ControlNetClear (SEGS)` - Clear applied ControlNet in SEGS
  * `IPAdapterApply (SEGS)` - To apply IPAdapter in SEGS, you need to use the Preprocessor Provider node from the Inspire Pack to utilize this node.


### Mask operation
  * `Pixelwise(SEGS & SEGS)` - Performs a 'pixelwise and' operation between two SEGS.
  * `Pixelwise(SEGS - SEGS)` - Subtracts one SEGS from another.
  * `Pixelwise(SEGS & MASK)` - Performs a pixelwise AND operation between SEGS and MASK.
  * `Pixelwise(SEGS & MASKS ForEach)` - Performs a pixelwise AND operation between SEGS and MASKS.
    * Please note that this operation is performed with batches of MASKS, not just a single MASK.
  * `Pixelwise(MASK & MASK)` - Performs a 'pixelwise and' operation between two masks.
  * `Pixelwise(MASK - MASK)` - Subtracts one mask from another.
  * `Pixelwise(MASK + MASK)` - Combine two masks.
  * `SEGM Detector (SEGS)` - Detects segmentation and returns SEGS from the input image.
  * `BBOX Detector (SEGS)` - Detects bounding boxes and returns SEGS from the input image.
  * `Dilate Mask` - Dilate Mask.
    * Support erosion for negative value.
  * `Gaussian Blur Mask` - Apply Gaussian Blur to Mask. You can utilize this for mask feathering.
  * `Mask Rect Area` - Create a rectangular mask defined by percentages with preview canvas.
  * `Mask Rect Area (Advanced)` - Create a rectangular mask defined by pixels and image size. 


### [Detailer nodes](https://github.com/ltdrdata/ComfyUI-extension-tutorials/blob/Main/ComfyUI-Impact-Pack/tutorial/detailers.md)
  * `Detailer (SEGS)` - Refines the image based on SEGS.
  * `Detailer (SEGS) with auto retry` - Refines the image based on SEGS and will automatically retry if the patch is all black.
  * `DetailerDebug (SEGS)` - Refines the image based on SEGS. Additionally, it provides the ability to monitor the cropped image and the refined image of the cropped image.
    * To prevent regeneration caused by the seed that does not change every time when using 'external_seed', please disable the 'seed random generate' option in the 'Detailer...' node.
  * `MASK to SEGS` - Generates SEGS based on the mask.
  * `MASK to SEGS For Video` - Generates SEGS based on the mask for Video. (Renamed from `MASK to SEGS For AnimateDiff`)
    * When using a single mask, convert it to SEGS to apply it to the entire frame.
    * When using a batch mask, the contour fill feature is disabled.
  * `MediaPipe FaceMesh to SEGS` - Separate each landmark from the mediapipe facemesh image to create labeled SEGS.
    * Usually, the size of images created through the MediaPipe facemesh preprocessor is downscaled. It resizes the MediaPipe facemesh image to the original size given as reference_image_opt for matching sizes during processing. 
  * `ToBinaryMask` - Separates the mask generated with alpha values between 0 and 255 into 0 and 255. The non-zero parts are always set to 255.
  * `Masks to Mask List` - This node converts the MASKS in batch form to a list of individual masks.
  * `Mask List to Masks` - This node converts the MASK list to MASK batch form.
  * `EmptySEGS` - Provides an empty SEGS.
  * `MaskPainter` - Provides a feature to draw masks.
  * `FaceDetailer` - Easily detects faces and improves them.
  * `FaceDetailer (pipe)` - Easily detects faces and improves them (for multipass).
  * `MaskDetailer (pipe)` - This is a simple inpaint node that applies the Detailer to the mask area.

  * `FromDetailer (SDXL/pipe)`, `BasicPipe -> DetailerPipe (SDXL)`, `Edit DetailerPipe (SDXL)` - These are pipe functions used in Detailer for utilizing the refiner model of SDXL.
  * `Any PIPE -> BasicPipe` - Convert the PIPE Value of other custom nodes that are not BASIC_PIPE but internally have the same structure as BASIC_PIPE to BASIC_PIPE. If an incompatible type is applied, it may cause runtime errors.


### SEGS Manipulation nodes
  * `SEGSDetailer` - Performs detailed work on SEGS without pasting it back onto the original image.
  * `SEGSPaste` - Pastes the results of SEGS onto the original image.
    * If `ref_image_opt` is present, the images contained within SEGS are ignored. Instead, the image within `ref_image_opt` corresponding to the crop area of SEGS is taken and pasted. The size of the image in `ref_image_opt` should be the same as the original image size.
    * This node can be used in conjunction with the processing results of AnimateDiff.
  * `SEGSPreview` - Provides a preview of SEGS.
     * This option is used to preview the improved image through `SEGSDetailer` before merging it into the original. Prior to going through ```SEGSDetailer```, SEGS only contains mask information without image information. If fallback_image_opt is connected to the original image, SEGS without image information will generate a preview using the original image. However, if SEGS already contains image information, fallback_image_opt will be ignored.
     * This node can be used in conjunction with the processing results of AnimateDiff.
  * `SEGSPreview (CNET Image)` - Show images configured with `ControlNetApply (SEGS)` for debugging purposes.
  * `SEGSToImageList` - Convert SEGS To Image List
  * `SEGSToMaskList` - Convert SEGS To Mask List
  * `SEGS Filter (label)` - This node filters SEGS based on the label of the detected areas. 
  * `SEGS Filter (ordered)` - This node sorts SEGS based on size and position and retrieves SEGs within a certain range. 
  * `SEGS Filter (range)` - This node retrieves only SEGs from SEGS that have a size and position within a certain range.
  * `SEGS Filter (non max suppression)` - This node filters SEGS by removing those with high overlap based on the Intersection over Union (IoU) threshold, keeping only the most confident detections.
  * `SEGS Filter (intersection)` - This node filters segs1, keeping only the SEGS that do not significantly overlap with any SEGS in segs2, based on the Intersection over Area (IoA) threshold.
  * `SEGS Assign (label)` - Assign labels sequentially to SEGS. This node is useful when used with `[LAB]` of FaceDetailer.
  * `SEGSConcat` - Concatenate segs1 and segs2. If source shape of segs1 and segs2 are different from segs2 will be ignored.
  * `SEGS Merge` - SEGS contains multiple SEGs. SEGS Merge integrates several SEGs into a single merged SEG. The label is changed to `merged` and the confidence becomes the minimum confidence. The applied controlnet and cropped_image are removed.
  * `Picker (SEGS)` - Among the input SEGS, you can select a specific SEG through a dialog. If no SEG is selected, it outputs an empty SEGS. Increasing the batch_size of SEGSDetailer can be used for the purpose of selecting from the candidates.
  * `Set Default Image For SEGS` - Set a default image for SEGS. SEGS with images set this way do not need to have a fallback image set. When override is set to false, the original image is preserved.
  * `Remove Image from SEGS` - Remove the image set for the SEGS that has been configured by "Set Default Image for SEGS" or SEGSDetailer. When the image for the SEGS is removed, the Detailer node will operate based on the currently processed image instead of the SEGS. 
  * `Make Tile SEGS` - [experimental] Create SEGS in the form of tiles from an image to facilitate experiments for Tiled Upscale using the Detailer.
    * The `filter_in_segs_opt` and `filter_out_segs_opt` are optional inputs. If these inputs are provided, when creating the tiles, the mask for each tile is generated by overlapping with the mask of `filter_in_segs_opt` and excluding the overlap with the mask of `filter_out_segs_opt`. Tiles with an empty mask will not be created as SEGS.
  * `Dilate Mask (SEGS)` - Dilate/Erosion Mask in SEGS
  * `Gaussian Blur Mask (SEGS)` - Apply Gaussian Blur to Mask in SEGS
  * `SEGS_ELT Manipulation` - experimental nodes
    * `DecomposeSEGS` - Decompose SEGS to allow for detailed manipulation.
    * `AssembleSEGS` - Reassemble the decomposed SEGS.
    * `From SEG_ELT` - Extract detailed information from SEG_ELT.
    * `Edit SEG_ELT` - Modify some of the information in SEG_ELT.
    * `Dilate SEG_ELT` - Dilate the mask of SEG_ELT.
    * `From SEG_ELT` bbox - Extract coordinate from bbox in SEG_ELT
    * `From SEG_ELT` crop_region - Extract coordinate from crop_region in SEG_ELT
  * `Count Elt in SEGS` - Number of Elts ins SEGS
 

### Pipe nodes
   * `ToDetailerPipe`, `FromDetailerPipe` - These nodes are used to bundle multiple inputs used in the detailer, such as models and vae, ..., into a single DETAILER_PIPE or extract the elements that are bundled in the DETAILER_PIPE.
   * `ToBasicPipe`, `FromBasicPipe` - These nodes are used to bundle model, clip, vae, positive conditioning, and negative conditioning into a single BASIC_PIPE, or extract each element from the BASIC_PIPE.
   * `EditBasicPipe`, `EditDetailerPipe` - These nodes are used to replace some elements in BASIC_PIPE or DETAILER_PIPE.
   * `FromDetailerPipe_v2`, `FromBasicPipe_v2` - It has the same functionality as `FromDetailerPipe` and `FromBasicPipe`, but it has an additional output that directly exports the input pipe. It is useful when editing EditBasicPipe and EditDetailerPipe.
* `Latent Scale (on Pixel Space)` - This node converts latent to pixel space, upscales it, and then converts it back to latent.
   * If upscale_model_opt is provided, it uses the model to upscale the pixel and then downscales it using the interpolation method provided in scale_method to the target resolution.
* `PixelKSampleUpscalerProvider` - An upscaler is provided that converts latent to pixels using VAEDecode, performs upscaling, converts back to latent using VAEEncode, and then performs k-sampling. This upscaler can be attached to nodes such as `Iterative Upscale` for use.
  * Similar to `Latent Scale (on Pixel Space)`, if upscale_model_opt is provided, it performs pixel upscaling using the model.
* `PixelTiledKSampleUpscalerProvider` - It is similar to `PixelKSampleUpscalerProvider`, but it uses `ComfyUI_TiledKSampler` and Tiled VAE Decoder/Encoder to avoid GPU VRAM issues at high resolutions.
  * You need to install the [BlenderNeko/ComfyUI_TiledKSampler](https://github.com/BlenderNeko/ComfyUI_TiledKSampler) node extension.


### PK_HOOK
  * `DenoiseScheduleHookProvider` - IterativeUpscale provides a hook that gradually changes the denoise to target_denoise as the iterative-step progresses.
  * `CfgScheduleHookProvider` - IterativeUpscale provides a hook that gradually changes the cfg to target_cfg as the iterative-step progresses.
  * `StepsScheduleHookProvider` - IterativeUpscale provides a hook that gradually changes the sampling-steps to target_steps as the iterative-step progresses.
  * `NoiseInjectionHookProvider` - During each iteration of IterativeUpscale, noise is injected into the latent space while varying the strength according to a schedule.
    * You need to install the [BlenderNeko/ComfyUI_Noise](https://github.com/BlenderNeko/ComfyUI_Noise) node extension.
    * The seed serves as the initial value required for generating noise, and it increments by 1 with each iteration as the process unfolds.
    * The source determines the types of CPU noise and GPU noise to be configured.
    * Currently, there is only a simple schedule available, where the strength of the noise varies from start_strength to end_strength during the progression of each iteration.
  * `UnsamplerHookProvider` - Apply Unsampler during each iteration. To use this node, ComfyUI_Noise must be installed.
  * `PixelKSampleHookCombine` - This is used to connect two PK_HOOKs. hook1 is executed first and then hook2 is executed.
    * If you want to simultaneously change cfg and denoise, you can combine the PK_HOOKs of CfgScheduleHookProvider and PixelKSampleHookCombine.
 

### DETAILER_HOOK
  * `NoiseInjectionDetailerHookProvider` - The `detailer_hook` is a hook in the `Detailer` that injects noise during the processing of each SEGS.
  * `UnsamplerDetailerHookProvider` - Apply Unsampler during each cycle. To use this node, ComfyUI_Noise must be installed.
  * `DenoiseSchedulerDetailerHookProvider` - During the progress of the cycle, the detailer's denoise is altered up to the `target_denoise`. 
  * `CoreMLDetailerHookProvider` - CoreML supports only 512x512, 512x768, 768x512, 768x768 size sampling. CoreMLDetailerHookProvider precisely fixes the upscale of the crop_region to this size. When using this hook, it will always be selected size, regardless of the guide_size. However, if the guide_size is too small, skipping will occur.
  * `DetailerHookCombine` - This is used to connect two DETAILER_HOOKs. Similar to PixelKSampleHookCombine.
  * `SEGSOrderedFilterDetailerHook`, SEGSRangeFilterDetailerHook, SEGSLabelFilterDetailerHook - There are a wrapper node that provides SEGSFilter nodes to be applied in FaceDetailer or Detector by creating DETAILER_HOOK.
  * `PreviewDetailerHook` - Connecting this hook node helps provide assistance for viewing previews whenever SEGS Detailing tasks are completed. When working with a large number of SEGS, such as Make Tile SEGS, it allows for monitoring the situation as improvements progress incrementally.
    * Since this is the hook applied when pasting onto the original image, it has no effect on nodes like `SEGSDetailer`.
  * `VariationNoiseDetailerHookProvider` - Apply variation seed to the detailer. It can be applied in multiple stages through combine.
  * `CustomSamplerDetailerHookProvider` - Apply a hook that allows you to use a custom sampler in the Detailer nodes. When using `DetailerHookCombine`, the sampler from the first hook is applied.
  * `LamaRemoverDetailerHookProvider` – Applies Lama Remover to the upscaled image during the detailing stage. If `skip_sampling` is set to True, Lama Remover can be used alone without the detailing stage, allowing it to simply remove detected regions.
      * Not applicable for **AnimateDiff** detailers. When using `DetailerHookCombine`, `skip_sampling` is only applied if it is set to `True` for all hooks.
      * To use this node, the node pack at [Layer-norm/comfyui-lama-remover](https://github.com/Layer-norm/comfyui-lama-remover) must be installed.


### Iterative Upscale nodes
  * `Iterative Upscale (Latent/on Pixel Space)` - The upscaler takes the input upscaler and splits the scale_factor into steps, then iteratively performs upscaling. 
  This takes latent as input and outputs latent as the result.
  * `Iterative Upscale (Image)` - The upscaler takes the input upscaler and splits the scale_factor into steps, then iteratively performs upscaling. This takes image as input and outputs image as the result.
    * Internally, this node uses 'Iterative Upscale (Latent)'.


### TwoSamplers nodes
* `TwoSamplersForMask` - This node can apply two samplers depending on the mask area. The base_sampler is applied to the area where the mask is 0, while the mask_sampler is applied to the area where the mask is 1.
  * Note: The latent encoded through VAEEncodeForInpaint cannot be used.
* `KSamplerProvider` - This is a wrapper that enables KSampler to be used in TwoSamplersForMask TwoSamplersForMaskUpscalerProvider.
* `TiledKSamplerProvider` - ComfyUI_TiledKSampler is a wrapper that provides KSAMPLER.
  * You need to install the [BlenderNeko/ComfyUI_TiledKSampler](https://github.com/BlenderNeko/ComfyUI_TiledKSampler) node extension.
  
* `TwoAdvancedSamplersForMask` - TwoSamplersForMask is similar to TwoAdvancedSamplersForMask, but they differ in their operation. TwoSamplersForMask performs sampling in the mask area only after all the samples in the base area are finished. On the other hand, TwoAdvancedSamplersForMask performs sampling in both the base area and the mask area sequentially at each step.
* `KSamplerAdvancedProvider` - This is a wrapper that enables KSampler to be used in TwoAdvancedSamplersForMask, RegionalSampler.
  * sigma_factor: By multiplying the denoise schedule by the sigma_factor, you can adjust the amount of denoising based on the configured denoise.

* `TwoSamplersForMaskUpscalerProvider` - This is an Upscaler that extends TwoSamplersForMask to be used in Iterative Upscale.
  * TwoSamplersForMaskUpscalerProviderPipe - pipe version of TwoSamplersForMaskUpscalerProvider.


### Image Utils
  * `PreviewBridge (image)` - This custom node can be used with a bridge for image when using the MaskEditor feature of Clipspace.
  * `PreviewBridge (latent)` - This custom node can be used with a bridge for latent image when using the MaskEditor feature of Clipspace.
    * If a latent with a mask is provided as input, it displays the mask. Additionally, the mask output provides the mask set in the latent.
    * If a latent without a mask is provided as input, it outputs the original latent as is, but the mask output provides an output with the entire region set as a mask.
    * When set mask through MaskEditor, a mask is applied to the latent, and the output includes the stored mask. The same mask is also output as the mask output.
    * When connected to `vae_opt`, it takes higher priority than the `preview_method`.
  * `ImageSender`, `ImageReceiver` - The images generated in ImageSender are automatically sent to the ImageReceiver with the same link_id.
  * `LatentSender`, `LatentReceiver` - The latent generated in LatentSender are automatically sent to the LatentReceiver with the same link_id.
    * Furthermore, LatentSender is implemented with PreviewLatent, which stores the latent in payload form within the image thumbnail.
    * Due to the current structure of ComfyUI, it is unable to distinguish between SDXL latent and SD1.5/SD2.1 latent. Therefore, it generates thumbnails by decoding them using the SD1.5 method.


### Switch nodes
  * `Switch (image,mask)`, `Switch (latent)`, `Switch (SEGS)` - Among multiple inputs, it selects the input designated by the selector and outputs it. The first input must be provided, while the others are optional. However, if the input specified by the selector is not connected, an error may occur.
  * `Switch (Any)` - This is a Switch node that takes an arbitrary number of inputs and produces a single output. Its type is determined when connected to any node, and connecting inputs increases the available slots for connections.
  * `Inversed Switch (Any)` - In contrast to `Switch (Any)`, it takes a single input and outputs one of many.
  * NOTE: See this [tutorial](https://github.com/ltdrdata/ComfyUI-extension-tutorials/blob/Main/ComfyUI-Impact-Pack/tutorial/switch.md) 


### [Wildcards](http://github.com/ltdrdata/ComfyUI-extension-tutorials/blob/Main/ComfyUI-Impact-Pack/tutorial/ImpactWildcard.md) nodes
  * These are nodes that supports syntax in the form of `__wildcard-name__` and dynamic prompt syntax like `{a|b|c}`.
  * Wildcard files can be used by placing `.txt` or `.yaml` files under either `ComfyUI-Impact-Pack/wildcards` or `ComfyUI-Impact-Pack/custom_wildcards` paths.
    * You can download and use [Wildcard YAML](https://civitai.com/models/138970/billions-of-wildcards-all-in-one) files in this format.
    * After the first execution, you can change the custom wildcards path in the `custom_wildcards` entry within the `ComfyUI-Impact-Pack/impact-pack.ini` file created.
  * `ImpactWildcardProcessor` - The text is generated by processing the wildcard in the Text. If the mode is set to "populate", a dynamic prompt is generated with each execution and the input is filled in the second textbox. If the mode is set to "fixed", the content of the second textbox remains unchanged.
    * When an image is generated with the "fixed" mode, the prompt used for that particular generation is stored in the metadata.
  * `ImpactWildcardEncode` - Similar to ImpactWildcardProcessor, this provides the loading functionality of LoRAs (e.g. `<lora:some_awesome_lora:0.7:1.2>`). Populated prompts are encoded using the clip after all the lora loading is done.
    * If the `Inspire Pack` is installed, you can use **Lora Block Weight** in the form of `LBW=lbw spec;`
    * `<lora:chunli:1.0:1.0:LBW=B11:0,0,0,0,0,0,0,0,0,0,A,0,0,0,0,0,0;A=0.;>`, `<lora:chunli:1.0:1.0:LBW=0,0,0,0,0,0,0,0,0,0,A,B,0,0,0,0,0;A=0.5;B=0.2;>`, `<lora:chunli:1.0:1.0:LBW=SD-MIDD;>`


### Regional Sampling
  * These nodes offer the capability to divide regions and perform partial sampling using a mask. Unlike TwoSamplersForMask, sampling for each region is applied during each step.
  * `RegionalPrompt` - This node combines a **mask** for specifying regions and the **sampler** to apply to each region to create `REGIONAL_PROMPTS`.
  * `CombineRegionalPrompts` - Combine multiple `REGIONAL_PROMPTS` to create a single `REGIONAL_PROMPTS`.
  * `RegionalSampler` - This node performs sampling using a base sampler and regional prompts. Sampling by the base sampler is executed at each step, while sampling for each region is performed through the sampler bound to each region.
    * overlap_factor - Specifies the amount of overlap for each region to blend well with the area outside the mask.
    * restore_latent - When sampling each region, restore the areas outside the mask to the base latent, preventing additional noise from being introduced outside the mask during region sampling.
  * `RegionalSamplerAdvanced` - This is the Advanced version of the RegionalSampler. You can control it using `step` instead of `denoise`.
    > NOTE: The `sde` sampler and `uni_pc` sampler introduce additional noise during each step of the sampling process. To mitigate this, when sampling each region, the `uni_pc` sampler applies additional `dpmpp_fast`, and the sde sampler applies the `dpmpp_2m` sampler as an additional measure.


### Impact KSampler
  * These samplers support basic_pipe and AYS/OSS/GITS scheduler
  * `KSampler (pipe)` - pipe version of KSampler
  * `KSampler (advanced/pipe)` - pipe version of KSamplerAdvacned
  * When converting the scheduler widget to input, refer to the `Impact Scheduler Adapter` node to resolve compatibility issues.
  * `GITSScheduler Func Provider` - provider scheduler function for GITSScheduler
  

### Batch/List Util
  * `Image Batch to Image List` - Convert Image batch to Image List
    - You can use images generated in a multi batch to handle them
  * `Image List to Image Batch` - Convert Image List to Image Batch 
  * `Make Image List` - Convert multiple images into a single image list
  * `Make Image Batch` - Convert multiple images into a single image batch
    - The input of images can be scaled up as needed
  * `Masks to Mask List`, `Mask List to Masks`, `Make Mask List`, `Make Mask Batch` - It has the same functionality as the nodes above, but uses mask as input instead of image.
  * `Flatten Mask Batch` - Flattens a Mask Batch into a single Mask. Normal operation is not guaranteed for non-binary masks. 
  * `Make List (Any)` - Create a list with arbitrary values.
  * `Select Nth Item (Any list)` - Selects the Nth item from a list. If the index is out of range, it returns the last item in the list. 


### Logics (experimental) 
  * These nodes are experimental nodes designed to implement the logic for loops and dynamic switching.
  * `ImpactCompare`, `ImpactConditionalBranch`, `ImpactConditionalBranchSelMode`, `ImpactInt`, `ImpactBoolean`, `ImpactValueSender`, `ImpactValueReceiver`, `ImpactImageInfo`, `ImpactMinMax`, `ImpactNeg`, `ImpactConditionalStopIteration`
  * `ImpactIsNotEmptySEGS` - This node returns `true` only if the input SEGS is not empty. 
  * `ImpactIfNone` - Returns `true` if any_input is None, and returns `false` if it is not None.
  * `Queue Trigger` - When this node is executed, it adds a new queue to assist with repetitive tasks. It will only execute if the signal's status changes.
  * `Queue Trigger (Countdown)` - Like the Queue Trigger, it adds a queue, but only adds it if it's greater than 1, and decrements the count by one each time it runs.
  * `Sleep` - Waits for the specified time (in seconds).
  * `Set Widget Value` - This node sets one of the optional inputs to the specified node's widget. An error may occur if the types do not match.
  * `Set Mute State` - This node changes the mute state of a specific node.
  * `Control Bridge` - This node modifies the state of the connected control nodes based on the `mode` and `behavior` . If there are nodes that require a change, the current execution is paused, the mute status is updated, and a new prompt queue is inserted. 
    * When the `mode` is `active`, it makes the connected control nodes active regardless of the behavior. 
    * When the `mode` is `Bypass/Mute`, it changes the state of the connected nodes based on whether the behavior is `Bypass` or `Mute`.
    * **Limitation**: Due to these characteristics, it does not function correctly when the batch count exceeds 1. Additionally, it does not guarantee proper operation when the seed is randomized or when the state of nodes is altered by actions such as `Queue Trigger`, `Set Widget Value`, `Set Mute`, before the Control Bridge.
    * When utilizing this node, please structure the workflow in such a way that `Queue Trigger`, `Set Widget Value`, `Set Mute State`, and similar actions are executed at the end of the workflow.
    * If you want to change the value of the seed at each iteration, please ensure that Set Widget Value is executed at the end of the workflow instead of using randomization.
      * It is not a problem if the seed changes due to randomization as long as it occurs after the Control Bridge section.
  * `Remote Boolean (on prompt)`, `Remote Int (on prompt)` - At the start of the prompt, this node forcibly sets the `widget_value` of `node_id`. It is disregarded if the target widget type is different.
  * You can find the `node_id` by checking through [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager) using the format `Badge: #ID Nickname`.
  * Experimental set of nodes for implementing loop functionality (tutorial to be prepared later / [example workflow](test/loop-test.json)).


### HuggingFace nodes
  * These nodes provide functionalities based on HuggingFace repository models.
  * The path where the HuggingFace model cache is stored can be changed through the `HF_HOME` environment variable.
  * `HF Transformers Classifier Provider` - This is a node that provides a classifier based on HuggingFace's transformers models.
    * The 'repo id' parameter should contain HuggingFace's repo id. When `preset_repo_id` is set to `Manual repo id`, use the manually entered repo id in `manual_repo_id`.
    * e.g. 'rizvandwiki/gender-classification-2' is a repository that provides a model for gender classification.
  * `SEGS Classify` - This node utilizes the `TRANSFORMERS_CLASSIFIER` loaded with 'HF Transformers Classifier Provider' to classify `SEGS`.
    * The 'expr' allows for forms like `label > number`, and in the case of `preset_expr` being `Manual expr`, it uses the expression entered in `manual_expr`.
    * For example, in the case of `male <= 0.4`, if the score of the `male` label in the classification result is less than or equal to 0.4, it is categorized as `filtered_SEGS`, otherwise, it is categorized as `remained_SEGS`.
      * For supported labels, please refer to the `config.json` of the respective HuggingFace repository.
    * `#Female` and `#Male` are symbols that group multiple labels such as `Female, women, woman, ...`, for convenience, rather than being single labels.


### Etc nodes
  * `Impact Scheduler Adapter` - With the addition of AYS to the scheduler of the Impact Pack and Inspire Pack, there is an issue of incompatibility when the existing scheduler widget is converted to input. The Impact Scheduler Adapter allows for an indirect connection to be possible.
  * `StringListToString` - Convert String List to String
  * `WildcardPromptFromString` - Create labeled wildcard for detailer from string. 
    * This node works well when used with MakeTileSEGS. [[Link](https://github.com/ltdrdata/ComfyUI-Impact-Pack/pull/536#discussion_r1586060779)]

  * `String Selector` - It selects and returns a portion of the string. When `multiline` mode is disabled, it simply returns the string of the line pointed to by the selector. When `multiline` mode is enabled, it divides the string based on lines that start with `#` and returns them. If the `select` value is larger than the number of items, it will start counting from the first line again and return accordingly.
  * `Combine Conditionings` - It takes multiple conditionings as input and combines them into a single conditioning.
  * `Concat Conditionings` - It takes multiple conditionings as input and concat them into a single conditioning.
  * `Negative Cond Placeholder` - Models like FLUX.1 do not use Negative Conditioning. This is a placeholder node for them. You can use FLUX.1 by replacing the Negative Conditioning used in Impact KSampler, KSampler (Inspire), and Detailer with this node.
  * `Execution Order Controller` - A helper node that can forcibly control the execution order of nodes.
    * Connect the output of the node that should be executed first to the signal, and make the input of the node that should be executed later pass through this node.
  * `List Bridge` - When passing the list output through this node, it collects and organizes the data before forwarding it, which ensures that the previous stage's sub-workflow has been completed.


## Feature
* `Interactive SAM Detector (Clipspace)` - When you right-click on a node that has 'MASK' and 'IMAGE' outputs, a context menu will open. From this menu, you can either open a dialog to create a SAM Mask using 'Open in SAM Detector', or copy the content (likely mask data) using 'Copy (Clipspace)' and generate a mask using 'Impact SAM Detector' from the clipspace menu, and then paste it using 'Paste (Clipspace)'.
* Providing a feature to detect errors that occur when mixing models and clips from checkpoints such as `SDXL Base`, `SDXL Refiner`, `SD1.x`, `SD2.x` during sample execution, and reporting appropriate errors.


## How To Install?

### Install via ComfyUI-Manager (Recommended)
* Search `ComfyUI Impact Pack` in ComfyUI-Manager and click `Install` button.

### Manual Install (Not Recommended)
1. `cd custom_nodes`
2. `git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack`
3. `cd ComfyUI-Impact-Pack`
4. `pip install -r requirements.txt`
    * **IMPORTANT**:
        * You must install it within the Python environment where ComfyUI is running.
        * For the portable version, use `<installed path>\python_embeded\python.exe -m pip` instead of `pip`. For a `venv`, activate the `venv` first and then use `pip`.
5. Restart ComfyUI

* NOTE1: If an error occurs during the installation process, please refer to [Troubleshooting Page](troubleshooting/TROUBLESHOOTING.md) for assistance. 
* NOTE2: You can use this colab notebook [colab notebook](https://colab.research.google.com/github/ltdrdata/ComfyUI-Impact-Pack/blob/Main/notebook/comfyui_colab_impact_pack.ipynb) to launch it. This notebook automatically downloads the impact pack to the custom_nodes directory, installs the tested dependencies, and runs it.
* NOTE3: If you create an empty file named `skip_download_model` in the `ComfyUI/custom_nodes/` directory, it will skip the model download step during the installation of the impact pack.


## Package Dependencies (If you need to manual setup.)

* pip install
   * segment-anything
   * scikit-image
   * piexif 
   * opencv-python
   * scipy
   * numpy<2
   * dill
   * matplotlib
   * (optional) onnxruntime
   * (deprecated) openmim      # for mim
   * (deprecated) pycocotools  # for mim
   
* linux packages (ubuntu)
  * libgl1-mesa-glx
  * libglib2.0-0


## Config example
* Once you run the Impact Pack for the first time, an `impact-pack.ini` file will be automatically generated in the Impact Pack directory. You can modify this configuration file to customize the default behavior.
  * `dependency_version` - don't touch this
  * `sam_editor_cpu` - use cpu for `SAM editor` instead of gpu
  * sam_editor_model: Specify the SAM model for the SAM editor.
    * You can download various SAM models using ComfyUI-Manager.
    * Path to SAM model: `ComfyUI/models/sams`
```
[default]
sam_editor_cpu = False
sam_editor_model = sam_vit_b_01ec64.pth
```


## Other Materials (auto-download when installing)

* ComfyUI/models/sams <= https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth


## Troubleshooting page
* [Troubleshooting Page](troubleshooting/TROUBLESHOOTING.md)


## How To Use (DDetailer feature)

#### 1. Basic auto face detection and refine exapmle.
![simple](https://github.com/ltdrdata/ComfyUI-extension-tutorials/raw/Main/ComfyUI-Impact-Pack/images/simple.png)
* The face that has been damaged due to low resolution is restored with high resolution by generating and synthesizing it, in order to restore the details.
* The FaceDetailer node is a combination of a Detector node for face detection and a Detailer node for image enhancement. See the [Advanced Tutorial](https://github.com/ltdrdata/ComfyUI-extension-tutorials/raw/Main/ComfyUI-Impact-Pack/tutorial/advanced.md) for a more detailed explanation.
* The MASK output of FaceDetailer provides a visualization of where the detected and enhanced areas are.

![simple-orig](https://github.com/ltdrdata/ComfyUI-extension-tutorials/raw/Main/ComfyUI-Impact-Pack/images/simple-original.png) ![simple-refined](https://github.com/ltdrdata/ComfyUI-extension-tutorials/raw/Main/ComfyUI-Impact-Pack/images/simple-refined.png)
* You can see that the face in the image on the left has increased detail as in the image on the right.

#### 2. 2Pass refine (restore a severely damaged face)
![2pass-workflow-example](https://github.com/ltdrdata/ComfyUI-extension-tutorials/raw/Main/ComfyUI-Impact-Pack/images/2pass-simple.png)
* Although two FaceDetailers can be attached together for a 2-pass configuration, various common inputs used in KSampler can be passed through DETAILER_PIPE, so FaceDetailerPipe can be used to configure easily.
* In 1pass, only rough outline recovery is required, so restore with a reasonable resolution and low options. However, if you increase the dilation at this time, not only the face but also the surrounding parts are included in the recovery range, so it is useful when you need to reshape the face other than the facial part.

![2pass-example-original](https://github.com/ltdrdata/ComfyUI-extension-tutorials/raw/Main/ComfyUI-Impact-Pack/images/2pass-original.png) ![2pass-example-middle](https://github.com/ltdrdata/ComfyUI-extension-tutorials/raw/Main/ComfyUI-Impact-Pack/images/2pass-1pass.png) ![2pass-example-result](https://github.com/ltdrdata/ComfyUI-extension-tutorials/raw/Main/ComfyUI-Impact-Pack/images/2pass-2pass.png)
* In the first stage, the severely damaged face is restored to some extent, and in the second stage, the details are restored

#### 3. Face Bbox(bounding box) + Person silhouette segmentation (prevent distortion of the background.)
![combination-workflow-example](https://github.com/ltdrdata/ComfyUI-extension-tutorials/raw/Main/ComfyUI-Impact-Pack/images/combination.jpg)
![combination-example-original](https://github.com/ltdrdata/ComfyUI-extension-tutorials/raw/Main/ComfyUI-Impact-Pack/images/combination-original.png) ![combination-example-refined](https://github.com/ltdrdata/ComfyUI-extension-tutorials/raw/Main/ComfyUI-Impact-Pack/images/combination-refined.png)

* Facial synthesis that emphasizes details is delicately aligned with the contours of the face, and it can be observed that it does not affect the image outside of the face.

* The BBoxDetectorForEach node is used to detect faces, and the SAMDetectorCombined node is used to find the segment related to the detected face. By using the Segs & Mask node with the two masks obtained in this way, an accurate mask that intersects based on segs can be generated. If this generated mask is input to the DetailerForEach node, only the target area can be created in high resolution from the image and then composited.

#### 4. Iterative Upscale
![upscale-workflow-example](https://github.com/ltdrdata/ComfyUI-extension-tutorials/raw/Main/ComfyUI-Impact-Pack/images/upscale-workflow.png)
 
* The IterativeUpscale node is a node that enlarges an image/latent by a scale_factor. In this process, the upscale is carried out progressively by dividing it into steps.
* IterativeUpscale takes an Upscaler as an input, similar to a plugin, and uses it during each iteration. PixelKSampleUpscalerProvider is an Upscaler that converts the latent representation to pixel space and applies ksampling.
  * The upscale_model_opt is an optional parameter that determines whether to use the upscale function of the model base if available. Using the upscale function of the model base can significantly reduce the number of iterative steps required. If an x2 upscaler is used, the image/latent is first upscaled by a factor of 2 and then downscaled to the target scale at each step before further processing is done.

* The following image is an image of 304x512 pixels and the same image scaled up to three times its original size using IterativeUpscale.

![combination-example-original](https://github.com/ltdrdata/ComfyUI-extension-tutorials/raw/Main/ComfyUI-Impact-Pack/images/upscale-original.png) ![combination-example-refined](https://github.com/ltdrdata/ComfyUI-extension-tutorials/raw/Main/ComfyUI-Impact-Pack/images/upscale-3x.png)


#### 5. Interactive SAM Detector (Clipspace)

* When you right-click on the node that outputs 'MASK' and 'IMAGE', a menu called "Open in SAM Detector" appears, as shown in the following picture. Clicking on the menu opens a dialog in SAM's functionality, allowing you to generate a segment mask.
![samdetector-menu](https://github.com/ltdrdata/ComfyUI-extension-tutorials/raw/Main/ComfyUI-Impact-Pack/images/SAMDetector-menu.png)

* By clicking the left mouse button on a coordinate, a positive prompt in blue color is entered, indicating the area that should be included. Clicking the right mouse button on a coordinate enters a negative prompt in red color, indicating the area that should be excluded. Positive prompts represent the areas that should be included, while negative prompts represent the areas that should be excluded.
* You can remove the points that were added by using the "undo" button. After selecting the points, pressing the "detect" button generates the mask. Additionally, you can adjust the fidelity slider to determine the extent to which the mask belongs to the confidence region.

![samdetector-dialog](https://github.com/ltdrdata/ComfyUI-extension-tutorials/raw/Main/ComfyUI-Impact-Pack/images/SAMDetector-dialog.jpg)

* If you opened the dialog through "Open in SAM Detector" from the node, you can directly apply the changes by clicking the "Save to node" button. However, if you opened the dialog through the "clipspace" menu, you can save it to clipspace by clicking the "Save" button.

![samdetector-result](https://github.com/ltdrdata/ComfyUI-extension-tutorials/raw/Main/ComfyUI-Impact-Pack/images/SAMDetector-result.jpg)

* When you execute using the reflected mask in the node, you can observe that the image and mask are displayed separately.


## Others Tutorials
* [ComfyUI-extension-tutorials/ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-extension-tutorials/tree/Main/ComfyUI-Impact-Pack) - You can find various tutorials and workflows on this page.
* [Advanced Tutorial](https://github.com/ltdrdata/ComfyUI-extension-tutorials/blob/Main/ComfyUI-Impact-Pack/tutorial/advanced.md)
* [SAM Application](https://github.com/ltdrdata/ComfyUI-extension-tutorials/blob/Main/ComfyUI-Impact-Pack/tutorial/sam.md)
* [PreviewBridge](https://github.com/ltdrdata/ComfyUI-extension-tutorials/blob/Main/ComfyUI-Impact-Pack/tutorial/previewbridge.md)
* [Mask Pointer](https://github.com/ltdrdata/ComfyUI-extension-tutorials/blob/Main/ComfyUI-Impact-Pack/tutorial/maskpointer.md)
* [ONNX Tutorial](https://github.com/ltdrdata/ComfyUI-extension-tutorials/blob/Main/ComfyUI-Impact-Pack/tutorial/ONNX.md)
* [CLIPSeg Tutorial](https://github.com/ltdrdata/ComfyUI-extension-tutorials/blob/Main/ComfyUI-Impact-Pack/tutorial/clipseg.md)
* [Extreme Highresolution Upscale](https://github.com/ltdrdata/ComfyUI-extension-tutorials/blob/Main/ComfyUI-Impact-Pack/tutorial/extreme-upscale.md)
* [TwoSamplersForMask](https://github.com/ltdrdata/ComfyUI-extension-tutorials/blob/Main/ComfyUI-Impact-Pack/tutorial/TwoSamplers.md)
* [TwoAdvancedSamplersForMask](https://github.com/ltdrdata/ComfyUI-extension-tutorials/blob/Main/ComfyUI-Impact-Pack/tutorial/TwoAdvancedSamplers.md)
* [Advanced Iterative Upscale: PK_HOOK](https://github.com/ltdrdata/ComfyUI-extension-tutorials/blob/Main/ComfyUI-Impact-Pack/tutorial/pk_hook.md)
* [Advanced Iterative Upscale: TwoSamplersForMask Upscale Provider](https://github.com/ltdrdata/ComfyUI-extension-tutorials/blob/Main/ComfyUI-Impact-Pack/tutorial/TwoSamplersUpscale.md)
* [Interactive SAM + PreviewBridge](https://github.com/ltdrdata/ComfyUI-extension-tutorials/blob/Main/ComfyUI-Impact-Pack/tutorial/sam_with_preview_bridge.md)
* [ImageSender/ImageReceiver/LatentSender/LatentReceiver](https://github.com/ltdrdata/ComfyUI-extension-tutorials/blob/Main/ComfyUI-Impact-Pack/tutorial/sender_receiver.md)
* [ImpactWildcardProcessor](https://github.com/ltdrdata/ComfyUI-extension-tutorials/blob/Main/ComfyUI-Impact-Pack/tutorial/ImpactWildcardProcessor.md)


## Credits

ComfyUI/[ComfyUI](https://github.com/comfyanonymous/ComfyUI) - A powerful and modular stable diffusion GUI.

dustysys/[ddetailer](https://github.com/dustysys/ddetailer) - DDetailer for Stable-diffusion-webUI extension.

Bing-su/[dddetailer](https://github.com/Bing-su/dddetailer) - The anime-face-detector used in ddetailer has been updated to be compatible with mmdet 3.0.0, and we have also applied a patch to the pycocotools dependency for Windows environment in ddetailer.

facebook/[segment-anything](https://github.com/facebookresearch/segment-anything) - Segmentation Anything!

hysts/[anime-face-detector](https://github.com/hysts/anime-face-detector) - Creator of `anime-face_yolov3`, which has impressive performance on a variety of art styles.

open-mmlab/[mmdetection](https://github.com/open-mmlab/mmdetection) - Object detection toolset. `dd-person_mask2former` was trained via transfer learning using their [R-50 Mask2Former instance segmentation model](https://github.com/open-mmlab/mmdetection/tree/master/configs/mask2former#instance-segmentation) as a base.

biegert/[ComfyUI-CLIPSeg](https://github.com/biegert/ComfyUI-CLIPSeg) - This is a custom node that enables the use of CLIPSeg technology, which can find segments through prompts, in ComfyUI.

BlenderNeok/[ComfyUI-TiledKSampler](https://github.com/BlenderNeko/ComfyUI_TiledKSampler) - The tile sampler allows high-resolution sampling even in places with low GPU VRAM.

BlenderNeok/[ComfyUI_Noise](https://github.com/BlenderNeko/ComfyUI_Noise) - The noise injection feature relies on this function and slerp code for noise variation

WASasquatch/[was-node-suite-comfyui](https://github.com/WASasquatch/was-node-suite-comfyui) - A powerful custom node extensions of ComfyUI.

Trung0246/[ComfyUI-0246](https://github.com/Trung0246/ComfyUI-0246) - Nice bypass hack!

Layer-norm/[comfyui-lama-remover](https://github.com/Layer-norm/comfyui-lama-remover) - Required for using `LamaRemoverDetailerHook`.
