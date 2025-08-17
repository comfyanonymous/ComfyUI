# **WAS** Node Suite &nbsp; [![Colab](https://camo.githubusercontent.com/84f0493939e0c4de4e6dbe113251b4bfb5353e57134ffd9fcab6b8714514d4d1/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/github/WASasquatch/was-node-suite-comfyui/blob/main/ComfyUI_%2B_WAS_Node_Suite_and_ComfyUI_Manager.ipynb) [![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FWASasquatch%2Fwas-node-suite-comfyui&count_bg=%233D9CC8&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com) [![Donate](https://img.shields.io/badge/Donate-PayPal-blue.svg)](https://paypal.me/ThompsonJordan?country.x=US&locale.x=en_US)

<p align="center">
    <img src="https://user-images.githubusercontent.com/1151589/228982359-4a6215cc-3ca9-4c24-8a7b-d229d7bce277.png">
</p>

### A node suite for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) with many new nodes, such as image processing, text processing, and more.

#### [Share Workflows](https://github.com/WASasquatch/was-node-suite-comfyui/wiki/Workflow-Examples) to the workflows wiki. Preferably embedded PNGs with workflows, but JSON is OK too. [You can use this tool to add a workflow to a PNG file easily](https://colab.research.google.com/drive/1hQMjNUdhMQ3rw1Wcm3_umvmOMeS_K4s8?usp=sharing).
#### Consider [donating to the project](https://paypal.me/ThompsonJordan?country.x=US&locale.x=en_US) to help it's continued development.

# Important Updates

- **12/15/2023** WAS-NS is not under active development. I do not have the time and have other obligations. Feel free to fork and continue the project. I will approve appropriate and beneficial PRs.
- **[Updated 10/8/2023]** BLIP is now a shipped module of WAS-NS and no longer requires the BLIP Repo
 - **[Updated 5/29/2023]** `ASCII` **is deprecated**. The new preferred method of text node output is `STRING`. This is a change from `ASCII` so that it is more clear what data is being passed.
   - The `was_suite_config.json` will automatically set `use_legacy_ascii_text` to `false`.
 -  [Video Nodes](https://github.com/WASasquatch/was-node-suite-comfyui#video-nodes) - There are two new video nodes, `Write to Video` and `Create Video from Path`. These are experimental nodes.

# Current Nodes:

### There is documentation from [Salt AI](https://getsalt.ai/) available here: https://docs.getsalt.ai/md/was-node-suite-comfyui/

<details>
	<summary>$\Large\color{orange}{Expand\ Node\ List}$</summary>

<br/>

 - BLIP Model Loader: Load a BLIP model to input into the BLIP Analyze node
 - BLIP Analyze Image: Get a text caption from a image, or interrogate the image with a question.
   - Model will download automatically from default URL, but you can point the download to another location/caption model in `was_suite_config`
   - Models will be stored in `ComfyUI/models/blip/checkpoints/`
 - SAM Model Loader: Load a SAM Segmentation model
 - SAM Parameters: Define your SAM parameters for segmentation of a image
 - SAM Parameters Combine: Combine SAM parameters
 - SAM Image Mask: SAM image masking
 - Image Bounds: Bounds a image
 - Inset Image Bounds: Inset a image bounds
 - Bounded Image Blend: Blend bounds image
 - Bounded Image Blend with Mask: Blend a bounds image by mask
 - Bounded Image Crop: Crop a bounds image
 - Bounded Image Crop with Mask: Crop a bounds image by mask
 - Bus Node: condense the 5 common connectors into one, keep your workspace tidy (Model, Clip, VAE, Positive Conditioning, Negative Conditioning)
 - Cache Node: Cache Latnet, Tensor Batches (Image), and Conditioning to disk to use later.
 - CLIPTextEncode (NSP): Parse noodle soups from the NSP pantry, or parse wildcards from a directory containing A1111 style wildacrds.
   - Wildcards are in the style of `__filename__`, which also includes subdirectories like `__appearance/haircolour__` (if you noodle_key is set to `__`)
   - You can set a custom wildcards path in `was_suite_config.json` file with key:
     - `    "wildcards_path": "E:\\python\\automatic\\webui3\\stable-diffusion-webui\\extensions\\sd-dynamic-prompts\\wildcards"`
     - If no path is set the wildcards dir is located at the root of WAS Node Suite as `/wildcards`
 - CLIP Input Switch: Switch between two CLIP inputs based on a boolean switch.
 - CLIP Vision Input Switch: Switch between two CLIP Vision inputs based on a boolean switch.
 - Conditioning Input Switch: Switch between two conditioning inputs.
 - Constant Number
 - Control Net Model Input Switch: Switch between two Control Net Model inputs based on a boolean switch.
 - Create Grid Image: Create a image grid from images at a destination with customizable glob pattern. Optional border size and color.
 - Create Grid Image from Batch: Create a grid image from a batch tensor of images.
 - Create Morph Image: Create a GIF/APNG animation from two images, fading between them.
 - Create Morph Image by Path: Create a GIF/APNG animation from a path to a directory containing images, with optional pattern.
 - Create Video from Path: Create video from images from a specified path.
 - CLIPSeg Masking: Mask a image with CLIPSeg and return a raw mask
 - CLIPSeg Masking Batch: Create a batch image (from image inputs) and batch mask with CLIPSeg
 - Dictionary to Console: Print a dictionary input to the console
 - Image Analyze
   - Black White Levels
   - RGB Levels
     - Depends on `matplotlib`, will attempt to install on first run
 - Diffusers Hub Down-Loader: Download a diffusers model from the HuggingFace Hub and load it
 - Image SSAO (Ambient Occlusion): [Expiremental Beta Node] Create Screen Space Ambient Occlusion with a image and MiDaS depth approximation (or provided depth map).
 - Image SSDO (Direct Occlusion): [Expiremental Beta Node] Create a Screen Space Direct Occlusion with a image input. Direct Occlusion presents you with direct lighting highliths, similar to how Ambient Occlusion finds the crevices and shadowy areas around objets.
 - Image Aspect Ratio: Fetch image aspect ratio in float format, common format (eg 16:9), and in if the image is portrait, landscape, or square.
 - Image Batch: Create one batch out of multiple batched tensors.
 - Image Blank: Create a blank image in any color
 - Image Blend by Mask: Blend two images by a mask
 - Image Blend: Blend two images by opacity
 - Image Blending Mode: Blend two images by various blending modes
 - Image Bloom Filter: Apply a high-pass based bloom filter
 - Image Canny Filter: Apply a canny filter to a image
 - Image Chromatic Aberration: Apply chromatic aberration lens effect to a image like in sci-fi films, movie theaters, and video games
 - Image Color Palette
   - Generate a color palette based on the input image.
     - Depends on `scikit-learn`, will attempt to install on first run.
   - Supports color range of 8-256
   - Utilizes font in `./res/` unless unavailable, then it will utilize internal better then nothing font.
 - Image Crop Face: Crop a face out of a image
   - **Limitations:**
     - Sometimes no faces are found in badly generated images, or faces at angles
	 - Sometimes face crop is black, this is because the padding is too large and intersected with the image edge. Use a smaller padding size.
	 - face_recognition mode sometimes finds random things as faces. It also requires a [CUDA] GPU.
	 - Only detects one face. This is a design choice to make it's use easy.
   - **Notes:**
     - Detection runs in succession. If nothing is found with the selected detection cascades, it will try the next available cascades file.
 - Image Crop Location: Crop a image to specified location in top, left, right, and bottom locations relating to the pixel dimensions of the image in X and Y coordinats.
 - Image Crop Square Location: Crop a location by X/Y center, creating a square crop around that point.
 - Image Displacement Warp: Warp a image by a displacement map image by a given amplitude.
 - Image Dragan Photography Filter: Apply a Andrzej Dragan photography style to a image
 - Image Edge Detection Filter: Detect edges in a image
 - Image Film Grain: Apply film grain to a image
 - Image Filter Adjustments: Apply various image adjustments to a image
 - Image Flip: Flip a image horizontal, or vertical
 - Image Gradient Map: Apply a gradient map to a image
 - Image Generate Gradient: Generate a gradient map with desired stops and colors
 - Image High Pass Filter: Apply a high frequency pass to the image returning the details
 - Image History Loader: Load images from history based on the Load Image Batch node. Can define max history in config file. *(requires restart to show last sessions files at this time)*
 - Image Input Switch: Switch between two image inputs based on a boolean switch
 - Image Levels Adjustment: Adjust the levels of a image
 - Image Load: Load a *image* from any path on the system, or a url starting with `http`
 - Image Median Filter: Apply a median filter to a image, such as to smooth out details in surfaces
 - Image Mix RGB Channels: Mix together RGB channels into a single iamge
 - Image Monitor Effects Filter: Apply various monitor effects to a image
   - Digital Distortion
     - A digital breakup distortion effect
   - Signal Distortion
     - A analog signal distortion effect on vertical bands like a CRT monitor
   - TV Distortion
     - A TV scanline and bleed distortion effect
 - Image Nova Filter: A image that uses a sinus frequency to break apart a image into RGB frequencies
 - Image Perlin Noise: Generate perlin noise
 - Image Perlin Power Fractal: Generate a perlin power fractal
 - Image Paste Face Crop: Paste face crop back on a image at it's original location and size
   - Features a better blending funciton than GFPGAN/CodeFormer so there shouldn't be visible seams, and coupled with Diffusion Result, looks better than GFPGAN/CodeFormer.
 - Image Paste Crop: Paste a crop (such as from Image Crop Location) at it's original location and size utilizing the `crop_data` node input. This uses a different blending algorithm then Image Paste Face Crop, which may be desired in certain instances.
 - Image Power Noise: Generate power-law noise
   - frequency: The frequency parameter controls the distribution of the noise across different frequencies. In the context of Fourier analysis, higher frequencies represent fine details or high-frequency components, while lower frequencies represent coarse details or low-frequency components. Adjusting the frequency parameter can result in different textures and levels of detail in the generated noise. The specific range and meaning of the frequency parameter may vary depending on the noise type.
   - attenuation: The attenuation parameter determines the strength or intensity of the noise. It controls how much the noise values deviate from the mean or central value. Higher values of attenuation lead to more significant variations and a stronger presence of noise, while lower values result in a smoother and less noticeable noise. The specific range and interpretation of the attenuation parameter may vary depending on the noise type.
   - noise_type: The tyoe of Power-Law noise to generate (white, grey, pink, green, blue)
 - Image Paste Crop by Location: Paste a crop top a custom location. This uses the same blending algorithm as Image Paste Crop.
 - Image Pixelate: Turn a image into pixel art! Define the max number of colors, the pixelation mode, the random state, and max iterations, and max those sprites shine.
 - Image Remove Background (Alpha): Remove the background from a image by threshold and tolerance.
 - Image Remove Color: Remove a color from a image and replace it with another
 - Image Resize
 - Image Rotate: Rotate an image
 - Image Rotate Hue: Rotate the hue of a image. A hue_shift of `0.0` would represent no change, and `1.0` would represent a full circle of the hue, and also exhibit no change.
 - Image Save: A save image node with format support and path support.
	- `show_history` will show previously saved images with the WAS Save Image node. ComfyUI unfortunately resizes displayed images to the same size however, so if images are in different sizes it will force them in a different size.
	- Doesn't display images saved outside `/ComfyUI/output/`
	- You can save as `webp` if you have webp available to you system. On windows you can get that support with this [precompiled libarary](https://storage.googleapis.com/downloads.webmproject.org/releases/webp/libwebp-1.3.0-windows-x64.zip) from the [webp project](https://developers.google.com/speed/webp/download). On linux you can run `apt-get install webp`.
 - Image Seamless Texture: Create a seamless texture out of a image with optional tiling
 - Image Select Channel: Select a single channel of an RGB image
 - Image Select Color: Return the select image only on a black canvas
 - Image Shadows and Highlights: Adjust the shadows and highlights of an image
 - Image Size to Number: Get the `width` and `height` of an input image to use with **Number** nodes.
 - Image Stitch: Stitch images together on different sides with optional feathering blending between them.
 - Image Style Filter: Style a image with Pilgram instragram-like filters
   - Depends on `pilgram` module
 - Image Threshold: Return the desired threshold range of a image
 - Image Tile: Split a image up into a image batch of tiles. Can be used with Tensor Batch to Image to select a individual tile from the batch.
 - Image Transpose
 - Image fDOF Filter: Apply a fake depth of field effect to an image
 - Image to Latent Mask: Convert a image into a latent mask
 - Image to Noise: Convert a image into noise, useful for init blending or init input to theme a diffusion.
 - Images to RGB: Convert a tensor image batch to RGB if they are RGBA or some other mode.
 - Image to Seed: Convert a image to a reproducible seed
 - Image Voronoi Noise Filter
   - A custom implementation of the worley voronoi noise diagram
 - Input Switch  (Disable until `*` wildcard fix)
 - KSampler (WAS): A sampler that accepts a seed as a node inputs
 - KSampler Cycle: A KSampler able to do HR pass loops, you can specify an upscale factor, and how many steps to achieve that factor. Accepts a upscale_model, as well as a 1x processor model. A secondary diffusion model can also be used.
 - Load Cache: Load cached Latent, Tensor Batch (image), and Conditioning files.
 - Load Text File
   - Now supports outputting a dictionary named after the file, or custom input.
   - The dictionary contains a list of all lines in the file.
 - Load Batch Images
   - Increment images in a folder, or fetch a single image out of a batch.
   - Will reset it's place if the path, or pattern is changed.
   - pattern is a glob that allows you to do things like `**/*` to get all files in the directory and subdirectory
     or things like `*.jpg` to select only JPEG images in the directory specified.
 - Mask to Image: Convert `MASK` to `IMAGE`
 - Mask Batch to Mask: Return a single mask from a batch of masks
 - Mask Invert: Invert a mask.
 - Mask Add: Add masks together.
 - Mask Subtract: Subtract from a mask by another.
 - Mask Dominant Region: Return the dominant region in a mask (the largest area)
 - Mask Minority Region: Return the smallest region in a mask (the smallest area)
 - Mask Crop Dominant Region: Crop mask to the dominant region with optional padding in pixels
 - Mask Crop Minority Region: Crop mask to the minority region with optional padding in pixels
 - Mask Crop Region: Crop to dominant or minority region and return `crop_data` for pasting back. Additionally outputs region location and size for other nodes like Crop Image Location.
 - Mask Arbitrary Region: Return a region that most closely matches the size input (size is not a direct representation of pixels, but approximate)
 - Mask Smooth Region: Smooth the boundaries of a mask
 - Mask Erode Region: Erode the boundaries of a mask
 - Mask Dilate Region: Dilate the boundaries of a mask
 - Mask Fill Region: Fill holes within the masks regions
 - Mask Ceiling Region": Return only white pixels within a offset range.
 - Mask Floor Region: Return the lower most pixel values as white (255)
 - Mask Threshold Region: Apply a thresholded image between a black value and white value
 - Mask Gaussian Region: Apply a Gaussian blur to the mask
 - Mask Rect Area: Create a rectangular mask defined by percentages.
 - Mask Rect Area (Advanced): Create a rectangular mask defined by pixels and image size.
 - Masks Combine Masks: Combine 2 or more masks into one mask.
 - Masks Combine Batch: Combine batched masks into one mask.
 - Model Input Switch: Switch between two model inputs based on a boolean switch
 - ComfyUI Loaders: A set of ComfyUI loaders that also output a string that contains the name of the model being loaded.
 - Latent Noise Injection: Inject latent noise into a latent image
 - Latent Size to Number: Latent sizes in tensor width/height
 - Latent Upscale by Factor: Upscale a latent image by a factor
 - Latent Input Switch: Switch between two latent inputs based on a boolean switch
 - Logic Boolean: A simple `1` or `0` output to use with logic
 - Logic Boolean Primitive: True/False boolean input, to use with native boolean nodes
 - Logic AND: Given 2 booleans, performs "AND"
 - Logic OR: Given 2 booleans, performs "OR"
 - Logic XOR: Given 2 booleans, performs "!="
 - Logic NOT: Given 1 boolean, returns the opposite
 - Lora Input Switch: Switch between two LORAs based on a boolean switch
 - MiDaS Model Loader: Load a MiDaS model as an optional input for MiDaS Depth Approximation
 - MiDaS Depth Approximation: Produce a depth approximation of a single image input
 - MiDaS Mask Image: Mask a input image using MiDaS with a desired color
 - Number Operation
 - Number to Seed
 - Number to Float
 - Number Input Switch: Switch between two number inputs based on a boolean switch
 - Number Input Condition: Compare between two inputs or against the A input
 - Number to Int
 - Number to String
 - Number to Text
 - Boolean to Text
 - Perlin Power Fractal Latent: Create a power fractal based latent image. Doesn't work with all samplers (unless you add noise).
 - Random Number
   - Random integer between min and max (inclusive), uniformly distributed random number
   - Random float between min and max (inclusive), uniformly distributed random number
   - Random number from 0 to 1 inclusive, this will be a 0 or 1 boolean if you use the 'int' output
   - Random shuffled list of integers between min and max inclusive.  E.g. if min=0 and max=3, a possible outcome would be the string '3,1,2,0'
 - Save Text File: Save a text string to a file
 - Samples Passthrough (Stat System): Logs RAM, VRAM, and Disk usage to the console.
 - Seed: Return a seed
 - Tensor Batch to Image: Select a single image out of a latent batch for post processing with filters
 - Text Add Tokens: Add custom tokens to parse in filenames or other text.
 - Text Add Token by Input: Add custom token by inputs representing single **single line** name and value of the token
 - Text Compare: Compare two strings. Returns a boolean if they are the same, a score of similarity, and the similarity or difference text.
 - Text Concatenate: Merge two strings
 - Text Dictionary Update: Merge two dictionaries
 - Text Dictionary Get: Get a value from a dictionary (as a string)
 - Text Dictionary Convert: Convert text to dictionary object
 - Text Dictionary New: Create a new dictionary
 - Text Dictionary Keys: Returns the keys, as a list from a dictionary object
 - Text Dictionary To Text: Returns the dictionary as text
 - Text File History: Show previously opened text files *(requires restart to show last sessions files at this time)*
 - Text Find: Find a substring or pattern within another string. Returns boolean
 - Text Find and Replace: Find and replace a substring in a string
 - Text Find and Replace by Dictionary: Replace substrings in a ASCII text input with a dictionary.
   - The dictionary keys are used as the key to replace, and the list of lines it contains chosen at random based on the seed.
 - Text Input Switch: Switch between two text inputs
 - Text List: Create a list of text strings
 - Text Load Line From File: Load lines from a file sequentially each *batch prompt* run, or select a line index.
 - Text Concatenate: Merge lists of strings
 - Text Contains: Checks if substring is in another string (case insensitive optional)
 - Text Multiline: Write a multiline text string
 - Text Parse A1111 Embeddings: Convert embeddings filenames in your prompts to `embedding:[filename]]` format based on your `/ComfyUI/models/embeddings/` files.
 - Text Parse Noodle Soup Prompts: Parse NSP in a text input
 - Text Parse Tokens: Parse custom tokens in text.
 - Text Random Line: Select a random line from a text input string
 - Text Random Prompt: Feeling lucky? Get a random prompt based on a search seed, such as "superhero"
 - Text String: Write a single line text string value
 - Text String Truncate: Truncate a string from the beginning or end by characters or words.
 - Text to Conditioning: Convert a text string to conditioning.
 - True Random.org Number Generator: Generate a truly random number online from atmospheric noise with [Random.org](https://random.org/)
   - [Get your API key from your account page](https://accounts.random.org/)
 - Upscale Model Input Switch: Switch between two Upscale Models inputs based on a boolean switch.
 - Write to Morph GIF: Write a new frame to an existing GIF (or create new one) with interpolation between frames.
 - Write to Video: Write a frame as you generate to a video (Best used with FFV1 for lossless images)
 - VAE Input Switch: Switch between two VAE inputs based on boolean input
</details>


 <br>

 ### Extra Nodes

  - CLIPTextEncode (BlenderNeko Advanced + NSP): Only available if you have BlenderNeko's [Advanced CLIP Text Encode](https://github.com/BlenderNeko/ComfyUI_ADV_CLIP_emb). Allows for NSP and Wildcard use with their advanced CLIPTextEncode.


 ### Notes:

  - **CLIPTextEncode (NSP)** and **CLIPTextEncode (BlenderNeko Advanced + NSP)**: Accept dynamic prompts in `<option1|option2|option3>` format. This will respect the nodes input seed to yield reproducible results like NSP and Wildcards.
  - **CLIPTextEncode (NSP)** and **CLIPTextEncode (BlenderNeko Advanced + NSP)**: Assign variables with `$|prompt words|$` format. You can then print this word again within the prompt with the number corresponding the order you created them. So the first prompt var would be printed with `$1` and the second with `$2` and so on.

---


## Video Nodes

### Codecs
You can use codecs that are available to your ffmpeg binaries by adding their fourcc ID (in one string), and appropriate container extension to the `was_suite_config.json`

Example [H264 Codecs](https://github.com/cisco/openh264/releases/tag/v1.8.0) (Defaults)
```
    "ffmpeg_extra_codecs": {
        "avc1": ".mp4",
        "h264": ".mkv"
    }
```

### Notes
  - For now I am only supporting **Windows** installations for video nodes.
    - I do not have access to Mac or a stand-alone linux distro. If you get them working and want to PR a patch/directions, feel free.
  - Video nodes require [FFMPEG](https://ffmpeg.org/download.html). You should download the proper FFMPEG binaries for you system and set the FFMPEG path in the config file.
  - Additionally, if you want to use H264 codec need to [download OpenH264 1.8.0](https://github.com/cisco/openh264/releases/tag/v1.8.0) and place it in the root of ComfyUI (Example: `C:\ComfyUI_windows_portable`).
  - FFV1 will complain about invalid container. You can ignore this. The resulting MKV file is readable. I have not figured out what this issue is about. Documentaion tells me to use MKV, but it's telling me it's unsupported.
    - If you know how to resolve this, I'd love a PR
  - `Write to Video` node should use a lossless video codec or when it copies frames, and reapplies compression, it will start expontentially ruining the starting frames run to run.

---

# Text Tokens
Text tokens can be used in the Save Text File and Save Image nodes. You can also add your own custom tokens with the Text Add Tokens node.

The token name can be anything excluding the `:` character to define your token. It can also be simple Regular Expressions.

## Built-in Tokens
  - [time]
    - The current system microtime
  - [time(`format_code`)]
    - The current system time in human readable format. Utilizing [datetime](https://docs.python.org/3/library/datetime.html) formatting
    - Example: `[hostname]_[time]__[time(%Y-%m-%d__%I-%M%p)]` would output: **SKYNET-MASTER_1680897261__2023-04-07__07-54PM**
  - [hostname]
    - The hostname of the system executing ComfyUI
  - [cuda_device]
    - The cuda device from `comfy.model_management.get_cuda_device()`
  - [cuda_name]
    - The cuda name from `comfy.model_management.get_cuda_device_name()`

<br>

<details>
	<summary>$\color{orange}{Expand\ Date\ Code\ List}$</summary>

<br>

| Directive | Meaning | Example | Notes |
| --- | --- | --- | --- |
| %a | Weekday as locale’s abbreviated name. |  Sun, Mon, …, Sat (en_US); So, Mo, …, Sa (de_DE)   | (1) |
| %A | Weekday as locale’s full name. |  Sunday, Monday, …, Saturday (en_US); Sonntag, Montag, …, Samstag (de_DE)   | (1) |
| %w | Weekday as a decimal number, where 0 is Sunday and 6 is Saturday. | 0, 1, …, 6 |  |
| %d | Day of the month as a zero-padded decimal number. | 01, 02, …, 31 | (9) |
| %b | Month as locale’s abbreviated name. |  Jan, Feb, …, Dec (en_US); Jan, Feb, …, Dez (de_DE)   | (1) |
| %B | Month as locale’s full name. |  January, February, …, December (en_US); Januar, Februar, …, Dezember (de_DE)   | (1) |
| %m | Month as a zero-padded decimal number. | 01, 02, …, 12 | (9) |
| %y | Year without century as a zero-padded decimal number. | 00, 01, …, 99 | (9) |
| %Y | Year with century as a decimal number. | 0001, 0002, …, 2013, 2014, …, 9998, 9999 | (2) |
| %H | Hour (24-hour clock) as a zero-padded decimal number. | 00, 01, …, 23 | (9) |
| %I | Hour (12-hour clock) as a zero-padded decimal number. | 01, 02, …, 12 | (9) |
| %p | Locale’s equivalent of either AM or PM. |  AM, PM (en_US); am, pm (de_DE)   | (1), (3) |
| %M | Minute as a zero-padded decimal number. | 00, 01, …, 59 | (9) |
| %S | Second as a zero-padded decimal number. | 00, 01, …, 59 | (4), (9) |
| %f | Microsecond as a decimal number, zero-padded to 6 digits. | 000000, 000001, …, 999999 | (5) |
| %z | UTC offset in the form ±HHMM[SS[.ffffff]] (empty string if the object is naive). | (empty), +0000, -0400, +1030, +063415, -030712.345216 | (6) |
| %Z | Time zone name (empty string if the object is naive). | (empty), UTC, GMT | (6) |
| %j | Day of the year as a zero-padded decimal number. | 001, 002, …, 366 | (9) |
| %U | Week number of the year (Sunday as the first day of the week) as a zero-padded decimal number. All days in a new year preceding the first Sunday are considered to be in week 0. | 00, 01, …, 53 | (7), (9) |
| %W | Week number of the year (Monday as the first day of the week) as a zero-padded decimal number. All days in a new year preceding the first Monday are considered to be in week 0. | 00, 01, …, 53 | (7), (9) |
| %c | Locale’s appropriate date and time representation. |  Tue Aug 16 21:30:00 1988 (en_US); Di 16 Aug 21:30:00 1988 (de_DE)   | (1) |
| %x | Locale’s appropriate date representation. |  08/16/88 (None); 08/16/1988 (en_US); 16.08.1988 (de_DE)   | (1) |
| %X | Locale’s appropriate time representation. |  21:30:00 (en_US); 21:30:00 (de_DE)   | (1) |
| %% | A literal '%' character. | % |  |

</details>

<br>

---

# Other Features

### Import AUTOMATIC1111 WebUI Styles
When using the latest builds of WAS Node Suite a `was_suite_config.json` file will be generated (if it doesn't exist). In this file you can setup a A1111 styles import.

  - Run ComfyUI to generate the new `/custom-nodes/was-node-suite-comfyui/was_Suite_config.json` file.
  - Open the `was_suite_config.json` file with a text editor.
  - Replace the `webui_styles` value from `None` to the path of your A1111 styles file called **styles.csv**. Be sure to use double backslashes for Windows paths.
    - Example `C:\\python\\stable-diffusion-webui\\styles.csv`
  - Restart ComfyUI
  - Select a style with the `Prompt Styles Node`.
    - The first ASCII output is your positive prompt, and the second ASCII output is your negative prompt.

You can set `webui_styles_persistent_update` to `true` to update the WAS Node Suite styles from WebUI every start of ComfyUI

# Recommended Installation:
If you're running on Linux, or non-admin account on windows you'll want to ensure `/ComfyUI/custom_nodes`, `was-node-suite-comfyui`, and `WAS_Node_Suite.py` has write permissions.

There is now a **install.bat** you can run to install to portable if detected. Otherwise it will default to system and assume you followed ConfyUI's manual installation steps.

  - Navigate to your `/ComfyUI/custom_nodes/` folder
  - Run `git clone https://github.com/WASasquatch/was-node-suite-comfyui/`
  - Navigate to your `was-node-suite-comfyui` folder
    - Portable/venv:
       - Run `path/to/ComfUI/python_embeded/python.exe -s -m pip install -r requirements.txt`
	- With system python
	   - Run `pip install -r requirements.txt`
  - Start ComfyUI
    - WAS Suite should uninstall legacy nodes automatically for you.
    - Tools will be located in the WAS Suite menu.

## Alternate [Legacy] Installation:
If you're running on Linux, or non-admin account on windows you'll want to ensure `/ComfyUI/custom_nodes`, and `WAS_Node_Suite.py` has write permissions.

  - Download `WAS_Node_Suite.py`
  - Move the file to your `/ComfyUI/custom_nodes/` folder
  - WAS Node Suite will attempt install dependencies on it's own, but you may need to manually do so. The dependencies required are in the `requirements.txt` on this repo. See installation steps above.
    - If this process fails attempt the following:
      - Navigate to your `was-node-suite-comfyui` folder
      - Portable/venv:
        - Run `path/to/ComfUI/python_embeded/python.exe -s -m pip install -r requirements.txt`
      - With system python
	- Run `pip install -r requirements.txt`
  - Start, or Restart ComfyUI
    - WAS Suite should uninstall legacy nodes automatically for you.
    - Tools will be located in the WAS Suite menu.

This method will not install the resources required for Image Crop Face node, and you'll have to download the [./res/](https://github.com/WASasquatch/was-node-suite-comfyui/tree/main/res) folder yourself.

## Installing on Colab
Create a new cell and add the following code, then run the cell. You may need to edit the path to your `custom_nodes` folder. You can also use the [colab hosted here](https://colab.research.google.com/github/WASasquatch/comfyui-colab-was-node-suite/blob/main/ComfyUI_%2B_WAS_Node_Suite.ipynb)

  - `!git clone https://github.com/WASasquatch/was-node-suite-comfyui /content/ComfyUI/custom_nodes/was-node-suite-comfyui`
  - `!pip install -r /content/ComfyUI/custom_nodes/was-node-suite-comfyui/requirements.txt`
  - Restart Colab Runtime (don't disconnect)
    - Tools will be located in the WAS Suite menu.
