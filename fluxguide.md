<div align="center">

# Small Flux Guide

</div>

- under models\unet folder : your main flux model , it could be dev, schnell or any custom model based on those. I suggest : [https://civitai.com/models/686814/jib-mix-flux?modelVersionId=1193229](https://civitai.com/models/686814?modelVersionId=1249737)
  
- under models\clip folder : two necessary clip models, one small
(https://huggingface.co/zer0int/CLIP-GmP-ViT-L-14/blob/main/ViT-L-14-TEXT-detail-improved-hiT-GmP-TE-only-HF.safetensors)
and one big -this is the t5 model which is what seperates flux and sd3 and other newer models from previous ones such as sd 1.5, sdxl etc.-
(https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/t5xxl_fp8_e4m3fn_scaled.safetensors)
Keep in mind these are smaller versions of said models (in clip_l's case the one I shared is actually a optimised clip_l) , you can still use full t5 but this smaller versions let's use use less vram thus allowing the main model stay on vram so generate stuff faster.

- under models\vae folder : the default flux vae (https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/ae.safetensors)
  

- after you got all of the files downloaded and put on those directories, you need a basic flux workflow [HERE](basic-flux-workflow.json) Download it and load from comfy menu.
- load this workflow , change the unet , clips and vae if they are not set. (in our case they should be jibmixflux_v7... , t5xxl_fp8... , vit_l_14... , and vae ae.safetensors)
- queue and wait , the first image generation will take a while (10-15 min) , this is for zluda which we are using in the background to create a database for you gpu at the current configuration. After that you won't need to wait.
- if you successfully manage to get an image out of it , please use this workflow from now on, this uses more suggested nodes and various settings for flux. [HERE](better-flux-workflow.json) Download it and load from comfy menu.
