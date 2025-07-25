#    -----  链接模型  ------------

#wan t2v
ln -s /kaggle/input/wan-2-1-vae/wan_2.1_vae.safetensors ./models/vae/wan_2.1_vae.safetensors
ln -s /kaggle/input/umt5-xxl-fp8-e4m3fn-scaled/umt5_xxl_fp8_e4m3fn_scaled.safetensors ./models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors
ln -s /kaggle/input/wan2-1-t2v-1-3b-fp16/wan2.1_t2v_1.3B_fp16.safetensors ./models/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors

# wan vace
#wan2.1_vace_1.3B_fp16.safetensors, 目前仅支持480P的视频，不能支持720P的。
ln -s /kaggle/input/wan2-1-vace-1-3b-fp16/wan2.1_vace_1.3B_fp16.safetensors ./models/diffusion_models/wan2.1_vace_1.3B_fp16.safetensors
ln -s /kaggle/input/wan21-causvid-bidirect2-t2v-1-3b-lora-rank32/Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors ./models/loras/Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors
ln -s /kaggle/input/umt5-xxl-fp16/umt5_xxl_fp16.safetensors ./models/text_encoders/umt5_xxl_fp16.safetensors
#wan2.1_vace_14B_fp16.safetensors, T4 16G 爆了
# wget -c https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_vace_14B_fp16.safetensors -P ./models/diffusion_models
# ln -s /kaggle/input/wan21-causvid-14b-t2v-lora-rank32/Wan21_CausVid_14B_T2V_lora_rank32.safetensors ./models/loras/Wan21_CausVid_14B_T2V_lora_rank32.safetensors


#Flux 
ln -s /kaggle/input/flux-ae/ae.safetensors ./models/vae/ae.safetensors
ln -s /kaggle/input/clip-l/clip_l.safetensors ./models/text_encoders/clip_l.safetensors
ln -s /kaggle/input/t5xxl-fp8-e4m3fn-scaled/t5xxl_fp8_e4m3fn_scaled.safetensors ./models/text_encoders/t5xxl_fp8_e4m3fn_scaled.safetensors
wget -c https://huggingface.co/Comfy-Org/flux1-kontext-dev_ComfyUI/resolve/main/split_files/diffusion_models/flux1-dev-kontext_fp8_scaled.safetensors -P ./models/diffusion_models
ln -s /kaggle/input/flux1-dev-fp8/flux1-dev-fp8.safetensors ./models/checkpoints/flux1-dev-fp8.safetensors

#nunchaku
ln -s /kaggle/input/svdq-int4-r32-flux-1-kontext-dev/svdq-int4_r32-flux.1-kontext-dev.safetensors -P ./models/diffusion_models

#SD3
wget -c https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/resolve/main/sd3.5_large_fp8_scaled.safetensors -P /kaggle/working/
ln -s /kaggle/working/sd3.5_large_fp8_scaled.safetensors ./models/checkpoints/sd3.5_large_fp8_scaled.safetensors

#omnigen2
ln -s /kaggle/input/qwen-2-5-vl-fp16/qwen_2.5_vl_fp16.safetensors ./models/text_encoders/qwen_2.5_vl_fp16.safetensors
ln -s /kaggle/input/omnigen2-fp16/omnigen2_fp16.safetensors ./models/diffusion_models/omnigen2_fp16.safetensors

# wget -c https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/blob/main/split_files/diffusion_models/wan2.1_i2v_480p_14B_fp16.safetensors -P ./models/diffusion_models
ln -s /kaggle/input/clip-vision-h/clip_vision_h.safetensors ./models/clip_vision/clip_vision_h.safetensors




# ----------------   安装自定义插件节点  ----------------

# 1 ComfyUI-Manager
cd custom_nodes
git clone https://github.com/ltdrdata/ComfyUI-Manager comfyui-manager
cd /kaggle/ComfyUI

# 2 安装 nunchaku
pip install facexlib
pip install onnxruntime
pip install insightface
pip install https://huggingface.co/mit-han-lab/nunchaku/resolve/main/nunchaku-0.3.1+torch2.6-cp311-cp311-linux_x86_64.whl
# wget -c https://huggingface.co/mit-han-lab/nunchaku-flux.1-kontext-dev/resolve/main/svdq-int4_r32-flux.1-kontext-dev.safetensors -P ./models/diffusion_models
cd /kaggle/ComfyUI

cd custom_nodes
git clone https://github.com/mit-han-lab/ComfyUI-nunchaku nunchaku_nodes
cd /kaggle/ComfyUI

# 3 ComfyUI-GGUF
# wget -c https://huggingface.co/city96/FLUX.1-dev-gguf/resolve/main/flux1-dev-Q3_K_S.gguf -P ./models/unet
# cd custom_nodes
# git clone https://github.com/city96/ComfyUI-GGUF
# cd ComfyUI-GGUF
# pip install --upgrade gguf

# 4 encrypt image
# cd custom_nodes
# git clone https://github.com/Vander-Bilt/comfyui-encrypt-image.git
# cd /kaggle/ComfyUI

#5 Prompts Generator
cd custom_nodes
git clone https://github.com/fairy-root/Flux-Prompt-Generator.git
cd /kaggle/ComfyUI


# 6 Custom-Scripts
cd custom_nodes
git clone https://github.com/Vander-Bilt/ComfyUI-Custom-Scripts.git
cd /kaggle/ComfyUI

# 7 save2hf
cd custom_nodes
git clone https://github.com/Vander-Bilt/save2hf.git
cd save2hf
pip install -r requirements.txt
cd /kaggle/ComfyUI

#class MyHTMLNode:
cd custom_nodes
git clone https://github.com/Vander-Bilt/MyHTMLNode.git
cd /kaggle/ComfyUI

