#    -----  链接模型  ------------

# cosmos ？
ln -s /kaggle/input/cosmos-predict2-2b-video2world-480p-16fps/cosmos_predict2_2B_video2world_480p_16fps.safetensors ./models/diffusion_models/cosmos_predict2_2B_video2world_480p_16fps.safetensors

# sd lora ？
ln -s /kaggle/input/moxinv1/MoXinV1.safetensors ./models/loras/MoXinV1.safetensors
ln -s /kaggle/input/blindbox-v1-mix/blindbox_v1_mix.safetensors ./models/loras/blindbox_v1_mix.safetensors
ln -s /kaggle/input/dreamshaper-8/dreamshaper_8.safetensors ./models/checkpoints/dreamshaper_8.safetensors

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

# wan lora: dabaichui
wget -c https://huggingface.co/Heng365/dabaichui/resolve/main/dabaichui.safetensors -P ./models/loras

wget -c https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl.vae.safetensors -P ./models/vae

#Flux kontext
ln -s /kaggle/input/flux-ae/flux-ae.safetensors ./models/vae/ae.safetensors
ln -s /kaggle/input/clip-l/clip_l.safetensors ./models/text_encoders/clip_l.safetensors
ln -s /kaggle/input/t5xxl-fp8-e4m3fn-scaled/t5xxl_fp8_e4m3fn_scaled.safetensors ./models/text_encoders/t5xxl_fp8_e4m3fn_scaled.safetensors
# wget -c https://huggingface.co/Comfy-Org/flux1-kontext-dev_ComfyUI/resolve/main/split_files/diffusion_models/flux1-dev-kontext_fp8_scaled.safetensors -P ./models/diffusion_models

#Flux dev
ln -s /kaggle/input/flux1-dev-fp8/flux1-dev-fp8.safetensors ./models/checkpoints/flux1-dev-fp8.safetensors

# kontext turnaround sheet lora
wget -c https://huggingface.co/reverentelusarca/kontext-turnaround-sheet-lora-v1/resolve/main/kontext-turnaround-sheet-v1.safetensors -P ./models/loras

#nunchaku
ln -s /kaggle/input/svdq-int4-r32-flux-1-kontext-dev/svdq-int4_r32-flux.1-kontext-dev.safetensors -P ./models/diffusion_models

#SD3
# wget -c https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/resolve/main/sd3.5_large_fp8_scaled.safetensors -P /kaggle/working/
# ln -s /kaggle/working/sd3.5_large_fp8_scaled.safetensors ./models/checkpoints/sd3.5_large_fp8_scaled.safetensors

#iniverseMixSFWNSFW_ponyRealGuofengV51
wget -c https://huggingface.co/datasets/Heng365/mydataset/resolve/main/iniverseMixSFWNSFW_ponyRealGuofengV51.safetensors -P ./models/checkpoints

#omnigen2
# ln -s /kaggle/input/qwen-2-5-vl-fp16/qwen_2.5_vl_fp16.safetensors ./models/text_encoders/qwen_2.5_vl_fp16.safetensors
# ln -s /kaggle/input/omnigen2-fp16/omnigen2_fp16.safetensors ./models/diffusion_models/omnigen2_fp16.safetensors

# wan2.1 i2v
# wget -c https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_i2v_480p_14B_fp8_scaled.safetensors -P ./models/diffusion_models
# wget -c https://huggingface.co/UmeAiRT/ComfyUI-Auto_installer/resolve/main/models/unet/WAN/Wan2.1-VACE-14B-Q5_K_S.gguf -P ./models/diffusion_models

# wget -c https://huggingface.co/UmeAiRT/ComfyUI-Auto_installer/resolve/main/models/unet/WAN/Wan2.1-VACE-14B-Q4_K_S.gguf -P ./models/diffusion_models
wget -c https://huggingface.co/schwgHao/RealESRGAN_x4plus/resolve/main/RealESRGAN_x4plus.pth -P ./models/upscale_models

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
# cd custom_nodes
# git clone https://github.com/city96/ComfyUI-GGUF
# cd ComfyUI-GGUF
# pip install --upgrade gguf
# cd /kaggle/ComfyUI

# 4 encrypt image
cd custom_nodes
git clone https://github.com/Vander-Bilt/comfyui-encrypt-image.git
cd /kaggle/ComfyUI

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
# cd custom_nodes
# git clone https://github.com/Vander-Bilt/MyHTMLNode.git
# cd /kaggle/ComfyUI


# These custom nodes used by workflow VACE ControlNet 1.0 (base).json ------------ start ------------

cd custom_nodes
git clone https://github.com/kijai/ComfyUI-KJNodes.git
cd ComfyUI-KJNodes
pip install -r requirements.txt
cd /kaggle/ComfyUI

cd custom_nodes
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
cd ComfyUI-VideoHelperSuite
pip install -r requirements.txt
cd /kaggle/ComfyUI


cd custom_nodes
git clone https://github.com/Smirnov75/ComfyUI-mxToolkit.git
cd /kaggle/ComfyUI

cd custom_nodes
git clone https://github.com/facok/ComfyUI-HunyuanVideoMultiLora.git
cd /kaggle/ComfyUI


cd custom_nodes
git clone https://github.com/rgthree/rgthree-comfy.git
cd /kaggle/ComfyUI

cd custom_nodes
git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git
cd ComfyUI-Frame-Interpolation
python install.py
cd /kaggle/ComfyUI


cd custom_nodes
git clone https://github.com/WASasquatch/was-node-suite-comfyui.git
cd was-node-suite-comfyui
pip install -r requirements.txt
cd /kaggle/ComfyUI

cd custom_nodes
git clone https://github.com/kijai/ComfyUI-Florence2.git
cd ComfyUI-Florence2
pip install -r requirements.txt
cd /kaggle/ComfyUI


cd custom_nodes
git clone https://github.com/yuvraj108c/ComfyUI-Upscaler-Tensorrt.git
cd ComfyUI-Upscaler-Tensorrt
pip install -r requirements.txt
cd /kaggle/ComfyUI

cd custom_nodes
git clone https://github.com/pollockjj/ComfyUI-MultiGPU.git
cd /kaggle/ComfyUI


cd custom_nodes
git clone https://github.com/chflame163/ComfyUI_LayerStyle.git
cd ComfyUI_LayerStyle
pip install -r requirements.txt
cd /kaggle/ComfyUI


cd custom_nodes
git clone https://github.com/yolain/ComfyUI-Easy-Use
cd ComfyUI-Easy-Use
pip install -r requirements.txt
cd /kaggle/ComfyUI

cd custom_nodes
git clone https://github.com/Fannovel16/comfyui_controlnet_aux/
cd comfyui_controlnet_aux
pip install -r requirements.txt
cd /kaggle/ComfyUI

# These custom nodes used by workflow VACE ControlNet 1.0 (base).json ------------ end ------------


# MV-Adapter
cd custom_nodes
git clone https://github.com/huanngzh/ComfyUI-MVAdapter.git
cd ComfyUI-MVAdapter
pip install -r requirements.txt
cd /kaggle/ComfyUI



cd custom_nodes
git clone https://github.com/PowerHouseMan/ComfyUI-AdvancedLivePortrait.git
cd ComfyUI-AdvancedLivePortrait
pip install -r requirements.txt
cd /kaggle/ComfyUI

# REMOTE_PORT="$1"

# 传入参数3,4,5,...  这样便于扩展，如果以后用了其他的frp，也好调整。

# frp execution binary only. 这里放frpc执行文件，如果版本有变也好改。配置文件模版，也在项目中, 模版文件名为template_frpc
wget -O  /kaggle/working/frp_0.54.0_linux_amd64.tar.gz https://github.com/fatedier/frp/releases/download/v0.54.0/frp_0.54.0_linux_amd64.tar.gz
tar -xzvf /kaggle/working/frp_0.54.0_linux_amd64.tar.gz -C /kaggle/working
cp -p /kaggle/working/frp_0.54.0_linux_amd64/frpc /kaggle/working/frpc
cp -p /kaggle/ComfyUI/template_frpc /kaggle/working/frpc.toml

# 1, 2 主要是为了兼容之前的comfyUI notebook（不想一个一个的去修改了）
# FRP_CONFIG_FILE="/kaggle/working/frpc.toml"
# CHOICE="$1"
# if [ "$CHOICE" -eq 3 ]; then
#   TARGET_REMOTE_PORT="21663"
# elif [ "$CHOICE" -eq 4 ]; then
#   TARGET_REMOTE_PORT="21664"

# elif [ "$CHOICE" -eq 5 ]; then
#   TARGET_REMOTE_PORT="21665"
# elif [ "$CHOICE" -eq 6 ]; then
#   TARGET_REMOTE_PORT="21666"
# elif [ "$CHOICE" -eq -1 ]; then
#   TARGET_REMOTE_PORT="21673"
# else
#   echo "Invalid CHOICE: $CHOICE"
#   echo "Only 1 or 2 are supported."
#   exit 1 # 退出并返回错误码
# fi
# sed -i "s/REMOTE_PORT/$TARGET_REMOTE_PORT/g" "$FRP_CONFIG_FILE"
# sleep 2
