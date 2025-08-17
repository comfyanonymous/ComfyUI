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

# wan lora: dabaichui 试一试哦！！
wget -c https://huggingface.co/Heng365/dabaichui/resolve/main/dabaichui.safetensors -P ./models/loras

wget -c https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl.vae.safetensors -P ./models/vae

#Flux kontext
ln -s /kaggle/input/flux-ae/flux-ae.safetensors ./models/vae/ae.safetensors
ln -s /kaggle/input/clip-l/clip_l.safetensors ./models/text_encoders/clip_l.safetensors
ln -s /kaggle/input/t5xxl-fp8-e4m3fn-scaled/t5xxl_fp8_e4m3fn_scaled.safetensors ./models/text_encoders/t5xxl_fp8_e4m3fn_scaled.safetensors
ln -s /kaggle/input/t5xxl-fp8-e4m3fn/t5xxl_fp8_e4m3fn.safetensors ./models/text_encoders/t5xxl_fp8_e4m3fn.safetensors
ln -s /kaggle/input/flux1-dev-kontext-fp8-scaled/flux1-dev-kontext_fp8_scaled.safetensors ./models/diffusion_models/flux1-dev-kontext_fp8_scaled.safetensors

#Flux dev
ln -s /kaggle/input/flux1-dev-fp8/flux1-dev-fp8.safetensors ./models/diffusion_models/flux1-dev-fp8.safetensors

# kontext turnaround sheet lora: It won't work well on real human photos
# wget -c https://huggingface.co/reverentelusarca/kontext-turnaround-sheet-lora-v1/resolve/main/kontext-turnaround-sheet-v1.safetensors -P ./models/loras

#nunchaku
# ln -s /kaggle/input/svdq-int4-r32-flux-1-kontext-dev/svdq-int4_r32-flux.1-kontext-dev.safetensors ./models/diffusion_models


#iniverseMixSFWNSFW_ponyRealGuofengV51  dreamshaperXL_lightningDPMSDE
# wget -c "https://civitai.com/api/download/models/1759168?type=Model&format=SafeTensor&size=full&fp=fp16" -O ./models/checkpoints/Juggernaut-XL-Ragnarok.safetensors
ln -s /kaggle/input/juggernaut-xl-ragnarok/Juggernaut-XL-Ragnarok.safetensors ./models/checkpoints

#InstantID
mkdir -p ./models/instantid
# wget -c https://huggingface.co/InstantX/InstantID/resolve/main/ip-adapter.bin -P ./models/instantid
ln -s /kaggle/input/ip-adapter/ip-adapter.bin ./models/instantid/ip-adapter.bin

wget -c https://huggingface.co/MonsterMMORPG/tools/resolve/main/antelopev2.zip -P ./models
mkdir -p ./models/insightface/models
unzip ./models/antelopev2.zip -d ./models/insightface/models

# place it in the ComfyUI controlnet directory： -O overrides -P if both are specified.
# wget -c https://huggingface.co/InstantX/InstantID/resolve/main/ControlNetModel/diffusion_pytorch_model.safetensors -O ./models/controlnet/instantid-controlnet.safetensors
ln -s /kaggle/input/diffusion-pytorch-model/diffusion_pytorch_model.safetensors ./models/controlnet/instantid-controlnet.safetensors
wget -c https://huggingface.co/InstantX/InstantID/resolve/main/ControlNetModel/config.json -P ./models/controlnet

wget -c https://huggingface.co/lllyasviel/sd_control_collection/resolve/d1b278d0d1103a3a7c4f7c2c327d236b082a75b1/thibaud_xl_openpose.safetensors -P ./models/controlnet

# When using ultralytics models, save them separately in models/ultralytics/bbox and models/ultralytics/segm depending on the type of model.
mkdir -p ./models/ultralytics/bbox
wget -c https://huggingface.co/Tenofas/ComfyUI/resolve/d79945fb5c16e8aef8a1eb3ba1788d72152c6d96/ultralytics/bbox/Eyes.pt -P ./models/ultralytics/bbox

wget -c https://huggingface.co/YouLiXiya/YL-SAM/resolve/main/sam_vit_b_01ec64.pth -P  ./models/sams
wget -c https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/detection/bbox/face_yolov8m.pt -P ./models/ultralytics/bbox



# ComfyUI_IPAdapter_plus models
mkdir -p ./models/ipadapter
wget -c https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors -O ./models/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors
# SDXL ipadapter model
wget -c https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors -P ./models/ipadapter
wget -c https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors -P ./models/ipadapter
wget -c https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl_vit-h.safetensors -P ./models/ipadapter

# pulid model
mkdir -p ./models/pulid/
wget -c https://huggingface.co/guozinan/PuLID/resolve/main/pulid_flux_v0.9.1.safetensors -P ./models/pulid/

# ComfyUI-Kolors-MZ faceid做什么用的?
# wget -c https://huggingface.co/Kwai-Kolors/Kolors-IP-Adapter-Plus/resolve/main/ip_adapter_plus_general.bin -P ./models/ipadapter
# wget -c https://huggingface.co/Kwai-Kolors/Kolors-IP-Adapter-Plus/resolve/main/image_encoder/pytorch_model.bin -P ./models/clip_vision


# wan2.1 i2v
# wget -c https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_i2v_480p_14B_fp8_scaled.safetensors -P ./models/diffusion_models
# wget -c https://huggingface.co/UmeAiRT/ComfyUI-Auto_installer/resolve/main/models/unet/WAN/Wan2.1-VACE-14B-Q5_K_S.gguf -P ./models/diffusion_models

# wget -c https://huggingface.co/UmeAiRT/ComfyUI-Auto_installer/resolve/main/models/unet/WAN/Wan2.1-VACE-14B-Q4_K_S.gguf -P ./models/diffusion_models

# 放大
wget -c https://huggingface.co/schwgHao/RealESRGAN_x4plus/resolve/main/RealESRGAN_x4plus.pth -P ./models/upscale_models

ln -s /kaggle/input/clip-vision-h/clip_vision_h.safetensors ./models/clip_vision/clip_vision_h.safetensors




# ----------------   安装自定义插件节点  ----------------

# 1 ComfyUI-Manager
cd custom_nodes
git clone https://github.com/ltdrdata/ComfyUI-Manager comfyui-manager
cd /kaggle/ComfyUI

# 2 安装 nunchaku
# pip install facexlib
# pip install onnxruntime
# pip install insightface
# pip install https://huggingface.co/mit-han-lab/nunchaku/resolve/main/nunchaku-0.3.1+torch2.6-cp311-cp311-linux_x86_64.whl
# # wget -c https://huggingface.co/mit-han-lab/nunchaku-flux.1-kontext-dev/resolve/main/svdq-int4_r32-flux.1-kontext-dev.safetensors -P ./models/diffusion_models
# cd /kaggle/ComfyUI

# nunchaku_nodes: 去掉加载model时候的-P,还报错吗？！
# cd custom_nodes
# git clone https://github.com/mit-han-lab/ComfyUI-nunchaku nunchaku_nodes
# cd /kaggle/ComfyUI


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


cd custom_nodes
git clone https://github.com/kijai/ComfyUI-KJNodes.git
cd ComfyUI-KJNodes
pip install -r requirements.txt
cd /kaggle/ComfyUI


# These custom nodes used by workflow VACE ControlNet 1.0 (base).json ------------ start ------------


# cd custom_nodes
# git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
# cd ComfyUI-VideoHelperSuite
# pip install -r requirements.txt
# cd /kaggle/ComfyUI


# cd custom_nodes
# git clone https://github.com/Smirnov75/ComfyUI-mxToolkit.git
# cd /kaggle/ComfyUI

# cd custom_nodes
# git clone https://github.com/facok/ComfyUI-HunyuanVideoMultiLora.git
# cd /kaggle/ComfyUI


# cd custom_nodes
# git clone https://github.com/rgthree/rgthree-comfy.git
# cd /kaggle/ComfyUI

# cd custom_nodes
# git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git
# cd ComfyUI-Frame-Interpolation
# python install.py
# cd /kaggle/ComfyUI


# cd custom_nodes
# git clone https://github.com/WASasquatch/was-node-suite-comfyui.git
# cd was-node-suite-comfyui
# pip install -r requirements.txt
# cd /kaggle/ComfyUI

# cd custom_nodes
# git clone https://github.com/kijai/ComfyUI-Florence2.git
# cd ComfyUI-Florence2
# pip install -r requirements.txt
# cd /kaggle/ComfyUI


# cd custom_nodes
# git clone https://github.com/yuvraj108c/ComfyUI-Upscaler-Tensorrt.git
# cd ComfyUI-Upscaler-Tensorrt
# pip install -r requirements.txt
# cd /kaggle/ComfyUI

# cd custom_nodes
# git clone https://github.com/pollockjj/ComfyUI-MultiGPU.git
# cd /kaggle/ComfyUI


# cd custom_nodes
# git clone https://github.com/chflame163/ComfyUI_LayerStyle.git
# cd ComfyUI_LayerStyle
# pip install -r requirements.txt
# cd /kaggle/ComfyUI



# These custom nodes used by workflow VACE ControlNet 1.0 (base).json ------------ end ------------


# MV-Adapter
cd custom_nodes
git clone https://github.com/huanngzh/ComfyUI-MVAdapter.git
cd ComfyUI-MVAdapter
pip install -r requirements.txt
cd /kaggle/ComfyUI


#cog-consistent-character ??
cd custom_nodes
git clone --recurse-submodules https://github.com/fofr/cog-consistent-character.git
cd cog-consistent-character
python ./scripts/install_custom_nodes.py
cd /kaggle/ComfyUI


cd custom_nodes
git clone https://github.com/PowerHouseMan/ComfyUI-AdvancedLivePortrait.git
cd ComfyUI-AdvancedLivePortrait
pip install -r requirements.txt
cd /kaggle/ComfyUI



cd custom_nodes
git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack comfyui-impact-pack
cd comfyui-impact-pack
pip install -r requirements.txt
cd /kaggle/ComfyUI


cd custom_nodes
git clone https://github.com/ltdrdata/ComfyUI-Impact-Subpack
cd ComfyUI-Impact-Subpack
pip install -r requirements.txt
cd /kaggle/ComfyUI

cd custom_nodes
git clone https://github.com/cubiq/ComfyUI_InstantID
cd ComfyUI_InstantID
pip install -r requirements.txt
cd /kaggle/ComfyUI

cd custom_nodes
git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus
cd /kaggle/ComfyUI


# Goto test no pip install ...

cd custom_nodes
git clone https://github.com/cubiq/ComfyUI_essentials.git
cd ComfyUI_essentials
pip install -r requirements.txt
cd /kaggle/ComfyUI

cd custom_nodes
git clone https://github.com/crystian/comfyui-crystools.git
cd comfyui-crystools
pip install -r requirements.txt
cd /kaggle/ComfyUI

cd custom_nodes
git clone https://github.com/melMass/comfy_mtb.git
cd comfy_mtb
pip install -r requirements.txt
cd /kaggle/ComfyUI

# Test: no pip install ...
cd custom_nodes
git clone https://github.com/rgthree/rgthree-comfy.git
cd /kaggle/ComfyUI

cd custom_nodes
git clone https://github.com/kijai/ComfyUI-FluxTrainer.git
cd ComfyUI-FluxTrainer
pip install -r requirements.txt
cd /kaggle/ComfyUI


cd custom_nodes
git clone https://github.com/yolain/ComfyUI-Easy-Use
cd ComfyUI-Easy-Use
pip install -r requirements.txt
cd /kaggle/ComfyUI

cd custom_nodes
git clone https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes.git
cd /kaggle/ComfyUI

cd custom_nodes
git clone https://github.com/sipie800/ComfyUI-PuLID-Flux-Enhanced.git
cd ComfyUI-PuLID-Flux-Enhanced
pip install -r requirements.txt
cd /kaggle/ComfyUI

cd custom_nodes
git clone https://github.com/Fannovel16/comfyui_controlnet_aux/
cd comfyui_controlnet_aux
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
