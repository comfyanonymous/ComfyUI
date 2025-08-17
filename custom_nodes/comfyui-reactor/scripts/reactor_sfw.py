from transformers import pipeline
from PIL import Image
import io
import logging
import os
import comfy.model_management as model_management
from reactor_utils import download
from scripts.reactor_logger import logger

MODEL_EXISTS = False

def ensure_nsfw_model(nsfwdet_model_path):
    """Download NSFW detection model if it doesn't exist"""
    global MODEL_EXISTS
    downloaded = 0
    nd_urls = [
        "https://huggingface.co/AdamCodd/vit-base-nsfw-detector/resolve/main/config.json",
        "https://huggingface.co/AdamCodd/vit-base-nsfw-detector/resolve/main/model.safetensors",
        "https://huggingface.co/AdamCodd/vit-base-nsfw-detector/resolve/main/preprocessor_config.json",
    ]
    for model_url in nd_urls:
        model_name = os.path.basename(model_url)
        model_path = os.path.join(nsfwdet_model_path, model_name)
        if not os.path.exists(model_path):
            if not os.path.exists(nsfwdet_model_path):
                os.makedirs(nsfwdet_model_path)
            download(model_url, model_path, model_name)
        if os.path.exists(model_path):
            downloaded += 1
    MODEL_EXISTS = True if downloaded == 3 else False
    return MODEL_EXISTS

SCORE = 0.96

logging.getLogger("transformers").setLevel(logging.ERROR)

def nsfw_image(img_data, model_path: str):
    if not MODEL_EXISTS:
        logger.status("Ensuring NSFW detection model exists...")
        if not ensure_nsfw_model(model_path):
            return True
    device = model_management.get_torch_device()
    with Image.open(io.BytesIO(img_data)) as img:
        if "cpu" in str(device):
            predict = pipeline("image-classification", model=model_path)
        else:
            device_id = 0
            if "cuda" in str(device):
                device_id = int(str(device).split(":")[1])
            predict = pipeline("image-classification", model=model_path, device=device_id)
        result = predict(img)
        if result[0]["label"] == "nsfw" and result[0]["score"] > SCORE:
            logger.status(f'NSFW content detected with score={result[0]["score"]}, skipping...')
            return True
        return False
