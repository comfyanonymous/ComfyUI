import json
from pathlib import Path
import sys
import time
from typing import Tuple

import requests
import folder_paths
import numpy as np
import torch
from PIL import Image, ImageOps
import cv2
import io
from typing import Tuple
import torch
import subprocess
import torchvision.transforms as transforms
from .lib import image, utils
from .lib.image import pil2tensor, tensor2pil
import os
import logging

# setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def ffmpeg_process(args, file_path, env):
    res = None
    frame_data = yield
    total_frames_output = 0
    if res != b'':
        with subprocess.Popen(args + [file_path], stderr=subprocess.PIPE,
                            stdin=subprocess.PIPE, env=env) as proc:
            try:
                while frame_data is not None:
                    proc.stdin.write(frame_data)
                    frame_data = yield
                    total_frames_output+=1
                proc.stdin.flush()
                proc.stdin.close()
                res = proc.stderr.read()
            except BrokenPipeError as e:
                res = proc.stderr.read()
                raise Exception("An error occurred in the ffmpeg subprocess:\n" \
                        + res.decode("utf-8"))
    yield total_frames_output
    if len(res) > 0:
        print(res.decode("utf-8"), end="", file=sys.stderr)

class MD_LoadImageFromUrl:
    """Load an image from the given URL"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": (
                    "STRING",
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load"
    CATEGORY = "MemeDeck"

    def load(self, url):
        # strip out any quote characters
        url = url.replace("'", "")
        url = url.replace('"', '')
        
        if url is None:
            raise ValueError("URL is required")
        
        img = Image.open(requests.get(url, stream=True).raw)
        img = ImageOps.exif_transpose(img)
        return (pil2tensor(img),)

class MD_ImageToMotionPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "Image": ("IMAGE", {}),
                "clip": ("CLIP", {"tooltip": "The CLIP model used for encoding the text."}),
                "pre_prompt": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "masterpiece, 4k, HDR, cinematic,",
                    },
                ),
                "post_prompt": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "The scene appears to be from a movie or TV show.",
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Respond in a single flowing paragraph. Start with main action in a single sentence. Then add specific details about movements and gestures. Then describe character/object appearances precisely. After that, specify camera angles and movements, static camera motion, or minimal camera motion. Then describe lighting and colors.\nNo more than 200 words.\nAdditional instructions:",
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, unnatural motion, fused fingers, extra limbs, floating away, bad anatomy, weird hand, ugly, disappearing objects, closed captions, cross-eyed",
                    },
                ),
                "max_tokens": ("INT", {"min": 1, "max": 2048, "default": 200}),
            },
            # "optional": {
            #     "temperature": ("FLOAT", {"min": 0.0, "max": 1.0, "step": 0.01, "default": 0.2}),
            #     "top_p": ("FLOAT", {"min": 0.0, "max": 1.0, "step": 0.01, "default": 0.9}),
            # }
        }

    
    RETURN_TYPES = ("STRING", "STRING", "CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("prompt_string", "negative_prompt", "positive_conditioning", "negative_conditioning")
    FUNCTION = "generate_completion"
    CATEGORY = "MemeDeck"

    def generate_completion(
        self, pre_prompt: str, post_prompt: str, Image: torch.Tensor, clip, prompt: str, negative_prompt: str,
        # temperature: float, 
        # top_p: float, 
        max_tokens: int
    ) -> Tuple[str]:
        # start a timer
        start_time = time.time()
        b64image = image.pil2base64(image.tensor2pil(Image))
        # change this to a endpoint on localhost:5010/inference that takes a json with the image and the prompt
        
        response = requests.post("http://127.0.0.1:5010/inference", json={
            "image_url": f"data:image/jpeg;base64,{b64image}",
            "prompt": prompt,
            "temperature": 0.2,
            "top_p": 0.7,
            "max_gen_len": max_tokens,
        })
        if response.status_code != 200:
            raise Exception(f"Failed to generate completion: {response.text}")
        end_time = time.time()
        
        logger.info(f"Motion prompt took: {end_time - start_time} seconds")
        full_prompt = f"{pre_prompt}\n{response.json()['result']} {post_prompt}"
        
        pos_tokens = clip.tokenize(full_prompt)
        pos_output = clip.encode_from_tokens(pos_tokens, return_pooled=True, return_dict=True)
        pos_cond = pos_output.pop("cond")
        
        neg_tokens = clip.tokenize(negative_prompt)
        neg_output = clip.encode_from_tokens(neg_tokens, return_pooled=True, return_dict=True)
        neg_cond = neg_output.pop("cond")
        
        return (full_prompt, negative_prompt, [[pos_cond, pos_output]], [[neg_cond, neg_output]])


class MD_CompressAdjustNode:
    """
    Detect compression level and adjust to desired CRF.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "desired_crf": ("INT", {
                    "default": 28,
                    "min": 0,
                    "max": 51,
                    "step": 1
                }),
                "width": ("INT", {
                    "default": 640,
                    "description": "The width of the video."
                }),
                "height": ("INT", {
                    "default": 640,
                    "description": "The height of the video."
                }),
            },
            "optional": {
                "base_crf": ("INT", ),
                "weights": ("STRING", {
                    "multiline": True,
                    "default": json.dumps({
                        "ideal_blockiness": 600,
                        "ideal_edge_density": 12,
                        "ideal_color_variation": 10000,
                        "blockiness_weight": -0.006,
                        "edge_density_weight": 0.32,
                        "color_variation_weight": -0.00005
                    }),
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "INT", "INT")
    RETURN_NAMES = ("adjusted_image", "crf", "width", "height")
    FUNCTION = "tensor_to_video_and_back"
    CATEGORY = "MemeDeck"
    
    def __init__(self):
        self.base_crf = 28
        # baseline values
        self.ideal_blockiness = 600
        self.ideal_edge_density = 12
        self.ideal_color_variation = 10000
        
        # weights
        self.blockiness_weight = -0.006
        self.edge_density_weight = 0.32
        self.color_variation_weight = -0.00005
    
    def tensor_to_int(self,tensor, bits):
        tensor = tensor.cpu().numpy() * (2**bits-1)
        return np.clip(tensor, 0, (2**bits-1))

    def tensor_to_bytes(self, tensor):
        return self.tensor_to_int(tensor, 8).astype(np.uint8)
            
    def detect_image_clarity(self, image):
        # detect the clarity of the image
        # return a score between 0 and 100
        # 0 is the lowest clarity
        # 100 is the highest clarity
        return 100

    def analyze_compression_artifacts(self, img, width=640, height=640):
        """
        Analyzes an image for potential compression artifacts.

        Args:
            image_path (str): Path to the image file.

        Returns:
            dict: A dictionary containing metrics related to compression artifacts.
        """

        # img = cv2.imread(image_path)
        # resize image to 640x640
        img = cv2.resize(img, (width, height))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Calculate blockiness (common in high compression)
        blockiness = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Edge detection (blurring can indicate compression)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges) / (gray.shape[0] * gray.shape[1])

        # Color histogram analysis (color banding in low bitrate compression)
        hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        color_variation = np.std(hist)

        return {
            "blockiness": blockiness,
            "edge_density": edge_density,
            "color_variation": color_variation
        }
        
    def calculate_crf(self, analysis_results, ideal_blockiness, ideal_edge_density, 
                  ideal_color_variation, blockiness_weight, 
                  edge_density_weight, color_variation_weight):
        """
        Calculates the target CRF based on analysis results and weights.
        """

        target_crf = self.base_crf + (blockiness_weight * (analysis_results["blockiness"] - ideal_blockiness)) \
                    + (edge_density_weight * (analysis_results["edge_density"] - ideal_edge_density)) \
                    + (color_variation_weight * (analysis_results["color_variation"] - ideal_color_variation))

        # Clamp CRF to a reasonable range (optional)
        target_crf = max(18, min(35, target_crf))  
        target_crf = round(target_crf, 2)
        return target_crf
    
    def tensor_to_video_and_back(self, image, desired_crf=28, width=832, height=832, weights=None, base_crf=28):        
        temp_dir = "temp_video"
        filename = f"frame_{time.time()}".split('.')[0]
        os.makedirs(temp_dir, exist_ok=True)
        
        if base_crf:
            self.base_crf = base_crf
        
        if weights:
            weights = json.loads(weights)
            self.ideal_blockiness = weights["ideal_blockiness"]
            self.ideal_edge_density = weights["ideal_edge_density"]
            self.ideal_color_variation = weights["ideal_color_variation"]
            self.blockiness_weight = weights["blockiness_weight"]
            self.edge_density_weight = weights["edge_density_weight"]
            self.color_variation_weight = weights["color_variation_weight"]
        
        # Convert single image to list if necessary
        if len(image.shape) == 3:
            image = [image]

        first_image = image[0]
        
        has_alpha = first_image.shape[-1] == 4
        dim_alignment = 8
        if (first_image.shape[1] % dim_alignment) or (first_image.shape[0] % dim_alignment):
            # pad the image to the nearest multiple of 8
            to_pad = (-first_image.shape[1] % dim_alignment,
                        -first_image.shape[0] % dim_alignment)
            padding = (to_pad[0]//2, to_pad[0] - to_pad[0]//2,
                        to_pad[1]//2, to_pad[1] - to_pad[1]//2)
            padfunc = torch.nn.ReplicationPad2d(padding)
            def pad(image):
                image = image.permute((2,0,1))#HWC to CHW
                padded = padfunc(image.to(dtype=torch.float32))
                return padded.permute((1,2,0))
            # pad single image
            first_image = pad(first_image)
            new_dims = (-first_image.shape[1] % dim_alignment + first_image.shape[1],
                        -first_image.shape[0] % dim_alignment + first_image.shape[0])
            dimensions = f"{new_dims[0]}x{new_dims[1]}"
            logger.warn("Output images were not of valid resolution and have had padding applied")
        else:
            dimensions = f"{first_image.shape[1]}x{first_image.shape[0]}"
        
        first_image_bytes = self.tensor_to_bytes(first_image).tobytes()

        if has_alpha:
            i_pix_fmt = 'rgba'
        else:
            i_pix_fmt = 'rgb24'
        
        # default bitrate and frame rate
        frame_rate = 24

        image_cv2 = cv2.cvtColor(np.array(tensor2pil(image)), cv2.COLOR_RGB2BGR)
        # calculate the crf based on the image
        analysis_results = self.analyze_compression_artifacts(image_cv2, width=width, height=height)
        logger.info(f"compression analysis_results: {analysis_results}")
        calculated_crf = self.calculate_crf(analysis_results, self.ideal_blockiness, self.ideal_edge_density, 
                  self.ideal_color_variation, self.blockiness_weight, 
                  self.edge_density_weight, self.color_variation_weight)
        
        if desired_crf is 0:
            desired_crf = calculated_crf
        
        logger.info(f"calculated_crf: {calculated_crf}")
        logger.info(f"desired_crf: {desired_crf}")
        args = [
            utils.ffmpeg_path, 
            "-v", "error", 
            "-f", "rawvideo", 
            "-pix_fmt", i_pix_fmt,
            "-s", dimensions, 
            "-r", str(frame_rate), 
            "-i", "-",
            "-y", 
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", str(desired_crf),
        ]
        
        video_path = os.path.abspath(str(Path(temp_dir) / f"{filename}.mp4"))
        env = os.environ.copy()
        output_process = ffmpeg_process(args, video_path, env)
        
        # Proceed to first yield
        output_process.send(None)
        output_process.send(first_image_bytes)
        try:
            output_process.send(None)  # Signal end of input
            next(output_process)  # Get the final yield
        except StopIteration:
            pass
        
        time.sleep(0.5)
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not created at {video_path}")
            
        # load the video h264 codec
        video = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        if not video.isOpened():
            raise RuntimeError(f"Failed to open video file: {video_path}")
        
        # read the first frame
        ret, frame = video.read()
        if not ret:
            raise RuntimeError("Failed to read frame from video")
        
        video.release()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            os.remove(video_path)
        except OSError as e:
            print(f"Warning: Could not remove temporary file {video_path}: {e}")
        
        # convert the frame to a PIL image for ComfyUI
        frame = Image.fromarray(frame)
        frame_tensor = pil2tensor(frame)
        
        return (frame_tensor, desired_crf, width, height)