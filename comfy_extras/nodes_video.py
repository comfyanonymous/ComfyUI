from __future__ import annotations

import os
import av
import torch
import folder_paths
import json
from typing import Optional, Literal
from fractions import Fraction
from comfy.comfy_types import IO, FileLocator, ComfyNodeABC
from comfy_api.input import ImageInput, AudioInput, VideoInput
from comfy_api.util import VideoContainer, VideoCodec, VideoComponents
from comfy_api.input_impl import VideoFromFile, VideoFromComponents
from comfy.cli_args import args
from huggingface_hub import HfApi, HfFolder

class SaveWEBM:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"images": ("IMAGE", ),
                     "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                     "codec": (["vp9", "av1"],),
                     "fps": ("FLOAT", {"default": 24.0, "min": 0.01, "max": 1000.0, "step": 0.01}),
                     "crf": ("FLOAT", {"default": 32.0, "min": 0, "max": 63.0, "step": 1, "tooltip": "Higher crf means lower quality with a smaller file size, lower crf means higher quality higher filesize."}),
                     },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "image/video"

    EXPERIMENTAL = True

    def save_images(self, images, codec, fps, filename_prefix, crf, prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])

        file = f"{filename}_{counter:05}_.webm"
        container = av.open(os.path.join(full_output_folder, file), mode="w")

        if prompt is not None:
            container.metadata["prompt"] = json.dumps(prompt)

        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                container.metadata[x] = json.dumps(extra_pnginfo[x])

        codec_map = {"vp9": "libvpx-vp9", "av1": "libsvtav1"}
        stream = container.add_stream(codec_map[codec], rate=Fraction(round(fps * 1000), 1000))
        stream.width = images.shape[-2]
        stream.height = images.shape[-3]
        stream.pix_fmt = "yuv420p10le" if codec == "av1" else "yuv420p"
        stream.bit_rate = 0
        stream.options = {'crf': str(crf)}
        if codec == "av1":
            stream.options["preset"] = "6"

        for frame in images:
            frame = av.VideoFrame.from_ndarray(torch.clamp(frame[..., :3] * 255, min=0, max=255).to(device=torch.device("cpu"), dtype=torch.uint8).numpy(), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        container.mux(stream.encode())
        container.close()

        results: list[FileLocator] = [{
            "filename": file,
            "subfolder": subfolder,
            "type": self.type
        }]

        return {"ui": {"images": results, "animated": (True,)}}  # TODO: frontend side

class SaveVideo(ComfyNodeABC):
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type: Literal["output"] = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (IO.VIDEO, {"tooltip": "The video to save."}),
                "filename_prefix": ("STRING", {"default": "video/ComfyUI", "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."}),
                "format": (VideoContainer.as_input(), {"default": "auto", "tooltip": "The format to save the video as."}),
                "codec": (VideoCodec.as_input(), {"default": "auto", "tooltip": "The codec to use for the video."}),
                "encrypt": ("BOOLEAN", {"default": False, "tooltip": "Enable simple XOR encryption"}),
                "encryption_key": ("STRING", {"default": "key4comfy", "tooltip": "Encryption key for XOR encryption"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "save_video"

    # OUTPUT_NODE = True

    CATEGORY = "image/video"
    DESCRIPTION = "Saves the input images to your ComfyUI output directory."


    def upload(self, hf_token, dataset_name, file_path):
        api = HfApi()
        HfFolder.save_token(hf_token)

        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=os.path.basename(file_path),
            repo_id=dataset_name,
            repo_type="dataset",
            token=hf_token,
        )


    def simple_xor_encrypt(self, data: bytes, key: str) -> bytes:
        """简单的XOR加密"""
        key_bytes = key.encode('utf-8')
        key_length = len(key_bytes)
        encrypted = bytearray()
        
        for i, byte in enumerate(data):
            encrypted.append(byte ^ key_bytes[i % key_length])
        
        return bytes(encrypted)

    def save_video(self, video: VideoInput, filename_prefix, format, codec, encrypt, encryption_key, prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        width, height = video.get_dimensions()
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix,
            self.output_dir,
            width,
            height
        )
        results: list[FileLocator] = list()
        saved_metadata = None
        if not args.disable_metadata:
            metadata = {}
            if extra_pnginfo is not None:
                metadata.update(extra_pnginfo)
            if prompt is not None:
                metadata["prompt"] = prompt
            if len(metadata) > 0:
                saved_metadata = metadata
        
        # 生成文件路径
        file_extension = VideoContainer.get_extension(format)
        temp_file_path = os.path.join(full_output_folder, f"temp_{filename}_{counter:05}_.{file_extension}")
        final_file_path = os.path.join(full_output_folder, f"{filename}_{counter:05}_.{file_extension}")
        
        # 先保存到临时文件
        video.save_to(
            temp_file_path,
            format=format,
            codec=codec,
            metadata=saved_metadata
        )
        
        # 如果启用加密，对文件进行XOR加密
        if encrypt:
            # 读取文件内容
            with open(temp_file_path, 'rb') as f:
                video_data = f.read()
            
            # 删除临时文件
            os.remove(temp_file_path)
            
            # 加密数据
            encrypted_data = self.simple_xor_encrypt(video_data, encryption_key)
            
            # 保存加密后的文件
            with open(final_file_path, 'wb') as f:
                f.write(encrypted_data)
        else:
            # 不加密，直接重命名
            os.rename(temp_file_path, final_file_path)

        # 上传到Hugging Face
        self.upload(args.hf_token, args.hf_dataset_name, final_file_path)
        final_filename = os.path.basename(final_file_path)

        url = f'<a href="http://comfy.helloitsme-docs.serv00.net/decrypt_and_serve_video?url=https://huggingface.co/datasets/{args.hf_dataset_name}/resolve/main/{final_filename}&key={encryption_key}" target="_blank">Video generated, click to open: {final_filename}</a>'
        # 只返回url到ui.text
        return (url,)

class CreateVideo(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": (IO.IMAGE, {"tooltip": "The images to create a video from."}),
                "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0, "step": 1.0}),
            },
            "optional": {
                "audio": (IO.AUDIO, {"tooltip": "The audio to add to the video."}),
            }
        }

    RETURN_TYPES = (IO.VIDEO,)
    FUNCTION = "create_video"

    CATEGORY = "image/video"
    DESCRIPTION = "Create a video from images."

    def create_video(self, images: ImageInput, fps: float, audio: Optional[AudioInput] = None):
        return (VideoFromComponents(
            VideoComponents(
            images=images,
            audio=audio,
            frame_rate=Fraction(fps),
            )
        ),)

class GetVideoComponents(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (IO.VIDEO, {"tooltip": "The video to extract components from."}),
            }
        }
    RETURN_TYPES = (IO.IMAGE, IO.AUDIO, IO.FLOAT)
    RETURN_NAMES = ("images", "audio", "fps")
    FUNCTION = "get_components"

    CATEGORY = "image/video"
    DESCRIPTION = "Extracts all components from a video: frames, audio, and framerate."

    def get_components(self, video: VideoInput):
        components = video.get_components()

        return (components.images, components.audio, float(components.frame_rate))

class LoadVideo(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["video"])
        return {"required":
                    {"file": (sorted(files), {"video_upload": True})},
                }

    CATEGORY = "image/video"

    RETURN_TYPES = (IO.VIDEO,)
    FUNCTION = "load_video"
    def load_video(self, file):
        video_path = folder_paths.get_annotated_filepath(file)
        return (VideoFromFile(video_path),)

    @classmethod
    def IS_CHANGED(cls, file):
        video_path = folder_paths.get_annotated_filepath(file)
        mod_time = os.path.getmtime(video_path)
        # Instead of hashing the file, we can just use the modification time to avoid
        # rehashing large files.
        return mod_time

    @classmethod
    def VALIDATE_INPUTS(cls, file):
        if not folder_paths.exists_annotated_filepath(file):
            return "Invalid video file: {}".format(file)

        return True

class LoadVideoEncrypted(ComfyNodeABC):
    """支持解密加密视频的加载节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        
        # 使用PIL来过滤视频和动画文件，包括WebP动画
        def is_video_or_animated_file(file_path):
            """使用PIL判断文件是否为视频或动画文件"""
            try:
                from PIL import Image
                with Image.open(file_path) as img:
                    # 检查是否为动画文件（包括WebP动画、GIF等）
                    if hasattr(img, 'is_animated') and img.is_animated:
                        return True
                    
                    # 检查帧数，如果大于1帧，可能是视频
                    if hasattr(img, 'n_frames') and img.n_frames > 1:
                        return True
                    
                    # 检查持续时间，如果有持续时间信息，可能是视频
                    if hasattr(img, 'duration') and img.duration > 0:
                        return True
                    
                    return False
            except Exception:
                # 如果PIL无法打开，使用备用方案
                import mimetypes
                mime_type, _ = mimetypes.guess_type(file_path)
                if mime_type:
                    return mime_type.startswith('video/')
                
                video_extensions = {'.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv', '.m4v'}
                file_ext = os.path.splitext(file_path)[1].lower()
                return file_ext in video_extensions
        
        # 过滤出视频和动画文件
        video_files = []
        for file in files:
            file_path = os.path.join(input_dir, file)
            if is_video_or_animated_file(file_path):
                video_files.append(file)
        
        return {
            "required": {
                "file": (sorted(video_files), {"video_upload": True}),
                "is_encrypted": ("BOOLEAN", {"default": False, "tooltip": "Whether the file is XOR encrypted"}),
                "encryption_key": ("STRING", {"default": "mysecretkey", "tooltip": "Encryption key for XOR decryption"}),
            }
        }

    CATEGORY = "image/video"
    RETURN_TYPES = (IO.VIDEO,)
    FUNCTION = "load_video_encrypted"
    
    def simple_xor_decrypt(self, data: bytes, key: str) -> bytes:
        """简单的XOR解密（XOR加密是对称的）"""
        key_bytes = key.encode('utf-8')
        key_length = len(key_bytes)
        decrypted = bytearray()
        
        for i, byte in enumerate(data):
            decrypted.append(byte ^ key_bytes[i % key_length])
        
        return bytes(decrypted)
    
    def load_video_encrypted(self, file, is_encrypted, encryption_key):
        video_path = folder_paths.get_annotated_filepath(file)
        
        if not is_encrypted:
            # 不加密，直接加载
            return (VideoFromFile(video_path),)
        else:
            # 读取加密文件
            with open(video_path, 'rb') as f:
                encrypted_data = f.read()
            
            # 解密数据
            decrypted_data = self.simple_xor_decrypt(encrypted_data, encryption_key)
            
            # 直接使用BytesIO，无需创建临时文件
            import io
            video_bytes_io = io.BytesIO(decrypted_data)
            
            return (VideoFromFile(video_bytes_io),)

    @classmethod
    def IS_CHANGED(cls, file, is_encrypted, encryption_key):
        video_path = folder_paths.get_annotated_filepath(file)
        mod_time = os.path.getmtime(video_path)
        return mod_time

    @classmethod
    def VALIDATE_INPUTS(cls, file, is_encrypted, encryption_key):
        if not folder_paths.exists_annotated_filepath(file):
            return "Invalid video file: {}".format(file)
        return True


class DisplayLinkNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "url_string": ("STRING", {"multiline": False, "default": "https://www.example.com"}),
            },
            "optional": {
                "link_text": ("STRING", {"multiline": False, "default": "Click Here"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "display_link"
    CATEGORY = "image/video"

    def display_link(self, url_string, link_text):
        # Create the HTML anchor tag
        # The target="_blank" attribute opens the link in a new tab
        html_link = f'<a href="{url_string}" target="_blank">{link_text}</a>'
        return (html_link,)

NODE_CLASS_MAPPINGS = {
    "SaveWEBM": SaveWEBM,
    "SaveVideo": SaveVideo,
    "DisplayLinkNode": DisplayLinkNode,
    "CreateVideo": CreateVideo,
    "GetVideoComponents": GetVideoComponents,
    "LoadVideo": LoadVideo,
    "LoadVideoEncrypted": LoadVideoEncrypted,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveVideo": "Save Video",
    "DisplayLinkNode": "Display Link",
    "CreateVideo": "Create Video",
    "GetVideoComponents": "Get Video Components",
    "LoadVideo": "Load Video",
    "LoadVideoEncrypted": "Load Video (Encrypted)",
}
