import json
import os
import re
import numpy as np
import folder_paths # type: ignore
from ..services.service_registry import ServiceRegistry
from ..metadata_collector.metadata_processor import MetadataProcessor
from ..metadata_collector import get_metadata
from PIL import Image, PngImagePlugin
import piexif

class SaveImage:
    NAME = "Save Image (LoraManager)"
    CATEGORY = "Lora Manager/utils"
    DESCRIPTION = "Save images with embedded generation metadata in compatible format"

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4
        self.counter = 0
    
    # Add pattern format regex for filename substitution
    pattern_format = re.compile(r"(%[^%]+%)")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {
                    "default": "ComfyUI", 
                    "tooltip": "Base filename for saved images. Supports format patterns like %seed%, %width%, %height%, %model%, etc."
                }),
                "file_format": (["png", "jpeg", "webp"], {
                    "tooltip": "Image format to save as. PNG preserves quality, JPEG is smaller, WebP balances size and quality."
                }),
            },
            "optional": {
                "lossless_webp": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "When enabled, saves WebP images with lossless compression. Results in larger files but no quality loss."
                }),
                "quality": ("INT", {
                    "default": 100, 
                    "min": 1, 
                    "max": 100,
                    "tooltip": "Compression quality for JPEG and lossy WebP formats (1-100). Higher values mean better quality but larger files."
                }),
                "embed_workflow": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Embeds the complete workflow data into the image metadata. Only works with PNG and WebP formats."
                }),
                "add_counter_to_filename": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Adds an incremental counter to filenames to prevent overwriting previous images."
                }),
            },
            "hidden": {
                "id": "UNIQUE_ID",
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "process_image"
    OUTPUT_NODE = True

    def get_lora_hash(self, lora_name):
        """Get the lora hash from cache"""
        scanner = ServiceRegistry.get_service_sync("lora_scanner")
        
        # Use the new direct filename lookup method
        hash_value = scanner.get_hash_by_filename(lora_name)
        if hash_value:
            return hash_value
            
        return None

    def get_checkpoint_hash(self, checkpoint_path):
        """Get the checkpoint hash from cache"""
        scanner = ServiceRegistry.get_service_sync("checkpoint_scanner")
        
        if not checkpoint_path:
            return None
            
        # Extract basename without extension
        checkpoint_name = os.path.basename(checkpoint_path)
        checkpoint_name = os.path.splitext(checkpoint_name)[0]
        
        # Try direct filename lookup first
        hash_value = scanner.get_hash_by_filename(checkpoint_name)
        if hash_value:
            return hash_value
                
        return None

    def format_metadata(self, metadata_dict):
        """Format metadata in the requested format similar to userComment example"""
        if not metadata_dict:
            return ""
        
        # Helper function to only add parameter if value is not None
        def add_param_if_not_none(param_list, label, value):
            if value is not None:
                param_list.append(f"{label}: {value}")
        
        # Extract the prompt and negative prompt
        prompt = metadata_dict.get('prompt', '')
        negative_prompt = metadata_dict.get('negative_prompt', '')
        
        # Extract loras from the prompt if present
        loras_text = metadata_dict.get('loras', '')
        lora_hashes = {}
        
        # If loras are found, add them on a new line after the prompt
        if loras_text:
            prompt_with_loras = f"{prompt}\n{loras_text}"
            
            # Extract lora names from the format <lora:name:strength>
            lora_matches = re.findall(r'<lora:([^:]+):([^>]+)>', loras_text)
            
            # Get hash for each lora
            for lora_name, strength in lora_matches:
                hash_value = self.get_lora_hash(lora_name)
                if hash_value:
                    lora_hashes[lora_name] = hash_value
        else:
            prompt_with_loras = prompt
        
        # Format the first part (prompt and loras)
        metadata_parts = [prompt_with_loras]
        
        # Add negative prompt
        if negative_prompt:
            metadata_parts.append(f"Negative prompt: {negative_prompt}")
        
        # Format the second part (generation parameters)
        params = []
        
        # Add standard parameters in the correct order
        if 'steps' in metadata_dict:
            add_param_if_not_none(params, "Steps", metadata_dict.get('steps'))
        
        # Combine sampler and scheduler information
        sampler_name = None
        scheduler_name = None
        
        if 'sampler' in metadata_dict:
            sampler = metadata_dict.get('sampler')
            # Convert ComfyUI sampler names to user-friendly names
            sampler_mapping = {
                'euler': 'Euler',
                'euler_ancestral': 'Euler a',
                'dpm_2': 'DPM2',
                'dpm_2_ancestral': 'DPM2 a',
                'heun': 'Heun',
                'dpm_fast': 'DPM fast',
                'dpm_adaptive': 'DPM adaptive',
                'lms': 'LMS',
                'dpmpp_2s_ancestral': 'DPM++ 2S a',
                'dpmpp_sde': 'DPM++ SDE',
                'dpmpp_sde_gpu': 'DPM++ SDE',
                'dpmpp_2m': 'DPM++ 2M',
                'dpmpp_2m_sde': 'DPM++ 2M SDE',
                'dpmpp_2m_sde_gpu': 'DPM++ 2M SDE',
                'ddim': 'DDIM'
            }
            sampler_name = sampler_mapping.get(sampler, sampler)
        
        if 'scheduler' in metadata_dict:
            scheduler = metadata_dict.get('scheduler')
            scheduler_mapping = {
                'normal': 'Simple',
                'karras': 'Karras',
                'exponential': 'Exponential',
                'sgm_uniform': 'SGM Uniform',
                'sgm_quadratic': 'SGM Quadratic'
            }
            scheduler_name = scheduler_mapping.get(scheduler, scheduler)
        
        # Add combined sampler and scheduler information
        if sampler_name:
            if scheduler_name:
                params.append(f"Sampler: {sampler_name} {scheduler_name}")
            else:
                params.append(f"Sampler: {sampler_name}")
        
        # CFG scale (Use guidance if available, otherwise fall back to cfg_scale or cfg)
        if 'guidance' in metadata_dict:
            add_param_if_not_none(params, "CFG scale", metadata_dict.get('guidance'))
        elif 'cfg_scale' in metadata_dict:
            add_param_if_not_none(params, "CFG scale", metadata_dict.get('cfg_scale'))
        elif 'cfg' in metadata_dict:
            add_param_if_not_none(params, "CFG scale", metadata_dict.get('cfg'))
        
        # Seed
        if 'seed' in metadata_dict:
            add_param_if_not_none(params, "Seed", metadata_dict.get('seed'))
        
        # Size
        if 'size' in metadata_dict:
            add_param_if_not_none(params, "Size", metadata_dict.get('size'))
        
        # Model info
        if 'checkpoint' in metadata_dict:
            # Ensure checkpoint is a string before processing
            checkpoint = metadata_dict.get('checkpoint')
            if checkpoint is not None:
                # Get model hash
                model_hash = self.get_checkpoint_hash(checkpoint)
                
                # Extract basename without path
                checkpoint_name = os.path.basename(checkpoint)
                # Remove extension if present
                checkpoint_name = os.path.splitext(checkpoint_name)[0]
                
                # Add model hash if available
                if model_hash:
                    params.append(f"Model hash: {model_hash[:10]}, Model: {checkpoint_name}")
                else:
                    params.append(f"Model: {checkpoint_name}")
        
        # Add LoRA hashes if available
        if lora_hashes:
            lora_hash_parts = []
            for lora_name, hash_value in lora_hashes.items():
                lora_hash_parts.append(f"{lora_name}: {hash_value[:10]}")
            
            if lora_hash_parts:
                params.append(f"Lora hashes: \"{', '.join(lora_hash_parts)}\"")
        
        # Combine all parameters with commas
        metadata_parts.append(", ".join(params))
        
        # Join all parts with a new line
        return "\n".join(metadata_parts)

    # credit to nkchocoai
    # Add format_filename method to handle pattern substitution
    def format_filename(self, filename, metadata_dict):
        """Format filename with metadata values"""
        if not metadata_dict:
            return filename
            
        result = re.findall(self.pattern_format, filename)
        for segment in result:
            parts = segment.replace("%", "").split(":")
            key = parts[0]
            
            if key == "seed" and 'seed' in metadata_dict:
                filename = filename.replace(segment, str(metadata_dict.get('seed', '')))
            elif key == "width" and 'size' in metadata_dict:
                size = metadata_dict.get('size', 'x')
                w = size.split('x')[0] if isinstance(size, str) else size[0]
                filename = filename.replace(segment, str(w))
            elif key == "height" and 'size' in metadata_dict:
                size = metadata_dict.get('size', 'x')
                h = size.split('x')[1] if isinstance(size, str) else size[1]
                filename = filename.replace(segment, str(h))
            elif key == "pprompt" and 'prompt' in metadata_dict:
                prompt = metadata_dict.get('prompt', '').replace("\n", " ")
                if len(parts) >= 2:
                    length = int(parts[1])
                    prompt = prompt[:length]
                filename = filename.replace(segment, prompt.strip())
            elif key == "nprompt" and 'negative_prompt' in metadata_dict:
                prompt = metadata_dict.get('negative_prompt', '').replace("\n", " ")
                if len(parts) >= 2:
                    length = int(parts[1])
                    prompt = prompt[:length]
                filename = filename.replace(segment, prompt.strip())
            elif key == "model" and 'checkpoint' in metadata_dict:
                model = metadata_dict.get('checkpoint', '')
                model = os.path.splitext(os.path.basename(model))[0]
                if len(parts) >= 2:
                    length = int(parts[1])
                    model = model[:length]
                filename = filename.replace(segment, model)
            elif key == "date":
                from datetime import datetime
                now = datetime.now()
                date_table = {
                    "yyyy": f"{now.year:04d}",
                    "yy": f"{now.year % 100:02d}",
                    "MM": f"{now.month:02d}",
                    "dd": f"{now.day:02d}",
                    "hh": f"{now.hour:02d}",
                    "mm": f"{now.minute:02d}",
                    "ss": f"{now.second:02d}",
                }
                if len(parts) >= 2:
                    date_format = parts[1]
                    for k, v in date_table.items():
                        date_format = date_format.replace(k, v)
                    filename = filename.replace(segment, date_format)
                else:
                    date_format = "yyyyMMddhhmmss"
                    for k, v in date_table.items():
                        date_format = date_format.replace(k, v)
                    filename = filename.replace(segment, date_format)
                    
        return filename

    def save_images(self, images, filename_prefix, file_format, id, prompt=None, extra_pnginfo=None, 
                   lossless_webp=True, quality=100, embed_workflow=False, add_counter_to_filename=True):
        """Save images with metadata"""
        results = []

        # Get metadata using the metadata collector
        raw_metadata = get_metadata()
        metadata_dict = MetadataProcessor.to_dict(raw_metadata, id)
            
        metadata = self.format_metadata(metadata_dict)
        
        # Process filename_prefix with pattern substitution
        filename_prefix = self.format_filename(filename_prefix, metadata_dict)
        
        # Get initial save path info once for the batch
        full_output_folder, filename, counter, subfolder, processed_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0]
        )
        
        # Create directory if it doesn't exist
        if not os.path.exists(full_output_folder):
            os.makedirs(full_output_folder, exist_ok=True)
        
        # Process each image with incrementing counter
        for i, image in enumerate(images):
            # Convert the tensor image to numpy array
            img = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
            
            # Generate filename with counter if needed
            base_filename = filename
            if add_counter_to_filename:
                # Use counter + i to ensure unique filenames for all images in batch
                current_counter = counter + i
                base_filename += f"_{current_counter:05}_"
                
            # Set file extension and prepare saving parameters
            if file_format == "png":
                file = base_filename + ".png"
                file_extension = ".png"
                # Remove "optimize": True to match built-in node behavior
                save_kwargs = {"compress_level": self.compress_level}
                pnginfo = PngImagePlugin.PngInfo()
            elif file_format == "jpeg":
                file = base_filename + ".jpg"
                file_extension = ".jpg"
                save_kwargs = {"quality": quality, "optimize": True}
            elif file_format == "webp":
                file = base_filename + ".webp" 
                file_extension = ".webp"
                # Add optimization param to control performance
                save_kwargs = {"quality": quality, "lossless": lossless_webp, "method": 0}
            
            # Full save path
            file_path = os.path.join(full_output_folder, file)
            
            # Save the image with metadata
            try:
                if file_format == "png":
                    if metadata:
                        pnginfo.add_text("parameters", metadata)
                    if embed_workflow and extra_pnginfo is not None:
                        workflow_json = json.dumps(extra_pnginfo["workflow"])
                        pnginfo.add_text("workflow", workflow_json)
                    save_kwargs["pnginfo"] = pnginfo
                    img.save(file_path, format="PNG", **save_kwargs)
                elif file_format == "jpeg":
                    # For JPEG, use piexif
                    if metadata:
                        try:
                            exif_dict = {'Exif': {piexif.ExifIFD.UserComment: b'UNICODE\0' + metadata.encode('utf-16be')}}
                            exif_bytes = piexif.dump(exif_dict)
                            save_kwargs["exif"] = exif_bytes
                        except Exception as e:
                            print(f"Error adding EXIF data: {e}")
                    img.save(file_path, format="JPEG", **save_kwargs)
                elif file_format == "webp":
                    try:
                        # For WebP, use piexif for metadata
                        exif_dict = {}

                        if metadata:
                            exif_dict['Exif'] = {piexif.ExifIFD.UserComment: b'UNICODE\0' + metadata.encode('utf-16be')}
                        
                        # Add workflow if needed
                        if embed_workflow and extra_pnginfo is not None:
                            workflow_json = json.dumps(extra_pnginfo["workflow"]) 
                            exif_dict['0th'] = {piexif.ImageIFD.ImageDescription: "Workflow:" + workflow_json}
                            
                        exif_bytes = piexif.dump(exif_dict)
                        save_kwargs["exif"] = exif_bytes
                    except Exception as e:
                        print(f"Error adding EXIF data: {e}")
                    
                    img.save(file_path, format="WEBP", **save_kwargs)
                
                results.append({
                    "filename": file,
                    "subfolder": subfolder,
                    "type": self.type
                })
                
            except Exception as e:
                print(f"Error saving image: {e}")
        
        return results

    def process_image(self, images, id, filename_prefix="ComfyUI", file_format="png", prompt=None, extra_pnginfo=None,
                     lossless_webp=True, quality=100, embed_workflow=False, add_counter_to_filename=True):
        """Process and save image with metadata"""
        # Make sure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # If images is already a list or array of images, do nothing; otherwise, convert to list
        if isinstance(images, (list, np.ndarray)):
            pass
        else:
            # Ensure images is always a list of images
            if len(images.shape) == 3:  # Single image (height, width, channels)
                images = [images]
            else:  # Multiple images (batch, height, width, channels)
                images = [img for img in images]
        
        # Save all images
        results = self.save_images(
            images, 
            filename_prefix, 
            file_format, 
            id,
            prompt, 
            extra_pnginfo,
            lossless_webp,
            quality,
            embed_workflow,
            add_counter_to_filename
        )
        
        return (images,)