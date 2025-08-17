import os

from .constants import MODELS, PROMPTS, SAMPLING, LORAS, SIZE, IMAGES, IS_SAMPLER


class NodeMetadataExtractor:
    """Base class for node-specific metadata extraction"""
    
    @staticmethod
    def extract(node_id, inputs, outputs, metadata):
        """Extract metadata from node inputs/outputs"""
        pass
        
    @staticmethod
    def update(node_id, outputs, metadata):
        """Update metadata with node outputs after execution"""
        pass
        
class GenericNodeExtractor(NodeMetadataExtractor):
    """Default extractor for nodes without specific handling"""
    @staticmethod
    def extract(node_id, inputs, outputs, metadata):
        pass
        
class CheckpointLoaderExtractor(NodeMetadataExtractor):
    @staticmethod
    def extract(node_id, inputs, outputs, metadata):
        if not inputs or "ckpt_name" not in inputs:
            return
            
        model_name = inputs.get("ckpt_name")
        if model_name:
            metadata[MODELS][node_id] = {
                "name": model_name,
                "type": "checkpoint",
                "node_id": node_id
            }

class TSCCheckpointLoaderExtractor(NodeMetadataExtractor):
    @staticmethod
    def extract(node_id, inputs, outputs, metadata):
        if not inputs or "ckpt_name" not in inputs:
            return
            
        model_name = inputs.get("ckpt_name")
        if model_name:
            metadata[MODELS][node_id] = {
                "name": model_name,
                "type": "checkpoint",
                "node_id": node_id
            }

        # For loader node has lora_stack input, like Efficient Loader from Efficient Nodes
        active_loras = []
        
        # Process lora_stack if available
        if "lora_stack" in inputs:
            lora_stack = inputs.get("lora_stack", [])
            for lora_path, model_strength, clip_strength in lora_stack:
                # Extract lora name from path (following the format in lora_loader.py)
                lora_name = os.path.splitext(os.path.basename(lora_path))[0]
                active_loras.append({
                    "name": lora_name,
                    "strength": model_strength
                })
        
        if active_loras:
            metadata[LORAS][node_id] = {
                "lora_list": active_loras,
                "node_id": node_id
            }
        
        # Extract positive and negative prompt text if available
        positive_text = inputs.get("positive", "")
        negative_text = inputs.get("negative", "")
        
        if positive_text or negative_text:
            if node_id not in metadata[PROMPTS]:
                metadata[PROMPTS][node_id] = {"node_id": node_id}
            
            # Store both positive and negative text
            metadata[PROMPTS][node_id]["positive_text"] = positive_text
            metadata[PROMPTS][node_id]["negative_text"] = negative_text
            
    @staticmethod
    def update(node_id, outputs, metadata):
        # Handle conditioning outputs from TSC_EfficientLoader
        # outputs is a list with [(model, positive_encoded, negative_encoded, {"samples":latent}, vae, clip, dependencies,)]
        if outputs and isinstance(outputs, list) and len(outputs) > 0:
            first_output = outputs[0]
            if isinstance(first_output, tuple) and len(first_output) >= 3:
                positive_conditioning = first_output[1]
                negative_conditioning = first_output[2]
                
                # Save both conditioning objects in metadata
                if node_id not in metadata[PROMPTS]:
                    metadata[PROMPTS][node_id] = {"node_id": node_id}
                    
                metadata[PROMPTS][node_id]["positive_encoded"] = positive_conditioning
                metadata[PROMPTS][node_id]["negative_encoded"] = negative_conditioning

class CLIPTextEncodeExtractor(NodeMetadataExtractor):
    @staticmethod
    def extract(node_id, inputs, outputs, metadata):
        if not inputs or "text" not in inputs:
            return
            
        text = inputs.get("text", "")
        metadata[PROMPTS][node_id] = {
            "text": text,
            "node_id": node_id
        }

    @staticmethod
    def update(node_id, outputs, metadata):
        if outputs and isinstance(outputs, list) and len(outputs) > 0:
            if isinstance(outputs[0], tuple) and len(outputs[0]) > 0:
                conditioning = outputs[0][0]
                metadata[PROMPTS][node_id]["conditioning"] = conditioning

# Base Sampler Extractor to reduce code redundancy
class BaseSamplerExtractor(NodeMetadataExtractor):
    """Base extractor for sampler nodes with common functionality"""
    @staticmethod
    def extract_sampling_params(node_id, inputs, metadata, param_keys):
        """Extract sampling parameters from inputs"""
        sampling_params = {}
        for key in param_keys:
            if key in inputs:
                sampling_params[key] = inputs[key]
                
        metadata[SAMPLING][node_id] = {
            "parameters": sampling_params,
            "node_id": node_id,
            IS_SAMPLER: True  # Add sampler flag
        }
    
    @staticmethod
    def extract_conditioning(node_id, inputs, metadata):
        """Extract conditioning objects from inputs"""
        # Store the conditioning objects directly in metadata for later matching
        pos_conditioning = inputs.get("positive", None)
        neg_conditioning = inputs.get("negative", None)

        # Save conditioning objects in metadata for later matching
        if pos_conditioning is not None or neg_conditioning is not None:
            if node_id not in metadata[PROMPTS]:
                metadata[PROMPTS][node_id] = {"node_id": node_id}
            
            metadata[PROMPTS][node_id]["pos_conditioning"] = pos_conditioning
            metadata[PROMPTS][node_id]["neg_conditioning"] = neg_conditioning
    
    @staticmethod
    def extract_latent_dimensions(node_id, inputs, metadata):
        """Extract dimensions from latent image"""
        # Extract latent image dimensions if available
        if "latent_image" in inputs and inputs["latent_image"] is not None:
            latent = inputs["latent_image"]
            if isinstance(latent, dict) and "samples" in latent:
                # Extract dimensions from latent tensor
                samples = latent["samples"]
                if hasattr(samples, "shape") and len(samples.shape) >= 3:
                    # Correct shape interpretation: [batch_size, channels, height/8, width/8]
                    # Multiply by 8 to get actual pixel dimensions
                    height = int(samples.shape[2] * 8)
                    width = int(samples.shape[3] * 8)
                    
                    if SIZE not in metadata:
                        metadata[SIZE] = {}
                        
                    metadata[SIZE][node_id] = {
                        "width": width,
                        "height": height,
                        "node_id": node_id
                    }
        
class SamplerExtractor(BaseSamplerExtractor):
    @staticmethod
    def extract(node_id, inputs, outputs, metadata):
        if not inputs:
            return
            
        # Extract common sampling parameters
        BaseSamplerExtractor.extract_sampling_params(
            node_id, inputs, metadata, 
            ["seed", "steps", "cfg", "sampler_name", "scheduler", "denoise"]
        )
        
        # Extract conditioning objects
        BaseSamplerExtractor.extract_conditioning(node_id, inputs, metadata)
        
        # Extract latent dimensions
        BaseSamplerExtractor.extract_latent_dimensions(node_id, inputs, metadata)

class KSamplerAdvancedExtractor(BaseSamplerExtractor):
    @staticmethod
    def extract(node_id, inputs, outputs, metadata):
        if not inputs:
            return
            
        # Extract common sampling parameters
        BaseSamplerExtractor.extract_sampling_params(
            node_id, inputs, metadata, 
            ["noise_seed", "steps", "cfg", "sampler_name", "scheduler", "add_noise"]
        )
        
        # Extract conditioning objects
        BaseSamplerExtractor.extract_conditioning(node_id, inputs, metadata)
        
        # Extract latent dimensions
        BaseSamplerExtractor.extract_latent_dimensions(node_id, inputs, metadata)

class KSamplerBasicPipeExtractor(BaseSamplerExtractor):
    """Extractor for KSamplerBasicPipe and KSampler_inspire_pipe nodes"""
    @staticmethod
    def extract(node_id, inputs, outputs, metadata):
        if not inputs:
            return
            
        # Extract common sampling parameters
        BaseSamplerExtractor.extract_sampling_params(
            node_id, inputs, metadata, 
            ["seed", "steps", "cfg", "sampler_name", "scheduler", "denoise"]
        )
        
        # Extract conditioning objects from basic_pipe
        if "basic_pipe" in inputs and inputs["basic_pipe"] is not None:
            basic_pipe = inputs["basic_pipe"]
            # Typically, basic_pipe structure is (model, clip, vae, positive, negative)
            if isinstance(basic_pipe, tuple) and len(basic_pipe) >= 5:
                pos_conditioning = basic_pipe[3]  # positive is at index 3
                neg_conditioning = basic_pipe[4]  # negative is at index 4
                
                # Save conditioning objects in metadata
                if node_id not in metadata[PROMPTS]:
                    metadata[PROMPTS][node_id] = {"node_id": node_id}
                
                metadata[PROMPTS][node_id]["pos_conditioning"] = pos_conditioning
                metadata[PROMPTS][node_id]["neg_conditioning"] = neg_conditioning
        
        # Extract latent dimensions
        BaseSamplerExtractor.extract_latent_dimensions(node_id, inputs, metadata)

class KSamplerAdvancedBasicPipeExtractor(BaseSamplerExtractor):
    """Extractor for KSamplerAdvancedBasicPipe nodes"""
    @staticmethod
    def extract(node_id, inputs, outputs, metadata):
        if not inputs:
            return
            
        # Extract common sampling parameters
        BaseSamplerExtractor.extract_sampling_params(
            node_id, inputs, metadata, 
            ["noise_seed", "steps", "cfg", "sampler_name", "scheduler", "add_noise"]
        )
        
        # Extract conditioning objects from basic_pipe
        if "basic_pipe" in inputs and inputs["basic_pipe"] is not None:
            basic_pipe = inputs["basic_pipe"]
            # Typically, basic_pipe structure is (model, clip, vae, positive, negative)
            if isinstance(basic_pipe, tuple) and len(basic_pipe) >= 5:
                pos_conditioning = basic_pipe[3]  # positive is at index 3
                neg_conditioning = basic_pipe[4]  # negative is at index 4
                
                # Save conditioning objects in metadata
                if node_id not in metadata[PROMPTS]:
                    metadata[PROMPTS][node_id] = {"node_id": node_id}
                
                metadata[PROMPTS][node_id]["pos_conditioning"] = pos_conditioning
                metadata[PROMPTS][node_id]["neg_conditioning"] = neg_conditioning
        
        # Extract latent dimensions
        BaseSamplerExtractor.extract_latent_dimensions(node_id, inputs, metadata)

class TSCSamplerBaseExtractor(NodeMetadataExtractor):
    @staticmethod
    def extract(node_id, inputs, outputs, metadata):
        # Store vae_decode setting for later use in update
        if inputs and "vae_decode" in inputs:
            if SAMPLING not in metadata:
                metadata[SAMPLING] = {}
                
            if node_id not in metadata[SAMPLING]:
                metadata[SAMPLING][node_id] = {"parameters": {}, "node_id": node_id}
                
            # Store the vae_decode setting
            metadata[SAMPLING][node_id]["vae_decode"] = inputs["vae_decode"]

    @staticmethod
    def update(node_id, outputs, metadata):
        # Check if vae_decode was set to "true"
        should_save_image = True
        if SAMPLING in metadata and node_id in metadata[SAMPLING]:
            vae_decode = metadata[SAMPLING][node_id].get("vae_decode")
            if vae_decode is not None:
                should_save_image = (vae_decode == "true")
        
        # Skip image saving if vae_decode isn't "true"
        if not should_save_image:
            return
        
        # Ensure IMAGES category exists
        if IMAGES not in metadata:
            metadata[IMAGES] = {}
        
        # Extract output_images from the TSC sampler format
        # outputs = [{"ui": {"images": preview_images}, "result": result}]
        # where result = (original_model, original_positive, original_negative, latent_list, optional_vae, output_images,)
        if outputs and isinstance(outputs, list) and len(outputs) > 0:
            # Get the first item in the list
            output_item = outputs[0]
            if isinstance(output_item, dict) and "result" in output_item:
                result = output_item["result"]
                if isinstance(result, tuple) and len(result) >= 6:
                    # The output_images is the last element in the result tuple
                    output_images = (result[5],)
                    
                    # Save image data under node ID index to be captured by caching mechanism
                    metadata[IMAGES][node_id] = {
                    "node_id": node_id,
                    "image": output_images
                    }
                    
                    # Only set first_decode if it hasn't been recorded yet
                    if "first_decode" not in metadata[IMAGES]:
                        metadata[IMAGES]["first_decode"] = metadata[IMAGES][node_id]

class TSCKSamplerExtractor(SamplerExtractor, TSCSamplerBaseExtractor):
    @staticmethod
    def extract(node_id, inputs, outputs, metadata):
        # Call parent extract methods
        SamplerExtractor.extract(node_id, inputs, outputs, metadata)
        TSCSamplerBaseExtractor.extract(node_id, inputs, outputs, metadata)

    # Update method is inherited from TSCSamplerBaseExtractor


class TSCKSamplerAdvancedExtractor(KSamplerAdvancedExtractor, TSCSamplerBaseExtractor):
    @staticmethod
    def extract(node_id, inputs, outputs, metadata):
        # Call parent extract methods
        KSamplerAdvancedExtractor.extract(node_id, inputs, outputs, metadata)
        TSCSamplerBaseExtractor.extract(node_id, inputs, outputs, metadata)

    # Update method is inherited from TSCSamplerBaseExtractor

class LoraLoaderExtractor(NodeMetadataExtractor):
    @staticmethod
    def extract(node_id, inputs, outputs, metadata):
        if not inputs or "lora_name" not in inputs:
            return
            
        lora_name = inputs.get("lora_name")
        # Extract base filename without extension from path
        lora_name = os.path.splitext(os.path.basename(lora_name))[0]
        strength_model = round(float(inputs.get("strength_model", 1.0)), 2)
        
        # Use the standardized format with lora_list
        metadata[LORAS][node_id] = {
            "lora_list": [
                {
                    "name": lora_name,
                    "strength": strength_model
                }
            ],
            "node_id": node_id
        }

class ImageSizeExtractor(NodeMetadataExtractor):
    @staticmethod
    def extract(node_id, inputs, outputs, metadata):
        if not inputs:
            return
        
        width = inputs.get("width", 512)
        height = inputs.get("height", 512)
        
        if SIZE not in metadata:
            metadata[SIZE] = {}
            
        metadata[SIZE][node_id] = {
            "width": width,
            "height": height,
            "node_id": node_id
        }

class LoraLoaderManagerExtractor(NodeMetadataExtractor):
    @staticmethod
    def extract(node_id, inputs, outputs, metadata):
        if not inputs:
            return
        
        active_loras = []
        
        # Process lora_stack if available
        if "lora_stack" in inputs:
            lora_stack = inputs.get("lora_stack", [])
            for lora_path, model_strength, clip_strength in lora_stack:
                # Extract lora name from path (following the format in lora_loader.py)
                lora_name = os.path.splitext(os.path.basename(lora_path))[0]
                active_loras.append({
                    "name": lora_name,
                    "strength": model_strength
                })
        
        # Process loras from inputs
        if "loras" in inputs:
            loras_data = inputs.get("loras", [])
            
            # Handle new format: {'loras': {'__value__': [...]}} 
            if isinstance(loras_data, dict) and '__value__' in loras_data:
                loras_list = loras_data['__value__']
            # Handle old format: {'loras': [...]}
            elif isinstance(loras_data, list):
                loras_list = loras_data
            else:
                loras_list = []
                
            # Filter for active loras
            for lora in loras_list:
                if isinstance(lora, dict) and lora.get("active", True) and not lora.get("_isDummy", False):
                    active_loras.append({
                        "name": lora.get("name", ""),
                        "strength": float(lora.get("strength", 1.0))
                    })
        
        if active_loras:
            metadata[LORAS][node_id] = {
                "lora_list": active_loras,
                "node_id": node_id
            }

class FluxGuidanceExtractor(NodeMetadataExtractor):
    @staticmethod
    def extract(node_id, inputs, outputs, metadata):
        if not inputs or "guidance" not in inputs:
            return
            
        guidance_value = inputs.get("guidance")
        
        # Store the guidance value in SAMPLING category
        if node_id not in metadata[SAMPLING]:
            metadata[SAMPLING][node_id] = {"parameters": {}, "node_id": node_id}
            
        metadata[SAMPLING][node_id]["parameters"]["guidance"] = guidance_value

class UNETLoaderExtractor(NodeMetadataExtractor):
    @staticmethod
    def extract(node_id, inputs, outputs, metadata):
        if not inputs or "unet_name" not in inputs:
            return
            
        model_name = inputs.get("unet_name")
        if model_name:
            metadata[MODELS][node_id] = {
                "name": model_name,
                "type": "checkpoint",
                "node_id": node_id
            }

class VAEDecodeExtractor(NodeMetadataExtractor):
    @staticmethod
    def extract(node_id, inputs, outputs, metadata):
        pass
        
    @staticmethod
    def update(node_id, outputs, metadata):
        # Ensure IMAGES category exists
        if IMAGES not in metadata:
            metadata[IMAGES] = {}
            
        # Save image data under node ID index to be captured by caching mechanism
        metadata[IMAGES][node_id] = {
            "node_id": node_id,
            "image": outputs
        }
        
        # Only set first_decode if it hasn't been recorded yet
        if "first_decode" not in metadata[IMAGES]:
            metadata[IMAGES]["first_decode"] = metadata[IMAGES][node_id]

class KSamplerSelectExtractor(NodeMetadataExtractor):
    @staticmethod
    def extract(node_id, inputs, outputs, metadata):
        if not inputs or "sampler_name" not in inputs:
            return
            
        sampling_params = {}
        if "sampler_name" in inputs:
            sampling_params["sampler_name"] = inputs["sampler_name"]
                
        metadata[SAMPLING][node_id] = {
            "parameters": sampling_params,
            "node_id": node_id,
            IS_SAMPLER: False  # Mark as non-primary sampler
        }

class BasicSchedulerExtractor(NodeMetadataExtractor):
    @staticmethod
    def extract(node_id, inputs, outputs, metadata):
        if not inputs:
            return
            
        sampling_params = {}
        for key in ["scheduler", "steps", "denoise"]:
            if key in inputs:
                sampling_params[key] = inputs[key]
                
        metadata[SAMPLING][node_id] = {
            "parameters": sampling_params,
            "node_id": node_id,
            IS_SAMPLER: False  # Mark as non-primary sampler
        }

class SamplerCustomAdvancedExtractor(BaseSamplerExtractor):
    @staticmethod
    def extract(node_id, inputs, outputs, metadata):
        if not inputs:
            return
            
        sampling_params = {}

        # Handle noise.seed as seed
        if "noise" in inputs and inputs["noise"] is not None and hasattr(inputs["noise"], "seed"):
            noise = inputs["noise"]
            sampling_params["seed"] = noise.seed
                
        metadata[SAMPLING][node_id] = {
            "parameters": sampling_params,
            "node_id": node_id,
            IS_SAMPLER: True  # Add sampler flag
        }
        
        # Extract latent dimensions
        BaseSamplerExtractor.extract_latent_dimensions(node_id, inputs, metadata)

import json

class CLIPTextEncodeFluxExtractor(NodeMetadataExtractor):
    @staticmethod
    def extract(node_id, inputs, outputs, metadata):
        if not inputs or "clip_l" not in inputs or "t5xxl" not in inputs:
            return
            
        clip_l_text = inputs.get("clip_l", "")
        t5xxl_text = inputs.get("t5xxl", "")
        
        # If both are empty, use empty string
        if not clip_l_text and not t5xxl_text:
            combined_text = ""
        # If one is empty, use the non-empty one
        elif not clip_l_text:
            combined_text = t5xxl_text
        elif not t5xxl_text:
            combined_text = clip_l_text
        # If both have content, use JSON format
        else:
            combined_text = json.dumps({
                "T5": t5xxl_text,
                "CLIP-L": clip_l_text
            })
        
        metadata[PROMPTS][node_id] = {
            "text": combined_text,
            "node_id": node_id
        }
        
        # Extract guidance value if available
        if "guidance" in inputs:
            guidance_value = inputs.get("guidance")
            
            # Store the guidance value in SAMPLING category
            if SAMPLING not in metadata:
                metadata[SAMPLING] = {}
                
            if node_id not in metadata[SAMPLING]:
                metadata[SAMPLING][node_id] = {"parameters": {}, "node_id": node_id}
                
            metadata[SAMPLING][node_id]["parameters"]["guidance"] = guidance_value

    @staticmethod
    def update(node_id, outputs, metadata):
        if outputs and isinstance(outputs, list) and len(outputs) > 0:
            if isinstance(outputs[0], tuple) and len(outputs[0]) > 0:
                conditioning = outputs[0][0]
                metadata[PROMPTS][node_id]["conditioning"] = conditioning

class CFGGuiderExtractor(NodeMetadataExtractor):
    @staticmethod
    def extract(node_id, inputs, outputs, metadata):
        if not inputs or "cfg" not in inputs:
            return
            
        cfg_value = inputs.get("cfg")
        
        # Store the cfg value in SAMPLING category
        if SAMPLING not in metadata:
            metadata[SAMPLING] = {}
            
        if node_id not in metadata[SAMPLING]:
            metadata[SAMPLING][node_id] = {"parameters": {}, "node_id": node_id}
            
        metadata[SAMPLING][node_id]["parameters"]["cfg"] = cfg_value

class CR_ApplyControlNetStackExtractor(NodeMetadataExtractor):
    @staticmethod
    def extract(node_id, inputs, outputs, metadata):
        if not inputs:
            return
            
        # Save the original conditioning inputs
        base_positive = inputs.get("base_positive")
        base_negative = inputs.get("base_negative")
        
        if base_positive is not None or base_negative is not None:
            if node_id not in metadata[PROMPTS]:
                metadata[PROMPTS][node_id] = {"node_id": node_id}
            
            metadata[PROMPTS][node_id]["orig_pos_cond"] = base_positive
            metadata[PROMPTS][node_id]["orig_neg_cond"] = base_negative

    @staticmethod
    def update(node_id, outputs, metadata):
        # Extract transformed conditionings from outputs
        # outputs structure: [(base_positive, base_negative, show_help, )]
        if outputs and isinstance(outputs, list) and len(outputs) > 0:
            first_output = outputs[0]
            if isinstance(first_output, tuple) and len(first_output) >= 2:
                transformed_positive = first_output[0]
                transformed_negative = first_output[1]
                
                # Save transformed conditioning objects in metadata
                if node_id not in metadata[PROMPTS]:
                    metadata[PROMPTS][node_id] = {"node_id": node_id}
                    
                metadata[PROMPTS][node_id]["positive_encoded"] = transformed_positive
                metadata[PROMPTS][node_id]["negative_encoded"] = transformed_negative

# Registry of node-specific extractors
# Keys are node class names
NODE_EXTRACTORS = {
    # Sampling
    "KSampler": SamplerExtractor,
    "KSamplerAdvanced": KSamplerAdvancedExtractor,
    "SamplerCustomAdvanced": SamplerCustomAdvancedExtractor,
    "TSC_KSampler": TSCKSamplerExtractor,   # Efficient Nodes
    "TSC_KSamplerAdvanced": TSCKSamplerAdvancedExtractor,  # Efficient Nodes
    "KSamplerBasicPipe": KSamplerBasicPipeExtractor,    # comfyui-impact-pack
    "KSamplerAdvancedBasicPipe": KSamplerAdvancedBasicPipeExtractor,    # comfyui-impact-pack
    "KSampler_inspire_pipe": KSamplerBasicPipeExtractor,    # comfyui-inspire-pack
    "KSamplerAdvanced_inspire_pipe": KSamplerAdvancedBasicPipeExtractor,  # comfyui-inspire-pack
    # Sampling Selectors
    "KSamplerSelect": KSamplerSelectExtractor,  # Add KSamplerSelect
    "BasicScheduler": BasicSchedulerExtractor,  # Add BasicScheduler
    # Loaders
    "CheckpointLoaderSimple": CheckpointLoaderExtractor,
    "comfyLoader": CheckpointLoaderExtractor,  # easy comfyLoader
    "TSC_EfficientLoader": TSCCheckpointLoaderExtractor,  # Efficient Nodes
    "UNETLoader": UNETLoaderExtractor,          # Updated to use dedicated extractor
    "UnetLoaderGGUF": UNETLoaderExtractor,  # Updated to use dedicated extractor
    "LoraLoader": LoraLoaderExtractor,
    "LoraManagerLoader": LoraLoaderManagerExtractor,
    # Conditioning
    "CLIPTextEncode": CLIPTextEncodeExtractor,
    "CLIPTextEncodeFlux": CLIPTextEncodeFluxExtractor,  # Add CLIPTextEncodeFlux
    "WAS_Text_to_Conditioning": CLIPTextEncodeExtractor,
    "AdvancedCLIPTextEncode": CLIPTextEncodeExtractor,  # From https://github.com/BlenderNeko/ComfyUI_ADV_CLIP_emb
    "smZ_CLIPTextEncode": CLIPTextEncodeExtractor,  # From https://github.com/shiimizu/ComfyUI_smZNodes
    "CR_ApplyControlNetStack": CR_ApplyControlNetStackExtractor,  # Add CR_ApplyControlNetStack
    # Latent
    "EmptyLatentImage": ImageSizeExtractor,
    # Flux
    "FluxGuidance": FluxGuidanceExtractor,      # Add FluxGuidance
    "CFGGuider": CFGGuiderExtractor,            # Add CFGGuider
    # Image
    "VAEDecode": VAEDecodeExtractor,  # Added VAEDecode extractor
    # Add other nodes as needed
}
