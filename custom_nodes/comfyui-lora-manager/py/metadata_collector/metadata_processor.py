import json
import sys
from .constants import IMAGES

# Check if running in standalone mode
standalone_mode = 'nodes' not in sys.modules

from .constants import MODELS, PROMPTS, SAMPLING, LORAS, SIZE, IS_SAMPLER

class MetadataProcessor:
    """Process and format collected metadata"""
    
    @staticmethod
    def find_primary_sampler(metadata, downstream_id=None):
        """
        Find the primary KSampler node that executed before the given downstream node
        
        Parameters:
        - metadata: The workflow metadata
        - downstream_id: Optional ID of a downstream node to help identify the specific primary sampler
        """
        if downstream_id is None:
            if IMAGES in metadata and "first_decode" in metadata[IMAGES]:
                downstream_id = metadata[IMAGES]["first_decode"]["node_id"]
                
        # If we have a downstream_id and execution_order, use it to narrow down potential samplers
        if downstream_id and "execution_order" in metadata:
            execution_order = metadata["execution_order"]
            
            # Find the index of the downstream node in the execution order
            if downstream_id in execution_order:
                downstream_index = execution_order.index(downstream_id)
                
                # Extract all sampler nodes that executed before the downstream node
                candidate_samplers = {}
                for i in range(downstream_index):
                    node_id = execution_order[i]
                    # Use IS_SAMPLER flag to identify true sampler nodes
                    if node_id in metadata.get(SAMPLING, {}) and metadata[SAMPLING][node_id].get(IS_SAMPLER, False):
                        candidate_samplers[node_id] = metadata[SAMPLING][node_id]
                
                # If we found candidate samplers, apply primary sampler logic to these candidates only
                if candidate_samplers:
                    # Collect potential primary samplers based on different criteria
                    custom_advanced_samplers = []
                    advanced_add_noise_samplers = []
                    high_denoise_samplers = []
                    max_denoise = -1
                    high_denoise_id = None
                    
                    # First, check for SamplerCustomAdvanced among candidates
                    prompt = metadata.get("current_prompt")
                    if prompt and prompt.original_prompt:
                        for node_id in candidate_samplers:
                            node_info = prompt.original_prompt.get(node_id, {})
                            if node_info.get("class_type") == "SamplerCustomAdvanced":
                                custom_advanced_samplers.append(node_id)
                    
                    # Next, check for KSamplerAdvanced with add_noise="enable" among candidates
                    for node_id, sampler_info in candidate_samplers.items():
                        parameters = sampler_info.get("parameters", {})
                        add_noise = parameters.get("add_noise")
                        if add_noise == "enable":
                            advanced_add_noise_samplers.append(node_id)
                    
                    # Find the sampler with highest denoise value among candidates
                    for node_id, sampler_info in candidate_samplers.items():
                        parameters = sampler_info.get("parameters", {})
                        denoise = parameters.get("denoise")
                        if denoise is not None and denoise > max_denoise:
                            max_denoise = denoise
                            high_denoise_id = node_id
                    
                    if high_denoise_id:
                        high_denoise_samplers.append(high_denoise_id)
                    
                    # Combine all potential primary samplers
                    potential_samplers = custom_advanced_samplers + advanced_add_noise_samplers + high_denoise_samplers
                    
                    # Find the most recent potential primary sampler (closest to downstream node)
                    for i in range(downstream_index - 1, -1, -1):
                        node_id = execution_order[i]
                        if node_id in potential_samplers:
                            return node_id, candidate_samplers[node_id]
                    
                    # If no potential sampler found from our criteria, return the most recent sampler
                    if candidate_samplers:
                        for i in range(downstream_index - 1, -1, -1):
                            node_id = execution_order[i]
                            if node_id in candidate_samplers:
                                return node_id, candidate_samplers[node_id]
        
        # If no downstream_id provided or no suitable sampler found, fall back to original logic
        primary_sampler = None
        primary_sampler_id = None
        max_denoise = -1
        
        # First, check for SamplerCustomAdvanced
        prompt = metadata.get("current_prompt")
        if prompt and prompt.original_prompt:
            for node_id, node_info in prompt.original_prompt.items():
                if node_info.get("class_type") == "SamplerCustomAdvanced":
                    # Check if the node is in SAMPLING and has IS_SAMPLER flag
                    if node_id in metadata.get(SAMPLING, {}) and metadata[SAMPLING][node_id].get(IS_SAMPLER, False):
                        return node_id, metadata[SAMPLING][node_id]
        
        # Next, check for KSamplerAdvanced with add_noise="enable" using IS_SAMPLER flag
        for node_id, sampler_info in metadata.get(SAMPLING, {}).items():
            # Skip if not marked as a sampler
            if not sampler_info.get(IS_SAMPLER, False):
                continue
                
            parameters = sampler_info.get("parameters", {})
            add_noise = parameters.get("add_noise")
            if add_noise == "enable":
                primary_sampler = sampler_info
                primary_sampler_id = node_id
                break
        
        # If no specialized sampler found, find the sampler with highest denoise value
        if primary_sampler is None:
            for node_id, sampler_info in metadata.get(SAMPLING, {}).items():
                # Skip if not marked as a sampler
                if not sampler_info.get(IS_SAMPLER, False):
                    continue
                    
                parameters = sampler_info.get("parameters", {})
                denoise = parameters.get("denoise")
                if denoise is not None and denoise > max_denoise:
                    max_denoise = denoise
                    primary_sampler = sampler_info
                    primary_sampler_id = node_id
                
        return primary_sampler_id, primary_sampler
    
    @staticmethod
    def trace_node_input(prompt, node_id, input_name, target_class=None, max_depth=10):
        """
        Trace an input connection from a node to find the source node
        
        Parameters:
        - prompt: The prompt object containing node connections
        - node_id: ID of the starting node
        - input_name: Name of the input to trace
        - target_class: Optional class name to search for (e.g., "CLIPTextEncode")
        - max_depth: Maximum depth to follow the node chain to prevent infinite loops
        
        Returns:
        - node_id of the found node, or None if not found
        """
        if not prompt or not prompt.original_prompt or node_id not in prompt.original_prompt:
            return None
            
        # For depth tracking
        current_depth = 0
        
        current_node_id = node_id
        current_input = input_name
        
        # If we're just tracing to origin (no target_class), keep track of the last valid node
        last_valid_node = None
        
        while current_depth < max_depth:
            if current_node_id not in prompt.original_prompt:
                return last_valid_node if not target_class else None
                
            node_inputs = prompt.original_prompt[current_node_id].get("inputs", {})
            if current_input not in node_inputs:
                # We've reached a node without the specified input - this is our origin node
                # if we're not looking for a specific target_class
                return current_node_id if not target_class else None
                
            input_value = node_inputs[current_input]
            # Input connections are formatted as [node_id, output_index]
            if isinstance(input_value, list) and len(input_value) >= 2:
                found_node_id = input_value[0]  # Connected node_id
                
                # If we're looking for a specific node class
                if target_class and prompt.original_prompt[found_node_id].get("class_type") == target_class:
                    return found_node_id
                
                # If we're not looking for a specific class, update the last valid node
                if not target_class:
                    last_valid_node = found_node_id
                
                # Continue tracing through intermediate nodes
                current_node_id = found_node_id
                # For most conditioning nodes, the input we want to follow is named "conditioning"
                if "conditioning" in prompt.original_prompt[current_node_id].get("inputs", {}):
                    current_input = "conditioning"
                else:
                    # If there's no "conditioning" input, return the current node
                    # if we're not looking for a specific target_class
                    return found_node_id if not target_class else None
            else:
                # We've reached a node with no further connections
                return last_valid_node if not target_class else None
            
            current_depth += 1
            
        # If we've reached max depth without finding target_class
        return last_valid_node if not target_class else None
    
    @staticmethod
    def find_primary_checkpoint(metadata):
        """Find the primary checkpoint model in the workflow"""
        if not metadata.get(MODELS):
            return None
            
        # In most workflows, there's only one checkpoint, so we can just take the first one
        for node_id, model_info in metadata.get(MODELS, {}).items():
            if model_info.get("type") == "checkpoint":
                return model_info.get("name")
                
        return None
    
    @staticmethod
    def match_conditioning_to_prompts(metadata, sampler_id):
        """
        Match conditioning objects from a sampler to prompts in metadata
        
        Parameters:
        - metadata: The workflow metadata
        - sampler_id: ID of the sampler node to match
        
        Returns:
        - Dictionary with 'prompt' and 'negative_prompt' if found
        """
        result = {
            "prompt": "",
            "negative_prompt": ""
        }
        
        # Check if we have stored conditioning objects for this sampler
        if sampler_id in metadata.get(PROMPTS, {}) and (
            "pos_conditioning" in metadata[PROMPTS][sampler_id] or 
            "neg_conditioning" in metadata[PROMPTS][sampler_id]):
            
            pos_conditioning = metadata[PROMPTS][sampler_id].get("pos_conditioning")
            neg_conditioning = metadata[PROMPTS][sampler_id].get("neg_conditioning")
            
            # Helper function to recursively find prompt text for a conditioning object
            def find_prompt_text_for_conditioning(conditioning_obj, is_positive=True):
                if conditioning_obj is None:
                    return ""
                    
                # Try to match conditioning objects with those stored by extractors
                for prompt_node_id, prompt_data in metadata[PROMPTS].items():
                    # For nodes with single conditioning output
                    if "conditioning" in prompt_data:
                        if id(prompt_data["conditioning"]) == id(conditioning_obj):
                            return prompt_data.get("text", "")
                    
                    # For nodes with separate pos_conditioning and neg_conditioning outputs (like TSC_EfficientLoader)
                    if is_positive and "positive_encoded" in prompt_data:
                        if id(prompt_data["positive_encoded"]) == id(conditioning_obj):
                            if "positive_text" in prompt_data:
                                return prompt_data["positive_text"]
                            else:
                                orig_conditioning = prompt_data.get("orig_pos_cond", None)
                                if orig_conditioning is not None:
                                    # Recursively find the prompt text for the original conditioning
                                    return find_prompt_text_for_conditioning(orig_conditioning, is_positive=True)
                    
                    if not is_positive and "negative_encoded" in prompt_data:
                        if id(prompt_data["negative_encoded"]) == id(conditioning_obj):
                            if "negative_text" in prompt_data:
                                return prompt_data["negative_text"]
                            else:
                                orig_conditioning = prompt_data.get("orig_neg_cond", None)
                                if orig_conditioning is not None:
                                    # Recursively find the prompt text for the original conditioning
                                    return find_prompt_text_for_conditioning(orig_conditioning, is_positive=False)
                
                return ""
            
            # Find prompt texts using the helper function
            result["prompt"] = find_prompt_text_for_conditioning(pos_conditioning, is_positive=True)
            result["negative_prompt"] = find_prompt_text_for_conditioning(neg_conditioning, is_positive=False)
            
        return result
    
    @staticmethod
    def extract_generation_params(metadata, id=None):
        """
        Extract generation parameters from metadata using node relationships
        
        Parameters:
        - metadata: The workflow metadata
        - id: Optional ID of a downstream node to help identify the specific primary sampler
        """
        params = {
            "prompt": "",
            "negative_prompt": "",
            "seed": None,
            "steps": None,
            "cfg_scale": None,
            "guidance": None,  # Add guidance parameter
            "sampler": None,
            "scheduler": None,
            "checkpoint": None,
            "loras": "",
            "size": None,
            "clip_skip": None
        }
        
        # Get the prompt object for node relationship tracing
        prompt = metadata.get("current_prompt")
        
        # Find the primary KSampler node
        primary_sampler_id, primary_sampler = MetadataProcessor.find_primary_sampler(metadata, id)
        
        # Directly get checkpoint from metadata instead of tracing
        checkpoint = MetadataProcessor.find_primary_checkpoint(metadata)
        if checkpoint:
            params["checkpoint"] = checkpoint
        
        # Check if guidance parameter exists in any sampling node
        for node_id, sampler_info in metadata.get(SAMPLING, {}).items():
            parameters = sampler_info.get("parameters", {})
            if "guidance" in parameters and parameters["guidance"] is not None:
                params["guidance"] = parameters["guidance"]
                break
        
        if primary_sampler:
            # Extract sampling parameters
            sampling_params = primary_sampler.get("parameters", {})
            # Handle both seed and noise_seed
            params["seed"] = sampling_params.get("seed") if sampling_params.get("seed") is not None else sampling_params.get("noise_seed")
            params["steps"] = sampling_params.get("steps")
            params["cfg_scale"] = sampling_params.get("cfg")
            params["sampler"] = sampling_params.get("sampler_name")
            params["scheduler"] = sampling_params.get("scheduler")
            
            if prompt and primary_sampler_id:
                # Check if this is a SamplerCustomAdvanced node
                is_custom_advanced = False
                if prompt.original_prompt and primary_sampler_id in prompt.original_prompt:
                    is_custom_advanced = prompt.original_prompt[primary_sampler_id].get("class_type") == "SamplerCustomAdvanced"
                
                if is_custom_advanced:
                    # For SamplerCustomAdvanced, trace specific inputs
                    
                    # 1. Trace sigmas input to find BasicScheduler
                    scheduler_node_id = MetadataProcessor.trace_node_input(prompt, primary_sampler_id, "sigmas", "BasicScheduler", max_depth=5)
                    if scheduler_node_id and scheduler_node_id in metadata.get(SAMPLING, {}):
                        scheduler_params = metadata[SAMPLING][scheduler_node_id].get("parameters", {})
                        params["steps"] = scheduler_params.get("steps")
                        params["scheduler"] = scheduler_params.get("scheduler")
                    
                    # 2. Trace sampler input to find KSamplerSelect
                    sampler_node_id = MetadataProcessor.trace_node_input(prompt, primary_sampler_id, "sampler", "KSamplerSelect", max_depth=5)
                    if sampler_node_id and sampler_node_id in metadata.get(SAMPLING, {}):
                        sampler_params = metadata[SAMPLING][sampler_node_id].get("parameters", {})
                        params["sampler"] = sampler_params.get("sampler_name")
                    
                    # 3. Trace guider input for CFGGuider and CLIPTextEncode
                    guider_node_id = MetadataProcessor.trace_node_input(prompt, primary_sampler_id, "guider", max_depth=5)
                    if guider_node_id and guider_node_id in prompt.original_prompt:
                        # Check if the guider node is a CFGGuider
                        if prompt.original_prompt[guider_node_id].get("class_type") == "CFGGuider":
                            # Extract cfg value from the CFGGuider
                            if guider_node_id in metadata.get(SAMPLING, {}):
                                cfg_params = metadata[SAMPLING][guider_node_id].get("parameters", {})
                                params["cfg_scale"] = cfg_params.get("cfg")
                            
                            # Find CLIPTextEncode for positive prompt
                            positive_node_id = MetadataProcessor.trace_node_input(prompt, guider_node_id, "positive", "CLIPTextEncode", max_depth=10)
                            if positive_node_id and positive_node_id in metadata.get(PROMPTS, {}):
                                params["prompt"] = metadata[PROMPTS][positive_node_id].get("text", "")
                            
                            # Find CLIPTextEncode for negative prompt
                            negative_node_id = MetadataProcessor.trace_node_input(prompt, guider_node_id, "negative", "CLIPTextEncode", max_depth=10)
                            if negative_node_id and negative_node_id in metadata.get(PROMPTS, {}):
                                params["negative_prompt"] = metadata[PROMPTS][negative_node_id].get("text", "")
                        else:
                            positive_node_id = MetadataProcessor.trace_node_input(prompt, guider_node_id, "conditioning", max_depth=10)
                            if positive_node_id and positive_node_id in metadata.get(PROMPTS, {}):
                                params["prompt"] = metadata[PROMPTS][positive_node_id].get("text", "")
                
                else:
                    # For standard samplers, match conditioning objects to prompts
                    prompt_results = MetadataProcessor.match_conditioning_to_prompts(metadata, primary_sampler_id)
                    params["prompt"] = prompt_results["prompt"]
                    params["negative_prompt"] = prompt_results["negative_prompt"]

                    # If prompts were still not found, fall back to tracing connections
                    if not params["prompt"]:
                        # Original tracing for standard samplers
                        # Trace positive prompt - look specifically for CLIPTextEncode
                        positive_node_id = MetadataProcessor.trace_node_input(prompt, primary_sampler_id, "positive", max_depth=10)
                        if positive_node_id and positive_node_id in metadata.get(PROMPTS, {}):
                            params["prompt"] = metadata[PROMPTS][positive_node_id].get("text", "")
                        else:
                            # If CLIPTextEncode is not found, try to find CLIPTextEncodeFlux
                            positive_flux_node_id = MetadataProcessor.trace_node_input(prompt, primary_sampler_id, "positive", "CLIPTextEncodeFlux", max_depth=10)
                            if positive_flux_node_id and positive_flux_node_id in metadata.get(PROMPTS, {}):
                                params["prompt"] = metadata[PROMPTS][positive_flux_node_id].get("text", "")
                        
                        # Trace negative prompt - look specifically for CLIPTextEncode
                        negative_node_id = MetadataProcessor.trace_node_input(prompt, primary_sampler_id, "negative", max_depth=10)
                        if negative_node_id and negative_node_id in metadata.get(PROMPTS, {}):
                            params["negative_prompt"] = metadata[PROMPTS][negative_node_id].get("text", "")
                
            # Size extraction is same for all sampler types
            # Check if the sampler itself has size information (from latent_image)
            if primary_sampler_id in metadata.get(SIZE, {}):
                width = metadata[SIZE][primary_sampler_id].get("width")
                height = metadata[SIZE][primary_sampler_id].get("height")
                if width and height:
                    params["size"] = f"{width}x{height}"
        
        # Extract LoRAs using the standardized format
        lora_parts = []
        for node_id, lora_info in metadata.get(LORAS, {}).items():
            # Access the lora_list from the standardized format
            lora_list = lora_info.get("lora_list", [])
            for lora in lora_list:
                name = lora.get("name", "unknown")
                strength = lora.get("strength", 1.0)
                lora_parts.append(f"<lora:{name}:{strength}>")
        
        params["loras"] = " ".join(lora_parts)
        
        # Set default clip_skip value
        params["clip_skip"] = "1"  # Common default
        
        return params
    
    @staticmethod
    def to_dict(metadata, id=None):
        """
        Convert extracted metadata to the ComfyUI output.json format
        
        Parameters:
        - metadata: The workflow metadata
        - id: Optional ID of a downstream node to help identify the specific primary sampler
        """              
        if standalone_mode:
            # Return empty dictionary in standalone mode
            return {}
        
        params = MetadataProcessor.extract_generation_params(metadata, id)
        
        # Convert all values to strings to match output.json format
        for key in params:
            if params[key] is not None:
                params[key] = str(params[key])
        
        return params
    
    @staticmethod
    def to_json(metadata, id=None):
        """Convert metadata to JSON string"""
        params = MetadataProcessor.to_dict(metadata, id)
        return json.dumps(params, indent=4)
