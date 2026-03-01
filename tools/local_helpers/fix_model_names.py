
import json

input_file = r"c:\Users\jando\work\ComfyUI\workflow-360-surround-english.json"
output_file = r"c:\Users\jando\work\ComfyUI\workflow-360-surround-english_v2.json"

# Map of Node Type -> Widget Index -> Correct Filename
# Based on error logs and standard ComfyUI structure
# WanVideoModelLoader: inputs usually include 'model' which is a widget value
# WanVideoVAELoader: 'vae_name'
# LoadWanVideoT5TextEncoder: 'model_name'

# Let's try to update universally by searching for the WRONG string and replacing with RIGHT string in widgets_values
replacements = {
    # VAE
    "Wan2_1_VAE_bf16.safetensors": "wan_2.1_vae.safetensors",
    
    # Diffusion Models
    "Wan2_2-I2V-A14B-HIGH_fp8_e4m3fn_scaled_KJ.safetensors": "Wan2_1-I2V-14B-480p_fp8_e4m3fn_scaled_KJ.safetensors",
    "Wan2_2-I2V-A14B-LOW_fp8_e4m3fn_scaled_KJ.safetensors": "Wan2_1-I2V-14B-480p_fp8_e4m3fn_scaled_KJ.safetensors",
    
    # Text Encoder
    # The error said "Required input is missing: model_name". 
    # This might mean the widget value is empty or null.
    # We need to find the node and force set it.
    "umt5-xxl-enc-bf16.safetensors": "umt5_xxl_fp16.safetensors"
}

def fix_models():
    print(f"Loading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        workflow = json.load(f)

    nodes = workflow.get('nodes', [])
    updated_count = 0

    for node in nodes:
        widgets = node.get('widgets_values')
        
        # Strategy 1: String replacement in widgets list
        if isinstance(widgets, list):
            new_widgets = []
            changed = False
            for w in widgets:
                if isinstance(w, str) and w in replacements:
                    print(f"Node {node['id']}: Replaced '{w}' with '{replacements[w]}'")
                    new_widgets.append(replacements[w])
                    changed = True
                else:
                    new_widgets.append(w)
            
            if changed:
                node['widgets_values'] = new_widgets
                updated_count += 1
                
        # Strategy 2: Targeting Specific Nodes that might be missing values (T5)
        # LoadWanVideoT5TextEncoder (Node 11 in error log)
        if node.get('type') == 'LoadWanVideoT5TextEncoder':
            # This node usually takes a model name. 
            # If widgets_values is empty or doesn't have the model, we set it.
            # Assuming the first widget is the model name.
            if not node.get('widgets_values'):
                print(f"Node {node['id']} (T5): Setting missing model name.")
                node['widgets_values'] = ["umt5_xxl_fp16.safetensors"]
                updated_count += 1
            elif isinstance(node['widgets_values'], list):
                # Check if current value is valid, if not update it
                current = node['widgets_values'][0]
                if current not in ["umt5_xxl_fp16.safetensors"]:
                     print(f"Node {node['id']} (T5): Updating T5 model from '{current}' to correct file.")
                     node['widgets_values'][0] = "umt5_xxl_fp16.safetensors"
                     updated_count += 1

    print(f"Saving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(workflow, f, indent=4, ensure_ascii=False)
        
    print(f"Updated {updated_count} nodes.")

if __name__ == "__main__":
    fix_models()
