
import json
import os

# Target the file the user is actually using
input_file = r"c:\Users\jando\Downloads\workflow-wan22-image-2-video-nsfw-dBafwBOlEq8bTW32Vdwz-in_depth-openart.ai (1).json"
output_file = r"c:\Users\jando\Downloads\workflow-wan22-fixed-models.json"

replacements = {
    # VAE
    "Wan2_1_VAE_bf16.safetensors": "wan_2.1_vae.safetensors",
    
    # Diffusion Models
    "Wan2_2-I2V-A14B-HIGH_fp8_e4m3fn_scaled_KJ.safetensors": "Wan2_1-I2V-14B-480p_fp8_e4m3fn_scaled_KJ.safetensors",
    "Wan2_2-I2V-A14B-LOW_fp8_e4m3fn_scaled_KJ.safetensors": "Wan2_1-I2V-14B-480p_fp8_e4m3fn_scaled_KJ.safetensors",
    
    # T5
    "umt5-xxl-enc-bf16.safetensors": "umt5_xxl_fp16.safetensors"
}

def fix_active_workflow():
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return

    print(f"Loading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        workflow = json.load(f)

    nodes = workflow.get('nodes', [])
    updated_count = 0

    for node in nodes:
        widgets = node.get('widgets_values')
        
        # General widget replacement
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
        
        # Specific fix for T5 if it's missing (Node 11 in user logs)
        if node.get('type') == 'LoadWanVideoT5TextEncoder':
             # T5 node usually has [model_name, device, dtype] key structure
             # If empty or wrong, force it.
             if not widgets:
                 print(f"Node {node['id']} (T5): Setting missing model to umt5_xxl_fp16.safetensors")
                 node['widgets_values'] = ["umt5_xxl_fp16.safetensors", "default", "default"]
                 updated_count += 1
             elif isinstance(widgets, list) and len(widgets) > 0:
                 if widgets[0] not in ["umt5_xxl_fp16.safetensors"]:
                     print(f"Node {node['id']} (T5): Fixing model name")
                     node['widgets_values'][0] = "umt5_xxl_fp16.safetensors"
                     updated_count += 1

    print(f"Saving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(workflow, f, indent=4, ensure_ascii=False)
        
    print(f"Updated {updated_count} nodes.")

if __name__ == "__main__":
    fix_active_workflow()
