
import json
import os

input_file = r"c:\Users\jando\work\ComfyUI\workflow-360-surround-camerawan21-image-to-video-fully-automatic-ZU9mmfjO5XL2MJs3h22b-north_ai-openart.ai.json"
output_file = r"c:\Users\jando\work\ComfyUI\workflow-360-surround-radical.json"

def radical_clean():
    print(f"Loading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        workflow = json.load(f)

    # 1. Strip Top Level Metadata
    # We only keep 'nodes' and 'links', and maybe 'version'
    keys_to_keep = ['nodes', 'links', 'version']
    new_workflow = {k: workflow.get(k) for k in keys_to_keep if k in workflow}
    
    # Explicitly ensure 'extra' is GONE
    if 'extra' in new_workflow:
        del new_workflow['extra']
    
    print("Stripped top-level metadata (extra, ds, grouplinks, etc).")

    # 2. Filter Nodes (Banned Types)
    original_nodes = new_workflow.get('nodes', [])
    cleaned_nodes = []
    banned_types = ['RH_Captioner', 'RHHiddenNodes']
    removed_ids = set()

    for node in original_nodes:
        if node.get('type') in banned_types:
            print(f"Removing banned node ID {node.get('id')} ({node.get('type')})")
            removed_ids.add(node.get('id'))
        else:
            cleaned_nodes.append(node)
    
    new_workflow['nodes'] = cleaned_nodes
    valid_node_ids = set(n['id'] for n in cleaned_nodes)

    # 3. Filter Links
    # Remove links involving removed nodes
    original_links = new_workflow.get('links', [])
    cleaned_links = []
    
    for link in original_links:
        if not isinstance(link, list): continue
        # link format: [id, origin_id, origin_slot, target_id, target_slot, type]
        if link[1] in valid_node_ids and link[3] in valid_node_ids:
            cleaned_links.append(link)
    
    new_workflow['links'] = cleaned_links
    
    print(f"Saving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(new_workflow, f, indent=4)
        
    print("Radical Clean Complete.")

if __name__ == "__main__":
    radical_clean()
