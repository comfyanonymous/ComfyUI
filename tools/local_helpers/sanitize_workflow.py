
import json
import os

input_file = r"c:\Users\jando\work\ComfyUI\workflow-360-surround-camerawan21-image-to-video-fully-automatic-ZU9mmfjO5XL2MJs3h22b-north_ai-openart.ai.json"
output_file = r"c:\Users\jando\work\ComfyUI\workflow-360-surround-cleaned.json"

def sanitize_node(node):
    print(f"Sanitizing node ID {node.get('id')} (was {node.get('type')})")
    node['type'] = "Note"
    node['widgets_values'] = [f"Removed banned node: {node.get('type')}"]
    node['inputs'] = []
    # We keep outputs empty or minimal to avoid connection issues. 
    # Ideally removing connections to this node is best, but ComfyUI handles disconnected lines okay-ish.
    # But Note nodes don't have outputs usually.
    node['outputs'] = [] 
    # Preserve pos and size if possible, or reset
    if 'size' not in node: node['size'] = [200, 100]

def sanitize_workflow():
    print(f"Loading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        workflow = json.load(f)

    banned_types = ['RH_Captioner', 'RHHiddenNodes']
    count = 0

    # 1. Sanitize Main Nodes
    nodes = workflow.get('nodes', [])
    for node in nodes:
        if node.get('type') in banned_types:
            sanitize_node(node)
            count += 1

    # 2. Sanitize Group Nodes
    if 'extra' in workflow and 'groupNodes' in workflow['extra']:
        group_nodes_dict = workflow['extra']['groupNodes']
        for group_id, group_data in group_nodes_dict.items():
            inner_nodes = group_data.get('nodes', [])
            for node in inner_nodes:
                if node.get('type') in banned_types:
                    sanitize_node(node)
                    count += 1
    
    print(f"Sanitized {count} nodes.")
    print(f"Saving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(workflow, f, indent=4) # indent=4 makes it readable but bigger
        
    print("Done.")

if __name__ == "__main__":
    sanitize_workflow()
