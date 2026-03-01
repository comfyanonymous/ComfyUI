
import json
import os

input_file = r"c:\Users\jando\work\ComfyUI\workflow-360-surround-camerawan21-image-to-video-fully-automatic-ZU9mmfjO5XL2MJs3h22b-north_ai-openart.ai.json"
output_file = r"c:\Users\jando\work\ComfyUI\workflow-360-surround-cleaned.json"

def clean_workflow():
    print(f"Loading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        workflow = json.load(f)

    # 1. Clean Main Nodes
    nodes = workflow.get('nodes', [])
    cleaned_nodes = []
    removed_count = 0
    
    print(f"Scanning {len(nodes)} main nodes...")
    for node in nodes:
        node_type = node.get('type', '')
        # Remove RHHiddenNodes and RH_Captioner from main list
        if node_type in ['RH_Captioner', 'RHHiddenNodes']:
            print(f"Removing main node ID {node.get('id')} of type {node_type}")
            removed_count += 1
            continue
        cleaned_nodes.append(node)
    
    workflow['nodes'] = cleaned_nodes

    # 2. Clean Group Nodes (Hollow out, don't delete key)
    if 'extra' in workflow and 'groupNodes' in workflow['extra']:
        group_nodes_dict = workflow['extra']['groupNodes']
        print(f"Found groupNodes keys: {list(group_nodes_dict.keys())}")
        
        for group_id, group_data in group_nodes_dict.items():
            inner_nodes = group_data.get('nodes', [])
            has_banned = False
            for node in inner_nodes:
                if node.get('type') in ['RH_Captioner', 'RHHiddenNodes']:
                    has_banned = True
                    break
            
            if has_banned:
                print(f"Group {group_id} contains banned nodes. Emptying it.")
                # We clear the nodes list but keep the group structure valid
                group_data['nodes'] = []
                # Also clear links/external to be safe
                group_data['links'] = []
                group_data['external'] = []
                removed_count += 1
    else:
        print("No groupNodes found in extra.")

    print(f"Saving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(workflow, f, indent=4)
        
    print("Done.")

if __name__ == "__main__":
    clean_workflow()
