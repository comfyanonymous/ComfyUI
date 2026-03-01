
import json
import re

input_file = r"c:\Users\jando\work\ComfyUI\workflow-360-surround-fixed.json"

def remove_group_instances():
    print(f"Loading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        workflow = json.load(f)

    nodes = workflow.get('nodes', [])
    cleaned_nodes = []
    removed_ids = set()

    print(f"Scanning {len(nodes)} nodes for group instances...")
    
    for node in nodes:
        node_type = str(node.get('type', ''))
        # Check for types like "workflow>1", "workflow>2", etc.
        # Or blindly "workflow>" prefix
        if node_type.startswith('workflow>') or 'workflow>' in node_type:
            print(f"Removing Group Instance Node ID {node.get('id')} (Type: {node_type})")
            removed_ids.add(node.get('id'))
        else:
            cleaned_nodes.append(node)
    
    workflow['nodes'] = cleaned_nodes
    print(f"Nodes remaining: {len(cleaned_nodes)}")

    # Also clean links connected to removed nodes
    links = workflow.get('links', [])
    cleaned_links = []
    
    for link in links:
        if isinstance(link, list):
            # [id, origin_id, origin_slot, target_id, target_slot, type]
            if link[1] in removed_ids or link[3] in removed_ids:
                print(f"Removing broken link ID {link[0]}")
                continue
        cleaned_links.append(link)
        
    workflow['links'] = cleaned_links

    print(f"Saving to {input_file}...")
    with open(input_file, 'w', encoding='utf-8') as f:
        json.dump(workflow, f, indent=4)
        
    print("Cleaned Group Instances.")

if __name__ == "__main__":
    remove_group_instances()
