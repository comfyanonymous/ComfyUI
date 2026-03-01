
import json

input_file = r"c:\Users\jando\work\ComfyUI\workflow-360-surround-camerawan21-image-to-video-fully-automatic-ZU9mmfjO5XL2MJs3h22b-north_ai-openart.ai.json"
output_file = r"c:\Users\jando\work\ComfyUI\workflow-360-surround-fixed.json"

def deep_clean():
    print(f"Loading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        workflow = json.load(f)

    # 1. Identify Nodes to Remove
    # We remove RH_* nodes and also any node referencing a group
    banned_types = ['RH_Captioner', 'RHHiddenNodes']
    
    # We also want to remove any "Group Node Instance" that might be a wrapper.
    # Since we can't easily identify them by type "workflow>1", we'll just check if type is in banned list for now,
    # relying on the link cleaning to fix the graph if a group node is left behind.
    # Actually, if we delete the group definition, the node instance becomes invalid.
    
    original_nodes = workflow.get('nodes', [])
    valid_nodes = []
    removed_node_ids = set()
    valid_node_ids = set()

    for node in original_nodes:
        if node.get('type') in banned_types:
            print(f"Removing node ID {node.get('id')} ({node.get('type')})")
            removed_node_ids.add(node.get('id'))
        else:
            valid_nodes.append(node)
            valid_node_ids.add(node.get('id'))
    
    workflow['nodes'] = valid_nodes
    print(f"Nodes remaining: {len(valid_nodes)}")

    # 2. Clean Group Definitions
    if 'extra' not in workflow: workflow['extra'] = {}
    if 'groupNodes' in workflow['extra']:
        print("Emptying groupNodes definition...")
        workflow['extra']['groupNodes'] = {}
    
    # 3. Clean Workflow Links
    # Links format: [id, origin_id, origin_slot, target_id, target_slot, type]
    original_links = workflow.get('links', [])
    valid_links = []
    valid_link_ids = set()

    print(f"Scanning {len(original_links)} links...")
    for link in original_links:
        # Link might be [id, ...] or just an object? ComfyUI links are usually arrays.
        # But sometimes they are objects in other formats. Let's assume standard array.
        if not isinstance(link, list):
            continue
            
        link_id = link[0]
        origin_id = link[1]
        target_id = link[3]
        
        if origin_id in valid_node_ids and target_id in valid_node_ids:
            valid_links.append(link)
            valid_link_ids.add(link_id)
        else:
            # print(f"Removing broken link {link_id} (Nodes {origin_id}->{target_id})")
            pass
            
    workflow['links'] = valid_links
    print(f"Links remaining: {len(valid_links)}")

    # 4. Cleanup Node Inputs/Outputs
    # If a node input refers to a link ID that is no longer valid, set it to None
    for node in valid_nodes:
        # Inputs
        inputs = node.get('inputs', [])
        for inp in inputs:
            if 'link' in inp and inp['link'] is not None:
                if inp['link'] not in valid_link_ids:
                    print(f"Detaching input '{inp.get('name')}' on Node {node['id']} (Link {inp['link']} invalid)")
                    inp['link'] = None
        
        # Outputs
        outputs = node.get('outputs', [])
        for out in outputs:
            if 'links' in out and out['links']:
                # Filter valid links
                original_len = len(out['links'])
                out['links'] = [l for l in out['links'] if l in valid_link_ids]
                if len(out['links']) < original_len:
                    print(f"Removed dead output links on Node {node['id']}")

    print(f"Saving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(workflow, f, indent=4)
        
    print("Deep Clean Complete.")

if __name__ == "__main__":
    deep_clean()
