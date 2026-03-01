
import json

input_file = r"c:\Users\jando\work\ComfyUI\workflow-360-surround-camerawan21-image-to-video-fully-automatic-ZU9mmfjO5XL2MJs3h22b-north_ai-openart.ai.json"

def find_node():
    with open(input_file, 'r', encoding='utf-8') as f:
        workflow = json.load(f)

    group1 = workflow.get('extra', {}).get('groupNodes', {}).get('1')
    if not group1:
        print("Group 1 not found in original file.")
        return

    print("Analyzing Group 1 connections...")
    # External format: [[LinkID, SlotID, Type], ...] usually? Or something similar.
    external = group1.get('external', [])
    print(f"External connections: {external}")
    
    related_link_ids = []
    for item in external:
        # item is usually [internal_id, slot_id, type] or just link id if simplified?
        # Actually in groupNodes external list usually defines the boundary.
        # Let's just collect all integers we see in external linkage structure and check connections.
        pass
        
    # Actually, a better way: The Main Node representing the group will have 'inputs' or 'outputs' 
    # that correspond to the outer links. 
    # But simpler: A Group Node usually has `type` = `workflow>1` or similar. 
    # If I really can't find it by type, I'll search for nodes that have specific coordinates or widgets?
    # No, let's look at the dump of Group 1 again. 
    # It contains "Text Multiline", "RH_Captioner".
    # This group seems to auto-caption an image.
    # It probably takes an IMAGE as input and outputs STRING.
    
    # Let's search for any main node that inputs an IMAGE and outputs a STRING and is NOT one of the known nodes.
    # Or check if there is a node with ID that is NOT in the standard list.
    
    # Let's just print the raw JSON of `extra.groupNodes` to understand the link mapping.
    print(json.dumps(group1, indent=2))

    # Also, let's look for nodes that have `link` values involved with this group.
    # But first, simply seeing the group definition will help.

if __name__ == "__main__":
    find_node()
