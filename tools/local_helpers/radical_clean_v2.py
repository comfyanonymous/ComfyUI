
import json

input_file = r"c:\Users\jando\work\ComfyUI\workflow-360-surround-fixed.json"
output_file = r"c:\Users\jando\work\ComfyUI\workflow-360-surround-fixed.json"

def radical_clean():
    print(f"Loading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        workflow = json.load(f)

    # Completely Nuke 'extra'
    if 'extra' in workflow:
        print(f"Removing 'extra' field (contained {list(workflow['extra'].keys())})")
        del workflow['extra']
    
    # Also ensure no 'groups' field if it exists (old format?)
    if 'groups' in workflow:
        del workflow['groups']

    print(f"Saving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(workflow, f, indent=4)
        
    print("Radical Clean Applied to Fixed File.")

if __name__ == "__main__":
    radical_clean()
