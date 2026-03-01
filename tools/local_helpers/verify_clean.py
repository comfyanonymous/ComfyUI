
import json

cleaned_file = r"c:\Users\jando\work\ComfyUI\workflow-360-surround-cleaned.json"

def verify():
    with open(cleaned_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    nodes = data.get('nodes', [])
    print(f"Nodes count: {len(nodes)}")
    
    found_banned = False
    for node in nodes:
        if node.get('type') in ['RH_Captioner', 'RHHiddenNodes']:
            print(f"FAIL: Found banned node {node.get('type')} (ID: {node.get('id')}) in main nodes list.")
            found_banned = True
            
    if not found_banned:
        print("PASS: No banned nodes in main list.")

    if 'extra' in data:
        print(f"Extra keys: {list(data['extra'].keys())}")
        if 'groupNodes' in data['extra']:
            print(f"groupNodes keys: {list(data['extra']['groupNodes'].keys())}")
        else:
             print("groupNodes key MISSING from extra.")
    else:
        print("Extra key MISSING.")

if __name__ == "__main__":
    verify()
