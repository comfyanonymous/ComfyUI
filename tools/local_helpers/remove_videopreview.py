
import json
import os

input_file = r"c:\Users\jando\work\ComfyUI\workflow-360-surround-fixed.json"

def clean_videopreview():
    print(f"Loading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        workflow = json.load(f)

    nodes = workflow.get('nodes', [])
    modified_count = 0

    for node in nodes:
        if node.get('type') == 'VHS_VideoCombine':
            widgets = node.get('widgets_values')
            if isinstance(widgets, dict):
                if 'videopreview' in widgets:
                    print(f"Removing 'videopreview' from Node ID {node.get('id')} (VHS_VideoCombine)")
                    del widgets['videopreview']
                    modified_count += 1
            elif isinstance(widgets, list):
                # Sometimes it might be in a list? Unlikely for this specific error structure seen in JSON.
                # But if so, we can't easily identify keys. 
                # Given the view_file showed it as a dict, we focus on dict.
                pass
    
    if modified_count > 0:
        print(f"Saving changes to {input_file}...")
        with open(input_file, 'w', encoding='utf-8') as f:
            json.dump(workflow, f, indent=4)
        print("Done.")
    else:
        print("No 'videopreview' keys found to remove.")

if __name__ == "__main__":
    clean_videopreview()
