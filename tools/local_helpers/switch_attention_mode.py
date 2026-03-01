
import json

input_file = r"c:\Users\jando\Downloads\workflow-wan22-fixed-models.json"
output_file = r"c:\Users\jando\Downloads\workflow-wan22-fixed-attention.json"

def switch_attention():
    print(f"Loading {input_file}...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            workflow = json.load(f)
    except FileNotFoundError:
        print("Input file not found.")
        return

    nodes = workflow.get('nodes', [])
    updated_count = 0

    for node in nodes:
        widgets = node.get('widgets_values')
        if isinstance(widgets, list):
            new_widgets = []
            changed = False
            for w in widgets:
                if isinstance(w, str) and w == "sageattn":
                    print(f"Node {node['id']}: Switching 'sageattn' to 'sdpa' (Torch Scaled Dot Product Attention)")
                    new_widgets.append("sdpa")
                    changed = True
                else:
                    new_widgets.append(w)
            
            if changed:
                node['widgets_values'] = new_widgets
                updated_count += 1
                
    print(f"Saving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(workflow, f, indent=4, ensure_ascii=False)
        
    print(f"Updated {updated_count} nodes.")

if __name__ == "__main__":
    switch_attention()
