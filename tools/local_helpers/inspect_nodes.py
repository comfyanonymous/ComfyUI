
import json

input_file = r"c:\Users\jando\work\ComfyUI\workflow-360-surround-camerawan21-image-to-video-fully-automatic-ZU9mmfjO5XL2MJs3h22b-north_ai-openart.ai.json"
output_dump = r"c:\Users\jando\work\ComfyUI\nodes_dump.txt"

def dump_nodes():
    with open(input_file, 'r', encoding='utf-8') as f:
        workflow = json.load(f)

    nodes = workflow.get('nodes', [])
    
    with open(output_dump, 'w', encoding='utf-8') as out:
        for node in nodes:
            out.write(f"ID: {node.get('id')}\n")
            out.write(f"Type: {node.get('type')}\n")
            out.write(f"Full: {json.dumps(node, ensure_ascii=False)}\n")
            out.write("-" * 40 + "\n")
            
    print(f"Dumped {len(nodes)} nodes to {output_dump}")

if __name__ == "__main__":
    dump_nodes()
