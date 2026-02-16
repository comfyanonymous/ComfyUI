from PIL import Image
import json
import sys

try:
    img = Image.open('output/flux_nsfw_00003_.png')
    print(f"Image loaded, info keys: {list(img.info.keys())}")

    prompt = img.info.get('prompt', None)
    workflow = img.info.get('workflow', None)

    if prompt:
        data = json.loads(prompt)
        print(f"Found {len(data)} nodes")
        for node_id, node_data in data.items():
            class_type = node_data.get('class_type', '')
            if class_type == 'KSampler':
                print("=== KSampler settings from SHARP image ===")
                print(json.dumps(node_data.get('inputs', {}), indent=2))
            elif class_type == 'CLIPTextEncodeFlux':
                print(f"=== CLIPTextEncodeFlux node {node_id} ===")
                inputs = node_data.get('inputs', {})
                if 'clip_l' in inputs:
                    print(f"clip_l: {inputs['clip_l'][:100]}...")
    else:
        print("No prompt metadata found")
        print(f"Available keys: {img.info}")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
