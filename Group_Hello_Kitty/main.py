import requests
import json
import time
import os

# List of workflow JSON files to be processed
WORKFLOW_FILES = [
    "puteri_1.json",
    "puteri_2.json",
    "puteri_3.json",
    "puteri_4.json",
    "puteri_5.json"
]

# Folder where generated images will be stored or checked
SAVE_DIR = "output_images"

# ComfyUI server address (adjust if different)
COMFYUI_URL = "http://127.0.0.1:8188"

# HTTP request headers
HEADERS = {"Content-Type": "application/json"}

# Load workflow JSON from file
def load_workflow(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Send a workflow to ComfyUI for processing
def send_workflow(workflow_json):
    url = f"{COMFYUI_URL}/prompt"
    response = requests.post(url, headers=HEADERS, json=workflow_json)
    if response.status_code == 200:
        print("✅ Workflow sent.")
        return True
    else:
        print(f"❌ Failed to send workflow: {response.status_code}")
        return False

# Wait for the output image to appear in the folder
def wait_for_image(filename, timeout=60):
    full_path = os.path.join(SAVE_DIR, filename + ".png")
    waited = 0

    # Keep checking every 2 seconds until file appears or timeout
    while not os.path.exists(full_path):
        if waited >= timeout:
            print(f"⚠️ Timeout: {filename}.png not found after {timeout}s")
            return False
        time.sleep(2)
        waited += 2
        print(f"⏳ Waiting for {filename}.png... ({waited}s)")

    print(f"✅ Image {filename}.png detected.")
    return True

# Main script logic
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)  # Create folder if not exists

    # Loop through each workflow file
    for workflow_file in WORKFLOW_FILES:
        print(f"\n🚀 Starting workflow: {workflow_file}")
        workflow = load_workflow(workflow_file)

        # Try to extract the output filename from the SaveImage node
        save_node = next(
            (node for node in workflow["nodes"] if node["type"] == "SaveImage"), None)
        filename = save_node["widgets_values"][0] if save_node else "unknown"

        # Send the workflow to ComfyUI
        sent = send_workflow(workflow)
        if not sent:
            continue

        # Wait for the image to be generated using while loop
        image_ready = wait_for_image(filename)
        if not image_ready:
            print("⚠️ Skipping to next workflow.")

if name == "__main__":
    main()