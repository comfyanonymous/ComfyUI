import uuid
import json
import urllib.request
import urllib.parse
from PIL import Image
from websocket import WebSocket #NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import io
import requests
import time
import os
import subprocess

APP_NAME = os.getenv('APP_NAME') # Name of the application
API_COMMAND_LINE = os.getenv('API_COMMAND_LINE') # Command line to start the API server
API_URL = os.getenv('API_URL')  # URL of the API server
INSTANCE_IDENTIFIER = APP_NAME+'-'+str(uuid.uuid4()) # Unique identifier for this instance of the worker
TEST_PAYLOAD = os.getenv('TEST_PAYLOAD')

class ComfyConnector:
    _instance = None
    _process = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ComfyConnector, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.server_address = API_URL
            self.client_id = INSTANCE_IDENTIFIER
            self.ws = WebSocket()
            self.ws.connect(f"ws://{self.server_address}/ws?clientId={self.client_id}")
            self.initialized = True
            self.start_api()
    
    def start_api(self): # This method is used to start the API server
        api_command_line = API_COMMAND_LINE
        if self._process is None or self._process.poll() is not None: # Check if the process is not running or has terminated for some reason
            self._process = subprocess.Popen(api_command_line.split())
            print("API process started with PID:", self._process.pid)
            while not self.is_api_running(): # Block execution until the API server is running
                time.sleep(0.5)  # Wait for 0.5 seconds before checking again

    def is_api_running(self): # This method is used to check if the API server is running
        test_payload = TEST_PAYLOAD
        try:
            response = requests.get(API_URL)
            if response.status_code == 200: # Check if the API server tells us it's running by returning a 200 status code
                test_image = self.generate_images(payload=test_payload)
                if test_image:  # this ensures that the API server is actually running and not just the web server
                    return True
                return False
        except Exception as e:
            print("API not running:", e)
            return False

    def kill_api(self): # This method is used to kill the API server
        if self._process is not None and self._process.poll() is None:
            self._process.kill()
            self._process = None
            print("API process killed")

    def get_history(self, prompt_id): # This method is used to retrieve the history of a prompt from the API server
        with urllib.request.urlopen(f"http://{self.server_address}/history/{prompt_id}") as response:
            return json.loads(response.read())

    def get_image(self, filename, subfolder, folder_type): # This method is used to retrieve an image from the API server
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)
        with urllib.request.urlopen(f"http://{self.server_address}/view?{url_values}") as response:
            return response.read()

    def queue_prompt(self, prompt): # This method is used to queue a prompt for execution
        p = {"prompt": prompt, "client_id": self.client_id}
        data = json.dumps(p).encode('utf-8')
        print("Sending data:", data)  # Print the data for debugging
        headers = {'Content-Type': 'application/json'}  # Set Content-Type header
        req = urllib.request.Request(f"http://{self.server_address}/prompt", data=data, headers=headers)
        return json.loads(urllib.request.urlopen(req).read())

    def generate_images(self, prompt): # This method is used to generate images from a prompt and is the main method of this class
        prompt_id = self.queue_prompt(prompt)['prompt_id']
        while True:
            out = self.ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message['type'] == 'executing':
                    data = message['data']
                    if data['node'] is None and data['prompt_id'] == prompt_id:
                        break
        address = self.find_output_node(prompt)
        history = self.get_history(prompt_id)[prompt_id]
        filenames = eval(f"history['outputs']{address}")['images']  # Extract all images
        images = []
        for img_info in filenames:
            filename = img_info['filename']
            subfolder = img_info['subfolder']
            folder_type = img_info['type']
            image_data = self.get_image(filename, subfolder, folder_type)
            image_file = io.BytesIO(image_data)
            image = Image.open(image_file)
            images.append(image)
        return images

    def upload_image(self, filepath, subfolder=None, folder_type=None, overwrite=False): # This method is used to upload an image to the API server for use in img2img or controlnet
        url = f"http://{self.server_address}/upload/image"
        files = {'image': open(filepath, 'rb')}
        data = {
            'overwrite': str(overwrite).lower()
        }
        if subfolder:
            data['subfolder'] = subfolder
        if folder_type:
            data['type'] = folder_type
        response = requests.post(url, files=files, data=data)
        return response.json()

    @staticmethod
    def find_output_node(json_object): # This method is used to find the node containing the SaveImage class in a prompt
        for key, value in json_object.items():
            if isinstance(value, dict):
                if value.get("class_type") == "SaveImage":
                    return f"['{key}']"  # Return the key containing the SaveImage class
                result = ComfyConnector.find_output_node(value)
                if result:
                    return result
        return None
    
    @staticmethod
    def replace_key_value(json_object, target_key, new_value, class_type_list=None, exclude=True): # This method is used to edit the payload of a prompt
        for key, value in json_object.items():
            # Check if the current value is a dictionary and apply the logic recursively
            if isinstance(value, dict):
                class_type = value.get('class_type')                
                # Determine whether to apply the logic based on exclude and class_type_list
                should_apply_logic = (
                    (exclude and (class_type_list is None or class_type not in class_type_list)) or
                    (not exclude and (class_type_list is not None and class_type in class_type_list))
                )
                # Apply the logic to replace the target key with the new value if conditions are met
                if should_apply_logic and target_key in value:
                    value[target_key] = new_value
                # Recurse vertically (into nested dictionaries)
                ComfyConnector.replace_key_value(value, target_key, new_value, class_type_list, exclude)
            # Recurse sideways (into lists)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        ComfyConnector.replace_key_value(item, target_key, new_value, class_type_list, exclude)


###### test code below:

prompt_text = """
{
    "3": {
        "class_type": "KSampler",
        "inputs": {
            "cfg": 8,
            "denoise": 1,
            "latent_image": [
                "5",
                0
            ],
            "model": [
                "4",
                0
            ],
            "negative": [
                "7",
                0
            ],
            "positive": [
                "6",
                0
            ],
            "sampler_name": "euler",
            "scheduler": "normal",
            "seed": 8566257,
            "steps": 20
        }
    },
    "4": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {
            "ckpt_name": "1_bloody_mary_v1.safetensors"
        }
    },
    "5": {
        "class_type": "EmptyLatentImage",
        "inputs": {
            "batch_size": 1,
            "height": 512,
            "width": 512
        }
    },
    "6": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "clip": [
                "4",
                1
            ],
            "text": "masterpiece best quality girl"
        }
    },
    "7": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "clip": [
                "4",
                1
            ],
            "text": "bad hands"
        }
    },
    "8": {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": [
                "3",
                0
            ],
            "vae": [
                "4",
                2
            ]
        }
    },
    "9": {
        "class_type": "SaveImage",
        "inputs": {
            "filename_prefix": "ComfyUI",
            "images": [
                "8",
                0
            ]
        }
    }
}
"""

prompt = json.loads(prompt_text)



# Instantiate APISingleton
api_singleton = ComfyConnector()

# Test upload_image method
upload_result = api_singleton.upload_image('low_res_cat.png', folder_type='input')
print("Upload result:", upload_result)

# Test queue_prompt method
prompt_result = api_singleton.queue_prompt(prompt)
print("Prompt result:", prompt_result)

# Get the prompt ID
prompt_id = prompt_result['prompt_id']

# Test get_history method with the prompt ID
history_result = api_singleton.get_history(prompt_id)
print("History result:", history_result)

# Test generate_images method with the sample prompt
image_file = api_singleton.generate_images(prompt)
image = Image.open(image_file)
image_path = "image.png"  # Modify this path as needed
image.save(image_path)
print(f"Image saved to {image_path}")

# Assuming that the generated image filename, subfolder, and folder_type are known
filename = "image_00001_.png"
subfolder = ""
folder_type = "output"

# Test get_image method with the known details
image_data = api_singleton.get_image(filename, subfolder, folder_type)
image_file = io.BytesIO(image_data)
image = Image.open(image_file)
image.show()  # Display the retrieved image

print("All methods tested successfully.")