#This is an example that uses the websockets api to know when a prompt execution is done
#Once the prompt execution is done it downloads the images using the /history endpoint

import websocket #NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import json
import urllib.request
import urllib.parse

server_address = "127.0.0.1:8188"
client_id = str(uuid.uuid4())

def queue_prompt(prompt):
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req =  urllib.request.Request("http://{}/prompt".format(server_address), data=data)
    return json.loads(urllib.request.urlopen(req).read())

def get_image(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen("http://{}/view?{}".format(server_address, url_values)) as response:
        return response.read()

def get_history(prompt_id):
    with urllib.request.urlopen("http://{}/history/{}".format(server_address, prompt_id)) as response:
        return json.loads(response.read())

def get_images(ws, prompt):
    prompt_id = queue_prompt(prompt)['prompt_id']
    output_images = {}
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break #Execution is done
        else:
            continue #previews are binary data

    history = get_history(prompt_id)[prompt_id]
    for o in history['outputs']:
        for node_id in history['outputs']:
            node_output = history['outputs'][node_id]
            if 'images' in node_output:
                images_output = []
                for image in node_output['images']:
                    image_data = get_image(image['filename'], image['subfolder'], image['type'])
                    images_output.append(image_data)
            output_images[node_id] = images_output

    return output_images

# prompt_text = """

# {"3": {"inputs": {"seed": 160913364876129, "steps": 16, "cfg": 6.0, "sampler_name": "uni_pc", "scheduler": "normal", "denoise": 1.0, "model": ["14", 0], "positive": ["10", 0], "negative": ["7", 0], "latent_image": ["5", 0]}, "class_type": "KSampler"}, "5": {"inputs": {"width": 512, "height": 512, "batch_size": 1}, "class_type": "EmptyLatentImage"}, "6": {"inputs": {"text": "(solo) girl (flat chest:0.9), (fennec ears:1.1)\u00a0 (fox ears:1.1), (blonde hair:1.0), messy hair, sky clouds, standing in a grass field, (chibi), blue eyes", "clip": ["14", 1]}, "class_type": "CLIPTextEncode"}, "7": {"inputs": {"text": "(hands), text, error, cropped, (worst quality:1.2), (low quality:1.2), normal quality, (jpeg artifacts:1.3), signature, watermark, username, blurry, artist name, monochrome, sketch, censorship, censor, (copyright:1.2), extra legs, (forehead mark) (depth of field) (emotionless) (penis)", "clip": ["14", 1]}, "class_type": "CLIPTextEncode"}, "8": {"inputs": {"samples": ["3", 0], "vae": ["13", 0]}, "class_type": "VAEDecode"}, "9": {"inputs": {"filename_prefix": "ComfyUI", "images": ["8", 0]}, "class_type": "SaveImage"}, "10": {"inputs": {"strength": 0.8999999999999999, "conditioning": ["6", 0], "control_net": ["12", 0], "image": ["11", 0]}, "class_type": "ControlNetApply"}, "11": {"inputs": {"image": "ComfyUI_00005_ (1).png", "choose file to upload": "image"}, "class_type": "LoadImage", "is_changed": ["7657ee164339745ea7b5300e55a7655f0404fbb5a0a61d990748027a19e2f178"]}, "12": {"inputs": {"control_net_name": "control_v11p_sd15_canny_fp16.safetensors"}, "class_type": "ControlNetLoader"}, "13": {"inputs": {"vae_name": "vae-ft-mse-840000-ema-pruned.safetensors"}, "class_type": "VAELoader"}, "14": {"inputs": {"ckpt_name": "sd-v1-4.ckpt"}, "class_type": "CheckpointLoaderSimple"}}
# """

prompt_text = """

{"3": {"inputs": {"seed": 156680208700286, "steps": 20, "cfg": 8.0, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0, "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0]}, "class_type": "KSampler"}, "4": {"inputs": {"ckpt_name": "sd-v1-4.ckpt"}, "class_type": "CheckpointLoaderSimple"}, "5": {"inputs": {"width": 512, "height": 512, "batch_size": 1}, "class_type": "EmptyLatentImage"}, "6": {"inputs": {"text": "beautiful scenery nature glass bottle landscape, , purple galaxy bottle,", "clip": ["4", 1]}, "class_type": "CLIPTextEncode"}, "7": {"inputs": {"text": "text, watermark", "clip": ["4", 1]}, "class_type": "CLIPTextEncode"}, "8": {"inputs": {"samples": ["3", 0], "vae": ["4", 2]}, "class_type": "VAEDecode"}, "9": {"inputs": {"filename_prefix": "ComfyUI", "images": ["8", 0]}, "class_type": "SaveImage"}}

"""


prompt = json.loads(prompt_text)
#set the text prompt for our positive CLIPTextEncode
# prompt["6"]["inputs"]["text"] = "a girl in anime style, cute"

# prompt["11"]["inputs"]["image"] = "D:/Downloads/IMG_7150.jpeg"

# #for sd 2.1 
# prompt["12"]["inputs"]["control_net_name"] = "controlnetFurususSD21_21Canny.safetensors"
# prompt["14"]["inputs"]["ckpt_name"] = "v2-1_512-ema-pruned.ckpt"


# #set the seed for our KSampler node
# prompt["3"]["inputs"]["seed"] = 5

ws = websocket.WebSocket()
ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))
images = get_images(ws, prompt)

#Commented out code to display the output images:

for node_id in images:
    for image_data in images[node_id]:
        from PIL import Image
        import io
        image = Image.open(io.BytesIO(image_data))
        image.show()

