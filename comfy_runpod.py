#This is an example that uses the websockets api to know when a prompt execution is done
#Once the prompt execution is done it downloads the images using the /history endpoint

import websocket #NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import json
import urllib.request
import urllib.parse
from PIL import Image
import base64
import io

from custom_scripts_for_nodes.Rainbow import extract_rainbow

import runpod


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



def run_prompt(job):
    
    data = {'images':[],'rainbow':"None"}
    
    #Inferring from the Rainbow Script
    image_string = job['input']['image_string']
    prompt_text = job["input"]["prompt"]
    
    if image_string != 'None': 
        # Decode the base64 string into bytes
        decoded_bytes = base64.b64decode(image_string)
        # Convert the bytes to an in-memory file-like object using io.BytesIO
        image_data = io.BytesIO(decoded_bytes)
        image = Image.open(image_data)
        rnbw = extract_rainbow()
        rnbw_values = rnbw.main(image)
        
        data['rainbow'] = rnbw_values
    
    if prompt_text != "None":
        prompt = prompt_text
        ws = websocket.WebSocket()
        ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))
        images = get_images(ws, prompt)
        
        
        for node_id in images:
            for image_data in images[node_id]:
                image = Image.open(io.BytesIO(image_data))
                im_file = io.BytesIO()
                image.save(im_file, format="JPEG")
                im_bytes = im_file.getvalue()  # im_bytes: image in binary format.
                im_b64 = base64.b64encode(im_bytes)
                im_b64 = str(im_b64) 
                data['images'].append(im_b64)
    return data
    

runpod.serverless.start({"handler":run_prompt})
