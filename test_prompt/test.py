#This is an example that uses the websockets api and the SaveImageWebsocket node to get images directly without
#them being saved to disk

import websocket #NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import json
import urllib.request
import urllib.parse
import time
import requests
import random

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
    current_node = ""
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['prompt_id'] == prompt_id:
                    if data['node'] is None:
                        break #Execution is done
                    else:
                        current_node = data['node']
        else:
            if current_node == 'save_image_websocket_node':
                images_output = output_images.get(current_node, [])
                images_output.append(out[8:])
                output_images[current_node] = images_output

    return output_images


# prompt_text = """
# {
# ...
# }
# """

# prompt = json.loads(prompt_text)
def main():
    with open("workflow_api.json") as workflow_api:
        prompt = json.load(workflow_api)

    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))


    prompt_text = "23  degrees Celsius, no people, only show the sky, weather forecast, cloudy"

    # 更新 CLIPTextEncode 的文本提示
    prompt["6"]["inputs"]["text"] = prompt_text

    # 将提示加入队列并获取 prompt_id
    queue_prompt(prompt)['prompt_id']

    # 打印生成的提示文本
    print(prompt_text)

    
    #set the seed for our KSampler node
    # prompt["3"]["inputs"]["seed"] = 5

    # prompt["6"]["inputs"]["text"] = "green sky"
    # queue_prompt(prompt)['prompt_id']

    # images = get_images(ws, prompt)

    #Commented out code to display the output images:

    # for node_id in images:
    #     for image_data in images[node_id]:
    #         from PIL import Image
    #         import io
    #         image = Image.open(io.BytesIO(image_data))
    #         image.show()

if __name__ == "__main__":
    main()

####################### 沒有文字藝術的串接
#This is an example that uses the websockets api and the SaveImageWebsocket node to get images directly without
#them being saved to disk

import websocket #NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import json
import urllib.request
import urllib.parse
import time
import requests
import random

import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# 获取当前脚本文件所在的目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 构建相对 ComfyUI 目录的路径
comfyui_dir = os.path.abspath(os.path.join(script_dir, '..'))

from ECCV2022_RIFE.InterpolatorInterface import InterpolatorInterface

from generate_prompt import generate_prompt

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
    current_node = ""
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['prompt_id'] == prompt_id:
                    if data['node'] is None:
                        break #Execution is done
                    else:
                        current_node = data['node']
        else:
            if current_node == 'save_image_websocket_node':
                images_output = output_images.get(current_node, [])
                images_output.append(out[8:])
                output_images[current_node] = images_output

    return output_images

# prompt = json.loads(prompt_text)
def main():
    with open("workflow_api.json") as workflow_api:
        prompt = json.load(workflow_api)

    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))
    
    interpolator = InterpolatorInterface() # Initialize image to video

    temp = 300
    while(1):
        next = temp + 1
        # 生prompt
        prompt_text = generate_prompt()

        # 生圖
        prompt["6"]["inputs"]["text"] = prompt_text        # 更新 CLIPTextEncode 的文本提示
        queue_prompt(prompt)['prompt_id']        # 将提示加入队列并获取 prompt_id

        print(prompt_text)

        # temp_str = f"output/ComfyUI_00{str(temp)}_.png"
        # next_str = f"output/ComfyUI_00{str(next)}_.png"
        temp_str = os.path.join(comfyui_dir, f"output/ComfyUI_00{str(temp)}_.png")
        next_str = os.path.join(comfyui_dir, f"output/ComfyUI_00{str(next)}_.png")

        # print(temp_str)

        # image to video
        results = interpolator.generate(
            imgs = (temp_str, next_str),
            exp=4,
            # output_dir="interpolate_out"
            # output_dir=os.path.join(comfyui_dir, "interpolate_out") 
        )

        temp = next

        print("Output results to: ")
        print(results)
        print()

if __name__ == "__main__":
    main()
