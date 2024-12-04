#This is an example that uses the websockets api and the SaveImageWebsocket node to get images directly without
#them being saved to disk

import websocket #NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import json
import urllib.request
import urllib.parse
import numpy as np
from numpy.typing import NDArray
import random
import cv2
from concurrent.futures import ThreadPoolExecutor, wait
import sys
import os
import asyncio
from datetime import datetime

# 將 Cloud_Image 設定為根目錄
script_dir = os.path.dirname(os.path.abspath(__file__))
cloud_image_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))  # 獲取 Cloud_Image 資料夾的路徑
sys.path.append(cloud_image_dir)  # 添加 Cloud_Image 到 sys.path

# 使用絕對匯入
from ECCV2022_RIFE.InterpolatorInterface import InterpolatorInterface

from generate_prompt2 import generate_prompt, get_emotion, get_temperature
from temperature import draw_thermometer, overlay_images

# 設定儲存路徑
output_dir = '/Cloud_Image/ComfyUI/test_prompt/test_image'

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

# 讀取最新兩張圖片
def get_latest_images(directory, count=2):
    # 獲取資料夾中的所有文件，排除隱藏文件
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and not f.startswith('.')]
    
    if not files:
        return []  # 如果沒有圖片，返回空列表
    
    # 按照最後修改時間對文件排序
    files.sort(key=lambda f: os.path.getmtime(os.path.join(directory, f)), reverse=True)
    
    # 取得最新的兩張圖片的檔案名稱
    latest_images = files[:count]
    
    return latest_images  # 返回最新圖片的檔案名稱列表


# 假設這是處理 ComfyUI 的函數
def process_comfyui():
    with open("workflow_api.json") as workflow_api:
        prompt = json.load(workflow_api)

    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))
    
    interpolator = InterpolatorInterface() # Initialize image to video

    while True:
        # ComfyUI 製圖
        prompt_text = generate_prompt()
        prompt["3"]["inputs"]["seed"] = random.randint(0,999999999999999)
        prompt["6"]["inputs"]["text"] = prompt_text        
        queue_prompt(prompt)['prompt_id']  # 將提示加入隊列（假設這個函數已經實現）
        print(prompt_text)

        # 設定 output 資料夾的路徑
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
        # 讀取最新的兩張圖片檔案名稱
        latest_images = get_latest_images(output_dir)
        # print(latest_images[0])
        # 拼接字串
        image_path0 = "/Cloud_Image/ComfyUI/output/" + latest_images[0]
        image_path1 = "/Cloud_Image/ComfyUI/output/" + latest_images[1]

        # 如果有圖片，顯示或處理
        if latest_images:
            # image to video
            results = interpolator.generate(
                imgs=(image_path0, image_path1),
                exp=4,
                # output_dir="interpolate_out"
                # output_dir=os.path.join(comfyui_dir, "interpolate_out") 
            )
        else:
            print("No images found in the output directory.")
            results = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(9)]

        print("process_comfyUI完成")
        # print("Output results to: ")
        # print(results)
        # print()

        return results


def draw_temperature(frame):
    temperature = get_temperature()
    thermometer = draw_thermometer(temperature)
    origin = frame.copy()
    scale_factor = 1 / 3
    new_height = int(origin.shape[0] * scale_factor)
    new_width = int((new_height / thermometer.shape[0]) * thermometer.shape[1])
    thermometer = cv2.resize(thermometer, (new_width, new_height))

    x_offset = origin.shape[1] - thermometer.shape[1] - 10
    y_offset = 10
    combined_frame = overlay_images(origin, thermometer, x_offset, y_offset, alpha=0.7)

    return combined_frame


def main():
    global latest_full_frame

    while True:
        result_frame = process_comfyui()

        # 拿出 9 張照片添加溫不計
        for i in range(9):
            full_frame = draw_temperature(result_frame[i])
            print("full_frame finish")
            # print(type(full_frame))
            
            # 根據當前時間生成檔名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}.png"

            # 完整的儲存路徑
            output_path = os.path.join(output_dir, filename)

            # 使用 OpenCV 將陣列儲存成圖片
            cv2.imwrite(output_path, full_frame) 

if __name__ == "__main__":
    main()
