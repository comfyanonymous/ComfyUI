import json
from urllib import request, parse
import os


# 定义JSON文件所在的目录
json_dir = 'F:\\workspace\\Comfyui_api\\workflow1'
data = []

data_file_path = []
# 获取目录下所有的JSON文件
json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]

# 读取每个JSON文件
for json_file in json_files:
    file_path = os.path.join(json_dir, json_file)
    data_file_path.append(json_file)
    with open(file_path, 'r') as f:
        json_data = json.load(f)
        # 处理读取的数据
        data.append(json_data)


def queue_prompt(prompt):
    # header = {'icool':'niubi666'}
    p = {"prompt": prompt}
    data = json.dumps(p).encode('utf-8')
    req = request.Request("http://127.0.0.1:8188/prompt", data=data)
    request.urlopen(req)


for i in range(len(data)):
    queue_prompt(data[i])
    print(data_file_path[i], "success")

print("prompt success")
