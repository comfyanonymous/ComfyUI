# 开发人员使用（请勿运行）
# 将 https://github.com/AIGODLIKE/AIGODLIKE-ComfyUI-Translation 的翻译文件转换格式以适配 ComfyUI locales

import json
import os
import pathlib

old_json_path = 'ComfyUI-Easy-Use.json'
root_path = pathlib.Path(__file__).parent.parent
new_json_path = os.path.join(root_path,'locales/zh/nodeDefs.json')

def transform_dict(data):
    new_dict = {}
    for k, v in data.items():
        new_dict[k] = {
            "display_name": "",
            "inputs": {}
        }
        if isinstance(v, dict):
            for key, value in v.items():
                if key == 'title':
                    new_dict[k]['display_name'] = value
                elif key in ['inputs','widgets']:
                    for _key, _value in value.items():
                        new_dict[k]['inputs'] = {
                            **new_dict[k]['inputs'],
                            _key: {"name": _value}
                        }
                elif key == 'outputs':
                    if not new_dict[k].get('outputs'):
                        new_dict[k]['outputs'] = {}
                    for idx, (out_key, out_value) in enumerate(value.items()):
                        new_dict[k]['outputs'][idx] = {"name": out_value}
    return new_dict

def main():

    # 读取原始JSON文件
    with open(old_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 转换数据
    transformed_data = transform_dict(data)
    
    # 写入新的JSON文件
    with open(new_json_path, 'w', encoding='utf-8') as f:
        json.dump(transformed_data, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()
