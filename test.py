
import requests
import io
import base64
import json



from PIL import Image, ImageDraw, ImageFont, PngImagePlugin



def get_pil_metadata(pil_image):
    # Copy any text-only metadata
    metadata = PngImagePlugin.PngInfo()
    for key, value in pil_image.info.items():
        if isinstance(key, str) and isinstance(value, str):
            metadata.add_text(key, value)

    return metadata
# from framework.model import tb_data

def image_to_base64(img: Image, format='PNG') -> str:
    """Converts a PIL Image object to a base64 encoded string.

    Args:
        img (PIL.Image): The image object to convert.

    Returns:
        str: The base64 encoded string.
    """
    # Convert the image to bytes using an in-memory buffer
    buffer = io.BytesIO()
    # if format == 'PNG':
    #     img.save(buffer, format=format, pnginfo=get_pil_metadata(img))
    # else:
    #     img.save(buffer, format=format)
    img.save(buffer, format=format)
    img_bytes = buffer.getvalue()

    # Encode the bytes as base64
    base64_str = base64.b64encode(img_bytes).decode()

    return base64_str

flow_id = "test_img2img_flow_1227"

# tb_data.default_connect()


# flow_info = tb_data.Flow.objects(flowId=flow_id)[0]
# print(flow_info)
# print(flow_info.flowInput)
# print(flow_info.flowOutput)

url = f"http://127.0.0.1:8188/open_api/service/{flow_id}/run"
url1 = f"http://127.0.0.1:8188/open_api/service/upload_file"


img_filename = "input/a.jpg"

# # img = Image.open(img_filename)
# # img_str = image_to_base64(img)

# with open(img_filename, 'rb') as img_file:
#     image_data = img_file.read()

# body = {
#         "type": "",
#         "body": json.dumps({
#             "ref_image": image_data#img_str
#         })
# }
# response = requests.post(url=url, json=body)

# print(response.json)


# form_data = {
#     'field1': 'value1',
#     'field2': 'value2'
# }
# json_data = {
#     'key1': 'value1',
#     'key2': 'value2'
# }

# response = requests.post(url, data=form_data, json=json_data)
# 构建请求对象
# request = requests.Request('POST', url, data=form_data, json=json_data)
# request = requests.Request('POST', url, files={'file': open(img_filename, 'rb')})

# request = requests.Request('POST', url1, data={'file': image_data})


request = requests.Request('POST', url1, files={'file': open(img_filename, 'rb')})

# 准备发送请求
prepared_request = request.prepare()

# 打印请求的具体数据
print("Request Headers:")
print(prepared_request.headers)
print("Request Body:")
print(prepared_request.body)

# 发送请求
response = requests.Session().send(prepared_request)
print("Response:")
print(response.text)
print("\n\n")

resp_data = response.json()
body = {
        "type": "",
        "body": {
            "ref_image": resp_data["data"]["filename"]
        }
}

# ==========================

request = requests.Request('POST', url, json=body)
prepared_request = request.prepare()
# 打印请求的具体数据
print("Request Headers:")
print(prepared_request.headers)
print("Request Body:")
print(prepared_request.body)

# 发送请求
response = requests.Session().send(prepared_request)
print("Response:")
print(response.text)
