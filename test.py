
import requests
import io
import base64
import json



from PIL import Image, ImageDraw, ImageFont, PngImagePlugin

from framework.image_util import ImageUtil




# flow_id = "test_img2img_flow_1227"


# url = f"http://127.0.0.1:8188/open_api/service/{flow_id}/run"
# url1 = f"http://127.0.0.1:8188/open_api/service/upload_file"


# # img_filename = "input/a.jpg"
# img_filename = "input/example.png"


# ======================================================

# request = requests.Request('POST', url1, files={'file': open(img_filename, 'rb')})

# # 准备发送请求
# prepared_request = request.prepare()

# # 打印请求的具体数据
# print("Request Headers:")
# print(prepared_request.headers)
# print("Request Body:")
# print(prepared_request.body)

# # 发送请求
# response = requests.Session().send(prepared_request)
# print("Response:")
# print(response.text)
# print("\n\n")

# resp_data = response.json()
# body = {
#         "type": "",
#         "body": {
#             "ref_image": resp_data["data"]["filename"]
#         }
# }

# ==========================
# img = Image.open(img_filename)
# img_data = ImageUtil.image_to_base64(img, "png")

# body = {
#         "type": "",
#         "body": {
#             # "ref_image": "https://aiyo-1319341997.cos.ap-nanjing.myqcloud.com/test/02f1f71f-c649-494d-bce3-9ebf71cab13a.jpg?q-sign-algorithm=sha1&q-ak=AKID3Jzqub86IMtxogsO-dWD8oMslvxDnHBw6wBetPx-dGuvqgQzOHgs6vbG2Paw7TcD&q-sign-time=1703822106;1703825706&q-key-time=1703822106;1703825706&q-header-list=host&q-url-param-list=ci-process&q-signature=af75996b62ba5e73c8ad42002bbf08ab36191f6a&x-cos-security-token=SapgNsBV74n1LQ6vHrGnjOPeZfBbr0ga82f9daf21f19ddd63d8964452be5ec02biOgSaFdF4mC5nCCvd6U5JRKUH9BFogo_91qdMg4V9xvzc-WvkfSFPAhhY_tfaRp1ih5kWuJ-xb2nUnaxT2saMC-8kny0LB-DFWsMnwXEhS4ui5_FeEhEbyartQlkIT0KN6CMlVT4wg_nOU1mB7mfD8rvbYFtApUrfNiD_V_sgw6RfTWeEgJoS8WBtsjte1b&ci-process=originImage"
#             # "ref_image": "https://aiyo-1319341997.cos.ap-nanjing.myqcloud.com/test/02f1f71f-c649-494d-bce3-9ebf71cab13a.jpg"
#             "ref_image": img_data
#         }
# }

# request = requests.Request('POST', url, json=body)
# prepared_request = request.prepare()
# # 打印请求的具体数据
# print("Request Headers:")
# print(prepared_request.headers)
# print("Request Body:")
# print(prepared_request.body)

# # 发送请求
# response = requests.Session().send(prepared_request)
# print("Response:")
# print(response.text)


# ================================
# url_webhook = f"http://127.0.0.1:8188/open_api/service/{flow_id}/register_webhook"

# body = {
#     "on_start": "http://127.0.0.1:8188/test/on_start",
#     "on_end": "http://127.0.0.1:8188/test/on_end",
#     "on_processing": "http://127.0.0.1:8188/test/on_processing"
#     }

# request = requests.Request('POST', url_webhook, json=body)
# prepared_request = request.prepare()
# # 打印请求的具体数据
# print("Request Headers:")
# print(prepared_request.headers)
# print("Request Body:")
# print(prepared_request.body)

# # 发送请求
# response = requests.Session().send(prepared_request)
# print("Response:")
# print(response.text)




# =============================================

flow_id = "test_img2img_flow_1229"
url = f"http://127.0.0.1:8188/open_api/service/{flow_id}/run"
img_filename = "input/example.png"

img = Image.open(img_filename)
img_data = ImageUtil.image_to_base64(img, "png")

body = {
        "type": "",
        "body": {
            # "ref_image": "https://aiyo-1319341997.cos.ap-nanjing.myqcloud.com/test/02f1f71f-c649-494d-bce3-9ebf71cab13a.jpg?q-sign-algorithm=sha1&q-ak=AKID3Jzqub86IMtxogsO-dWD8oMslvxDnHBw6wBetPx-dGuvqgQzOHgs6vbG2Paw7TcD&q-sign-time=1703822106;1703825706&q-key-time=1703822106;1703825706&q-header-list=host&q-url-param-list=ci-process&q-signature=af75996b62ba5e73c8ad42002bbf08ab36191f6a&x-cos-security-token=SapgNsBV74n1LQ6vHrGnjOPeZfBbr0ga82f9daf21f19ddd63d8964452be5ec02biOgSaFdF4mC5nCCvd6U5JRKUH9BFogo_91qdMg4V9xvzc-WvkfSFPAhhY_tfaRp1ih5kWuJ-xb2nUnaxT2saMC-8kny0LB-DFWsMnwXEhS4ui5_FeEhEbyartQlkIT0KN6CMlVT4wg_nOU1mB7mfD8rvbYFtApUrfNiD_V_sgw6RfTWeEgJoS8WBtsjte1b&ci-process=originImage"
            # "ref_image": "https://aiyo-1319341997.cos.ap-nanjing.myqcloud.com/test/02f1f71f-c649-494d-bce3-9ebf71cab13a.jpg"
            "ref_image": img_data,
            "pos_prompt": "2 years-old, cute girl"
        }
}

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
