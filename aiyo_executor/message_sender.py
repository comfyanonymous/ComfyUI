
import copy, os

import asyncio
# import struct
# from io import BytesIO
import requests
import aiohttp


from config.config import CONFIG
from framework.app_log import AppLog
from framework.err_code import ErrorCode
from framework.image_util import ImageUtil
from PIL import Image
from framework.model import object_storage


class MessageManager():
    def __init__(self, loop):

        self.messages = asyncio.Queue()
        self.client_id = None
        self.loop = loop

        self.on_prompt_handlers = []
        
        self.message_sender = None

    
            

    async def send(self, event, data, sid=None):
        if self.message_sender is not None:
            await self.message_sender.send(event, data, sid)


    
    def send_sync(self, event, data, sid=None):
        self.loop.call_soon_threadsafe(
            self.messages.put_nowait, (event, data, sid))
        
        
        

    async def publish_loop(self):
        while True:
            msg = await self.messages.get()
            await self.send(*msg)
            



class APICall:

    @staticmethod
    def get_sync(api, json_data):
        """
        Request GET api to AIYoServer
        RETURN:
        bool, success or not
        dict, response data
        """
        url = api
        response = requests.get(url, json=json_data, timeout=5)
        if response.status_code != requests.codes.ok:
            AppLog.error(f'[Get Task] server not response: {response}, {response.reason}')        
            return False, None
        else:
            json_response = response.json()
            return True, json_response      
        
    
    @staticmethod
    def post_sync(api, json_data):  
        """
        Request get api to AIYoServer
        RETURN:
        bool, success or not
        dict, response data
        """
        url = api
        response = requests.post(url, json=json_data, timeout=5)
        if response.status_code != requests.codes.ok:
            AppLog.error(f'[Post Sync] server not response: {response}, {response.reason}')        
            return False, None
        else:
            try:
                json_response = response.json()
                return True, json_response
            except Exception as e:
                return True, None
        
    
    @staticmethod
    async def get_async(api, json_data):
        url = api
        async with aiohttp.ClientSession() as session:
            async with session.get(url, json=json_data) as response:
                if response.status == 200:
                    result = await response.json()
                    return True, result
                else:
                    error_message = await response.text()  # 获取错误信息
                    AppLog.error(f'[Get Async] server not response: {error_message}') 
                    return False, None
                
                
    @staticmethod
    async def post_async(api, json_data):
        url = api
        AppLog.info(f"[Post async] api:{api}, data: {AppLog.visible_convert(json_data)}")
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=json_data) as response:
                if response.status == 200:
                    try:
                        result = await response.json()
                        return True, result
                    except Exception as e:
                        return True, None
                else:
                    error_message = await response.text()  # 获取错误信息
                    AppLog.error(f'[Get Async] server not response: {AppLog.visible_convert(error_message)}') 
                    return False, None




class WeebhookSender:
    
    def __init__(self, on_start=None, on_processing=None, on_end=None) -> None:
        self.on_start = on_start
        self.on_processing = on_processing
        self.on_end = on_end
        
    
    async def send(self, event, data, sid=None):
        succ = True
        if event == "execution_start" and self.on_start is not None:
            task_id = data["prompt_id"]
            succ,_ = await APICall.post_async(self.on_start, {"task_id": task_id})
            
            if not succ:
                AppLog.warning(f"[WebhookSender] {event}, fail to call webhook:{self.on_start}, data:{AppLog.visible_convert(data)}")
            
        elif event == "execution_end" and self.on_end is not None:
            task_id = data["prompt_id"]
            results = copy.deepcopy(data["result"])
            result_info = data["result_info"]
            for res_name, res_val in results.items():
                if result_info[res_name]["type"] == "IMAGE":
                    file_basename = os.path.basename(res_val)
                    local_dir = CONFIG["resource"]["out_img_path_local"]
                    local_path = f"{local_dir}/{file_basename}"
                    img = Image.open(local_path)
                    img_base64 = ImageUtil.image_to_base64(img)
                    results[res_name] = img_base64

            succ, _ = await APICall.post_async(self.on_end, {"task_id": task_id, 
                                                       "code": ErrorCode.SUCCESS,
                                                       "result": results, "message": ""})
            
            self.on_start = None
            self.on_end = None
            self.on_processing = None
            if not succ:
                AppLog.warning(f"[WebhookSender] {event}, fail to call webhook:{self.on_end}, data:{AppLog.visible_convert(data)}")
            
        elif event == "execution_error" and self.on_end is not None:
            task_id = data["prompt_id"]
            exp_msg = data["exception_message"]
            node_type = data["class_type"]
            exp_type = data["exception_type"]
            msg = f"EXCEPTION: {exp_type}. \nNODE: {node_type}\nERROR:{exp_msg}"
            succ, _ = await APICall.post_async(self.on_end, {"task_id": task_id, 
                                                       "code": ErrorCode.EXE_UNEXP,
                                                       "result": None, "message": msg})
            if not succ:
                AppLog.warning(f"[WebhookSender] {event}, fail to call webhook:{self.on_end}, data:{data}")
            
        # elif event == "executed" and self.on_processing is not None:
            
        
            


class LocalAPISender:
    async def send(self, event, data, sid=None):
        url = CONFIG["server"]["url"]
        
        if event == "execution_end":
            url = f"{url}/task_exe/task_done"
            post_data = {
                "prompt_id": data["prompt_id"],
                "output": data["result"]
            }
            succ, _ =await APICall.post_async(url, post_data)
            if not succ:
                AppLog.warning(f"[LocalAPISender] {event}, fail to call :{url}, data:{data}")
        else:
            url = f"{url}/task_exe/send_task_msg"
            post_data = {
                "event_type": event,
                "data": data,
                "sid": sid
            }
            succ, _= await APICall.post_async(url, post_data)
            if not succ:
                AppLog.warning(f"[LocalAPISender] {event}, fail to call :{url}, data:{data}")