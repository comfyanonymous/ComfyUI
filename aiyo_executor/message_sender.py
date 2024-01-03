
import copy, os
import traceback
from io import BytesIO
import struct

import asyncio
# import struct
# from io import BytesIO
import requests
import aiohttp


from config.config import CONFIG
from framework.app_log import AppLog
from framework.err_code import ErrorCode
from framework.image_util import ImageUtil
from PIL import Image, ImageOps
from framework.model import object_storage
from framework.event_types import AIYohEventTypes


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
        try:
            response = requests.get(url, json=json_data, timeout=5)
        except Exception as e:
            AppLog.error(f"[API(Get)] get_syncs, ERROR. \n{e} \n{traceback.format_exc()}")
            return False, None
        
        if response.status_code != requests.codes.ok:
            AppLog.error(f'[API(Get)] server not response: {response}, {response.reason}')        
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


    @staticmethod
    async def post_data_async(api, data):
        url = api
        AppLog.info(f"[Post data async] api:{api}, data: {AppLog.visible_convert(data)}")
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data) as response:
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
        if event == AIYohEventTypes.EXE_START and self.on_start is not None:
            task_id = data["prompt_id"]
            succ,_ = await APICall.post_async(self.on_start, {"task_id": task_id})
            
            if not succ:
                AppLog.warning(f"[WebhookSender] {event}, fail to call webhook:{self.on_start}, data:{AppLog.visible_convert(data)}")
            
        elif event == AIYohEventTypes.EXE_END and self.on_end is not None:
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
            
        elif event == AIYohEventTypes.EXE_ERR and self.on_end is not None:
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
        
        if event == AIYohEventTypes.EXE_END:
            url = f"{url}/task_exe/task_done"
            post_data = {
                "prompt_id": data["prompt_id"],
                "output": data["result"]
            }
            succ, _ =await APICall.post_async(url, post_data)
            if not succ:
                AppLog.warning(f"[LocalAPISender] {event}, fail to call :{url}, data:{data}")
        elif event == AIYohEventTypes.UNENCODED_PREVIEW_IMAGE:
            await self.send_image(data, sid)

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
                
                
                
    async def send_image(self, image_data, sid=None):
        
        image_type = image_data[0]
        image = image_data[1]
        max_size = image_data[2]
        if max_size is not None:
            if hasattr(Image, 'Resampling'):
                resampling = Image.Resampling.BILINEAR
            else:
                resampling = Image.ANTIALIAS

            image = ImageOps.contain(image, (max_size, max_size), resampling)
        
        img_str = ImageUtil.image_to_base64(image, image_type)
        post_data = {
                "event_type": AIYohEventTypes.UNENCODED_PREVIEW_IMAGE,
                "data": img_str,
                "sid": sid
            }
        
        url = CONFIG["server"]["url"]
        url = f"{url}/task_exe/send_task_msg"
        succ, _ = await APICall.post_async(url, post_data)
        if not succ:
            AppLog.warning(f"[LocalAPISender] {AIYohEventTypes.UNENCODED_PREVIEW_IMAGE}, fail to call :{url}, data:{AppLog.visible_convert(post_data)}")
            
        # type_num = 1
        # if image_type == "JPEG":
        #     type_num = 1
        # elif image_type == "PNG":
        #     type_num = 2

        # bytesIO = BytesIO()
        # header = struct.pack(">I", type_num)
        # bytesIO.write(header)
        # image.save(bytesIO, format=image_type, quality=95, compress_level=1)
        # preview_bytes = bytesIO.getvalue()
        # await self.send_bytes(AIYohEventTypes.PREVIEW_IMAGE, preview_bytes, sid=sid)

    async def send_bytes(self, event, data, sid=None):
        message = self.encode_bytes(event, data)
        
        url = CONFIG["server"]["url"]
        url = f"{url}/task_exe/send_prev_image"
        post_data = {
            "event_type": event,
            "data": message,
            "sid": sid
        }

        succ, _ = await APICall.post_data_async(url, post_data)
        if not succ:
            AppLog.warning(f"[LocalAPISender] {event}, fail to call :{url}, data:{data}")
            
            
    def encode_bytes(self, event, data):
        if not isinstance(event, int):
            raise RuntimeError(f"Binary event types must be integers, got {event}")

        packed = struct.pack(">I", event)
        message = bytearray(packed)
        message.extend(data)
        return message
    