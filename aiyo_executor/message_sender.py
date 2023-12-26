
import asyncio
# import struct
# from io import BytesIO
import requests
import aiohttp


from config.config import CONFIG
from framework.app_log import AppLog


class MessageSender():
    def __init__(self, loop):

        self.messages = asyncio.Queue()
        self.client_id = None
        self.loop = loop

        self.on_prompt_handlers = []

    @staticmethod
    def get_sync(api, json_data):
        """
        Request GET api to AIYoServer
        RETURN:
        bool, success or not
        dict, response data
        """
        url = CONFIG["server"]["url"]
        url = f"{url}{api}"
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
        url = CONFIG["server"]["url"]
        url = f"{url}{api}"
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
        url = CONFIG["server"]["url"]
        url = f"{url}{api}"
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
        url = CONFIG["server"]["url"]
        url = f"{url}{api}"
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
                    AppLog.error(f'[Get Async] server not response: {error_message}') 
                    return False, None

            

    async def send(self, event, data, sid=None):
        url = CONFIG["server"]["url"]
        url = f"{url}/task_exe/send_task_msg"
        post_data = {
            "event_type": event,
            "data": data,
            "sid": sid
        }
        await MessageSender.post_async("/task_exe/send_task_msg", post_data)


    
    def send_sync(self, event, data, sid=None):
        self.loop.call_soon_threadsafe(
            self.messages.put_nowait, (event, data, sid))
        
        
        

    async def publish_loop(self):
        while True:
            msg = await self.messages.get()
            await self.send(*msg)
            

