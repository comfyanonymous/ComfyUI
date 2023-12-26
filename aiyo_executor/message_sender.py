
import asyncio
# import struct
# from io import BytesIO
import requests
import aiohttp

# from PIL import Image, ImageOps
# import aiohttp

from config.config import CONFIG

# class BinaryEventTypes:
#     PREVIEW_IMAGE = 1
#     UNENCODED_PREVIEW_IMAGE = 2


# async def send_socket_catch_exception(function, message):
#     try:
#         await function(message)
#     except (aiohttp.ClientError, aiohttp.ClientPayloadError, ConnectionResetError) as err:
#         print("send error:", err)
        

class MessageSender():
    def __init__(self, loop):

        self.messages = asyncio.Queue()
        self.client_id = None
        self.loop = loop

        self.on_prompt_handlers = []


    async def send(self, event, data, sid=None):
        
        # if event == BinaryEventTypes.UNENCODED_PREVIEW_IMAGE:
        #     await self.send_image(data, sid=sid)
        # elif isinstance(data, (bytes, bytearray)):
        #     await self.send_bytes(event, data, sid)
        # else:
        #     await self.send_json(event, data, sid)
        
        url = CONFIG["server"]["url"]
        url = f"{url}/task_exe/send_task_msg"
        # requests.post(url, json={
        #     "event_type": event,
        #     "data": data,
        #     "sid": sid
        # }, timeout=30)
        
        post_data = {
            "event_type": event,
            "data": data,
            "sid": sid
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=post_data) as response:
                # data = await response.json()
                return

    # def encode_bytes(self, event, data):
    #     if not isinstance(event, int):
    #         raise RuntimeError(f"Binary event types must be integers, got {event}")

    #     packed = struct.pack(">I", event)
    #     message = bytearray(packed)
    #     message.extend(data)
    #     return message

    # async def send_image(self, image_data, sid=None):
    #     image_type = image_data[0]
    #     image = image_data[1]
    #     max_size = image_data[2]
    #     if max_size is not None:
    #         if hasattr(Image, 'Resampling'):
    #             resampling = Image.Resampling.BILINEAR
    #         else:
    #             resampling = Image.ANTIALIAS

    #         image = ImageOps.contain(image, (max_size, max_size), resampling)
    #     type_num = 1
    #     if image_type == "JPEG":
    #         type_num = 1
    #     elif image_type == "PNG":
    #         type_num = 2

    #     bytesIO = BytesIO()
    #     header = struct.pack(">I", type_num)
    #     bytesIO.write(header)
    #     image.save(bytesIO, format=image_type, quality=95, compress_level=1)
    #     preview_bytes = bytesIO.getvalue()
    #     await self.send_bytes(BinaryEventTypes.PREVIEW_IMAGE, preview_bytes, sid=sid)

    # async def send_bytes(self, event, data, sid=None):
    #     message = self.encode_bytes(event, data)

    #     if sid is None:
    #         for ws in self.sockets.values():
    #             await send_socket_catch_exception(ws.send_bytes, message)
    #     elif sid in self.sockets:
    #         await send_socket_catch_exception(self.sockets[sid].send_bytes, message)

    # async def send_json(self, event, data, sid=None):
    #     message = {"type": event, "data": data}

    #     if sid is None:
    #         for ws in self.sockets.values():
    #             await send_socket_catch_exception(ws.send_json, message)
    #     elif sid in self.sockets:
    #         await send_socket_catch_exception(self.sockets[sid].send_json, message)

    
    def send_sync(self, event, data, sid=None):
        self.loop.call_soon_threadsafe(
            self.messages.put_nowait, (event, data, sid))


    async def publish_loop(self):
        while True:
            msg = await self.messages.get()
            await self.send(*msg)
            

