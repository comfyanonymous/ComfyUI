

import struct
from PIL import Image, ImageOps
from io import BytesIO
import aiohttp
import asyncio

from framework.app_log import AppLog
from framework.image_util import ImageUtil

class BinaryEventTypes:
    PREVIEW_IMAGE = 1
    UNENCODED_PREVIEW_IMAGE = 2
    


async def send_socket_catch_exception(function, message):
    try:
        await function(message)
    except (aiohttp.ClientError, aiohttp.ClientPayloadError, ConnectionResetError) as err:
        AppLog.info("send error:", err)
        
        
        
        
class ServerClientCommunicator:
    
    def __init__(self, server) -> None:
        self.server = server
        self.sockets = server.sockets
        self.messages = asyncio.Queue()
        
    
    async def process_one(self):
        msg = await self.messages.get()
        await self.send(*msg)
        
    async def send(self, event, data, sid=None):
        if event == BinaryEventTypes.UNENCODED_PREVIEW_IMAGE:
            await self.send_image(data, sid=sid)
        elif isinstance(data, (bytes, bytearray)):
            await self.send_bytes(event, data, sid)
        else:
            await self.send_json(event, data, sid)

    def encode_bytes(self, event, data):
        if not isinstance(event, int):
            raise RuntimeError(f"Binary event types must be integers, got {event}")

        packed = struct.pack(">I", event)
        message = bytearray(packed)
        message.extend(data)
        return message


    async def send_image(self, image_data, sid=None):

        image = ImageUtil.base64_to_image(image_data)
        image_type = image.format
        
        type_num = 1
        if image_type == "JPEG":
            type_num = 1
        elif image_type == "PNG":
            type_num = 2

        bytesIO = BytesIO()
        header = struct.pack(">I", type_num)
        bytesIO.write(header)
        image.save(bytesIO, format=image_type, quality=95, compress_level=1)
        preview_bytes = bytesIO.getvalue()
        await self.send_bytes(BinaryEventTypes.PREVIEW_IMAGE, preview_bytes, sid=sid)


    async def send_bytes(self, event, data, sid=None):
        message = self.encode_bytes(event, data)

        if sid is None:
            for ws in self.sockets.values():
                await send_socket_catch_exception(ws.send_bytes, message)
        elif sid in self.sockets:
            await send_socket_catch_exception(self.sockets[sid].send_bytes, message)


    async def send_json(self, event, data, sid=None):
        message = {"type": event, "data": data}

        if sid is None:
            for ws in self.sockets.values():
                await send_socket_catch_exception(ws.send_json, message)
        elif sid in self.sockets:
            await send_socket_catch_exception(self.sockets[sid].send_json, message)
            
            
    def send_sync(self, event, data, sid=None):
        self.server.loop.call_soon_threadsafe(
            self.messages.put_nowait, (event, data, sid))
        
        
        
    
    def queue_updated(self):
        """
        Send queue updated message
        """
        self.send_sync("status", { "status": self.server.prompt_queue.get_queue_info() })