import asyncio
import base64
from io import BytesIO
import os
import logging
import signal
import struct
import time
from typing import Optional
import uuid
from PIL import Image, ImageOps
from functools import partial

import pika
import json

import requests
import aiohttp

from dotenv import load_dotenv
load_dotenv()

amqp_addr = os.getenv('AMQP_ADDR') or 'amqp://api:gacdownatravKekmy9@51.8.120.154:5672/dev'

from enum import Enum

class QueueProgressKind(Enum):
    ImageGenerated = "image_generated"
    ImageGenerating = "image_generating"
    SamePrompt = "same_prompt"
    FaceswapGenerated = "faceswap_generated"
    FaceswapGenerating = "faceswap_generating"
    Failed = "failed"

class MemedeckWorker:
    class BinaryEventTypes:
        PREVIEW_IMAGE = 1
        UNENCODED_PREVIEW_IMAGE = 2

    class JsonEventTypes(Enum):
        PROGRESS = "progress"
        EXECUTING = "executing"
        EXECUTED = "executed"
        ERROR = "error"
        STATUS = "status"

    """
    MemedeckWorker is a class that is responsible for relaying messages between comfy and the memedeck backend api
    it is used to send images to the memedeck backend api and to receive prompts from the memedeck backend api
    """
    def __init__(self, loop):
        MemedeckWorker.instance = self
        logging.getLogger().setLevel(logging.INFO)
        
        self.loop = loop
        self.messages = asyncio.Queue()

        self.http_client = None
        self.prompt_queue = None
        self.validate_prompt = None
        self.last_prompt_id = None

        self.amqp_url = amqp_addr
        self.queue_name = os.getenv('QUEUE_NAME') or 'generic-queue'
        self.api_url = os.getenv('API_ADDRESS') or 'http://0.0.0.0:8079/v2'
        self.api_key = os.getenv('API_KEY') or 'eb46e20a-cc25-4ed4-a39b-f47ca8ff3383'

        # Internal job queue
        self.internal_job_queue = asyncio.Queue()

        # Dictionary to keep track of tasks by ws_id
        self.tasks_by_ws_id = {}

        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"\n[memedeck]: initialized with API URL: {self.api_url} and API Key: {self.api_key}\n")
        
    def on_connection_open(self, connection):
        self.connection = connection
        self.connection.channel(on_open_callback=self.on_channel_open)

    def on_channel_open(self, channel):
        self.channel = channel
        
        # only consume one message at a time
        self.channel.basic_qos(prefetch_size=0, prefetch_count=1)
        self.channel.queue_declare(queue=self.queue_name, durable=True)
        self.channel.basic_consume(queue=self.queue_name, on_message_callback=self.on_message_received, auto_ack=False)
        
    def start(self, prompt_queue, validate_prompt):
        self.prompt_queue = prompt_queue
        self.validate_prompt = validate_prompt
        
        # Start the process_job_queue task **after** prompt_queue is set
        self.loop.create_task(self.process_job_queue())
        
        parameters = pika.URLParameters(self.amqp_url)
        logging.getLogger('pika').setLevel(logging.WARNING) # suppress all logs from pika
        self.connection = pika.SelectConnection(parameters, on_open_callback=self.on_connection_open)

        try:
            self.connection.ioloop.start()
        except KeyboardInterrupt:
            self.connection.close()
            self.connection.ioloop.start()
        
    def on_message_received(self, channel, method, properties, body):        
        decoded_string = body.decode('utf-8')
        json_object = json.loads(decoded_string)
        payload = json_object[1]
              
        # execute the task
        prompt = payload["nodes"]
        valid =  self.validate_prompt(prompt)
        
        # Prepare task_info
        prompt_id = str(uuid.uuid4())
        outputs_to_execute = valid[2]
        task_info = {
            "prompt_id": prompt_id,
            "prompt": prompt,
            "outputs_to_execute": outputs_to_execute,
            "client_id": "memedeck-1",
            "is_memedeck": True,
            "websocket_node_id": None,
            "ws_id": payload["source_ws_id"],
            "context": payload["req_ctx"],
            "current_node": None,
            "current_progress": 0,
            "delivery_tag": method.delivery_tag,
            "task_status": "waiting",
        }

        # Find the websocket_node_id
        for node in prompt:
            if isinstance(prompt[node], dict) and prompt[node].get("class_type") == "SaveImageWebsocket":
                task_info['websocket_node_id'] = node
                break

        if valid[0]:
            # Enqueue the task into the internal job queue
            self.loop.call_soon_threadsafe(self.internal_job_queue.put_nowait, (prompt_id, prompt, task_info))
            self.logger.info(f"[memedeck]: Enqueued task for {task_info['ws_id']}")
        else:
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False) # unack the message

    async def process_job_queue(self):
        while True:
            prompt_id, prompt, task_info = await self.internal_job_queue.get()
            # Start a new coroutine for each task
            self.loop.create_task(self.process_task(prompt_id, prompt, task_info))

    async def process_task(self, prompt_id, prompt, task_info):
        ws_id = task_info['ws_id']
        # Add the task to tasks_by_ws_id
        self.tasks_by_ws_id[ws_id] = task_info
        # Put the prompt into the prompt_queue
        self.prompt_queue.put((0, prompt_id, prompt, { 
                "client_id": task_info["client_id"], 
                'is_memedeck': task_info['is_memedeck'], 
                'websocket_node_id': task_info['websocket_node_id'], 
                'ws_id': task_info['ws_id'], 
                'context': task_info['context'] 
            }, task_info['outputs_to_execute']))
        # Acknowledge the message
        self.channel.basic_ack(delivery_tag=task_info["delivery_tag"]) # ack the task  
        self.logger.info(f"[memedeck]: Acked task {prompt_id} {ws_id}")      

        self.logger.info(f"[memedeck]: Started processing prompt {prompt_id}")
        # Wait until the current task is completed
        await self.wait_for_task_completion(ws_id)
        # Task is done
        self.internal_job_queue.task_done()

    async def wait_for_task_completion(self, ws_id):
        """
        Wait until the task with the given ws_id is completed.
        """
        while ws_id in self.tasks_by_ws_id:
            await asyncio.sleep(0.5)         

    # --------------------------------------------------
    # callbacks for the prompt queue
    # --------------------------------------------------
    def queue_updated(self):
        info = self.get_queue_info()
        # self.send_sync("status", { "status": self.get_queue_info() })
          
    def get_queue_info(self):
        prompt_info = {}
        exec_info = {}
        exec_info['queue_remaining'] = self.prompt_queue.get_tasks_remaining()
        prompt_info['exec_info'] = exec_info
        return prompt_info
    
    def send_sync(self, event, data, sid=None):
        self.loop.call_soon_threadsafe(
            self.messages.put_nowait, (event, data, sid))
          
    async def publish_loop(self):
        while True:
            msg = await self.messages.get()
            await self.send(*msg)
            
    async def send(self, event, data, sid=None):
        if sid is None:
            self.logger.warning("Received event without sid")
            return

        # Retrieve the task based on sid
        task = self.tasks_by_ws_id.get(sid)
        if not task:
            self.logger.warning(f"Received event {event} for unknown sid: {sid}")
            return

        if event == MemedeckWorker.BinaryEventTypes.UNENCODED_PREVIEW_IMAGE:
            await self.send_preview(
                data,
                sid=sid,
                progress=task['current_progress'],
                context=task['context']
            )
        else:
            # Send JSON data / text data
            if event == "executing":
                task['current_node'] = data['node']
                task["task_status"] = "executing"
            elif event == "progress":
                if task['current_node'] == task['websocket_node_id']:
                    # If the node is the websocket node, then set the progress to 100
                    task['current_progress'] = 100
                else:
                    # If the node is not the websocket node, then set the progress based on the node's progress
                    task['current_progress'] = (data['value'] / data['max']) * 100
                    if task['current_progress'] == 100 and task['current_node'] != task['websocket_node_id']:
                        # In case the progress is 100 but the node is not the websocket node, set progress to 95
                        task['current_progress'] = 95  # Allows the full resolution image to be sent on the 100 progress event

                if data['value'] == 1:
                    # If the value is 1, send started to API
                    start_data = {
                        "ws_id": task['ws_id'],
                        "status": "started",
                        "info": None,
                    }
                    task["task_status"] = "executing"
                    await self.send_to_api(start_data)

            elif event == "status":
                self.logger.info(f"[memedeck]: sending status event: {data}")

            # Update the task in tasks_by_ws_id
            self.tasks_by_ws_id[sid] = task

    async def send_preview(self, image_data, sid=None, progress=None, context=None):
        if sid is None:
            self.logger.warning("Received preview without sid")
            return

        task = self.tasks_by_ws_id.get(sid)
        if not task:
            self.logger.warning(f"Received preview for unknown sid: {sid}")
            return

        if progress is None:
            progress = task['current_progress']

        # if progress is odd, then don't send the preview
        if int(progress) % 2 == 1:
            return
                
        image_type = image_data[0]
        image = image_data[1]
        max_size = image_data[2]
        if max_size is not None:
            if hasattr(Image, 'Resampling'):
                resampling = Image.Resampling.BILINEAR
            else:
                resampling = Image.ANTIALIAS

            image = ImageOps.contain(image, (max_size, max_size), resampling)

        bytesIO = BytesIO()
        image.save(bytesIO, format=image_type, quality=100 if progress == 95 else 75, compress_level=1)
        preview_bytes = bytesIO.getvalue()
            
        ai_queue_progress = {
            "ws_id": sid,
            "kind": "image_generating" if progress < 100 else "image_generated",
            "data": list(preview_bytes),
            "progress": int(progress),
            "context": context
        }
                   
        await self.send_to_api(ai_queue_progress)
        
        if progress == 100:
            del self.tasks_by_ws_id[sid] # Remove the task from tasks_by_ws_id
            self.logger.info(f"[memedeck]: Task {sid} completed")

    async def send_to_api(self, data):
        ws_id = data.get('ws_id')
        if not ws_id:
            self.logger.error("[memedeck]: Missing ws_id in data")
            return
        task = self.tasks_by_ws_id.get(ws_id)
        if not task:
            self.logger.error(f"[memedeck]: No task found for ws_id {ws_id}")
            return
        if task['websocket_node_id'] is None:
            self.logger.error(f"[memedeck]: websocket_node_id is None for {ws_id}")
            return
        try:
            post_func = partial(requests.post, f"{self.api_url}/generation/update", json=data)        
            await self.loop.run_in_executor(None, post_func)
        except Exception as e:
            self.logger.error(f"[memedeck]: error sending to api: {e}")
    
# --------------------------------------------------------------------------
# MemedeckAzureStorage
# --------------------------------------------------------------------------
# from azure.storage.blob.aio import BlobClient, BlobServiceClient
# from azure.storage.blob import ContentSettings  
# from typing import Optional, Tuple
# import cairosvg

# WATERMARK = '<svg width="256" height="256" viewBox="0 0 256 256" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M60.0859 196.8C65.9526 179.067 71.5526 161.667 76.8859 144.6C79.1526 137.4 81.4859 129.867 83.8859 122C86.2859 114.133 88.6859 106.333 91.0859 98.6C93.4859 90.8667 95.6859 83.4 97.6859 76.2C99.8193 69 101.686 62.3333 103.286 56.2C110.619 56.2 117.553 55.8 124.086 55C130.619 54.2 137.686 53.4667 145.286 52.8C144.886 55.7333 144.419 59.0667 143.886 62.8C143.486 66.4 142.953 70.2 142.286 74.2C141.753 78.2 141.153 82.3333 140.486 86.6C139.819 90.8667 139.019 96.3333 138.086 103C137.153 109.667 135.886 118 134.286 128H136.886C140.753 117.867 143.953 109.467 146.486 102.8C149.019 96 151.086 90.4667 152.686 86.2C154.286 81.9333 155.886 77.8 157.486 73.8C159.219 69.6667 160.819 65.8 162.286 62.2C163.886 58.4667 165.353 55.2 166.686 52.4C170.019 52.1333 173.153 51.8 176.086 51.4C179.019 51 181.953 50.6 184.886 50.2C187.819 49.6667 190.753 49.2 193.686 48.8C196.753 48.2667 200.086 47.6667 203.686 47C202.353 54.7333 201.086 62.6667 199.886 70.8C198.686 78.9333 197.619 87.0667 196.686 95.2C195.753 103.2 194.819 111.133 193.886 119C193.086 126.867 192.353 134.333 191.686 141.4C190.086 157.933 188.686 174.067 187.486 189.8L152.686 196C152.686 195.333 152.753 193.533 152.886 190.6C153.153 187.667 153.419 184.067 153.686 179.8C154.086 175.533 154.553 170.8 155.086 165.6C155.753 160.4 156.353 155.2 156.886 150C157.553 144.8 158.219 139.8 158.886 135C159.553 130.067 160.219 125.867 160.886 122.4H159.086C157.219 128 155.153 133.933 152.886 140.2C150.619 146.333 148.286 152.6 145.886 159C143.619 165.4 141.353 171.667 139.086 177.8C136.819 183.933 134.819 189.8 133.086 195.4C128.419 195.533 124.419 195.733 121.086 196C117.753 196.133 113.886 196.333 109.486 196.6L115.886 122.4H112.886C112.619 124.133 111.953 127.067 110.886 131.2C109.819 135.2 108.553 139.867 107.086 145.2C105.753 150.4 104.286 155.867 102.686 161.6C101.086 167.2 99.5526 172.467 98.0859 177.4C96.7526 182.2 95.6193 186.2 94.6859 189.4C93.7526 192.467 93.2193 194.2 93.0859 194.6L60.0859 196.8Z" fill="white"/></svg>'
# WATERMARK_SIZE = 40 

# class MemedeckAzureStorage:
#     def __init__(self, connection_string):
#         # get environment variables
#         self.storage_account = os.getenv('STORAGE_ACCOUNT')
#         self.storage_access_key = os.getenv('STORAGE_ACCESS_KEY')
#         self.storage_container = os.getenv('STORAGE_CONTAINER')
#         self.logger = logging.getLogger(__name__)

#         self.blob_service_client = BlobServiceClient.from_connection_string(conn_str=connection_string)

#     async def upload_image(
#         self,
#         by: str,
#         image_id: str,
#         source_url: Optional[str],
#         bytes_data: Optional[bytes],
#         filetype: Optional[str],
#     ) -> Tuple[str, Tuple[int, int]]:
#         """
#         Uploads an image to Azure Blob Storage.

#         Args:
#             by (str): Identifier for the uploader.
#             image_id (str): Unique identifier for the image.
#             source_url (Optional[str]): URL to fetch the image from.
#             bytes_data (Optional[bytes]): Image data in bytes.
#             filetype (Optional[str]): Desired file type (e.g., 'jpeg', 'png').

#         Returns:
#             Tuple[str, Tuple[int, int]]: URL of the uploaded image and its dimensions.
#         """
#         # Retrieve image bytes either from the provided bytes_data or by fetching from source_url
#         if source_url is None:
#             if bytes_data is None:
#                 raise ValueError("Could not get image bytes")
#             image_bytes = bytes_data
#         else:
#             self.logger.info(f"Requesting image from URL: {source_url}")
#             async with aiohttp.ClientSession() as session:
#                 try:
#                     async with session.get(source_url) as response:
#                         if response.status != 200:
#                             raise Exception(f"Failed to fetch image, status code {response.status}")
#                         image_bytes = await response.read()
#                 except Exception as e:
#                     raise Exception(f"Error fetching image from URL: {e}")

#         # Open image using Pillow to get dimensions and format
#         try:
#             img = Image.open(BytesIO(image_bytes))
#             width, height = img.size
#             inferred_filetype = img.format.lower()
#         except Exception as e:
#             raise Exception(f"Failed to decode image: {e}")

#         # Determine the final file type
#         final_filetype = filetype.lower() if filetype else inferred_filetype

#         # Construct the blob name
#         blob_name = f"{by}/{image_id.replace('image:', '')}.{final_filetype}"

#         # Upload the image to Azure Blob Storage
#         try:
#             image_url = await self.save_image(blob_name, img.format, image_bytes)
#             return image_url, (width, height)
#         except Exception as e:
#             self.logger.error(f"Trouble saving image: {e}")
#             raise Exception(f"Trouble saving image: {e}")

#     async def save_image(
#         self,
#         blob_name: str,
#         content_type: str,
#         bytes_data: bytes
#     ) -> str:
#         """
#         Saves image bytes to Azure Blob Storage.

#         Args:
#             blob_name (str): Name of the blob in Azure Storage.
#             content_type (str): MIME type of the content.
#             bytes_data (bytes): Image data in bytes.

#         Returns:
#             str: URL of the uploaded blob.
#         """
#         # Retrieve environment variables
#         account = os.getenv("STORAGE_ACCOUNT")
#         access_key = os.getenv("STORAGE_ACCESS_KEY")
#         container = os.getenv("STORAGE_CONTAINER")

#         if not all([account, access_key, container]):
#             raise EnvironmentError("Missing STORAGE_ACCOUNT, STORAGE_ACCESS_KEY, or STORAGE_CONTAINER environment variables")

#         # Initialize BlobServiceClient
#         blob_service_client = BlobServiceClient(
#             account_url=f"https://{account}.blob.core.windows.net",
#             credential=access_key
#         )
#         blob_client = blob_service_client.get_blob_client(container=container, blob=blob_name)

#         # Upload the blob
#         try:
#             await blob_client.upload_blob(
#                 bytes_data,
#                 overwrite=True,
#                 content_settings=ContentSettings(content_type=content_type)
#             )
#         except Exception as e:
#             raise Exception(f"Failed to upload blob: {e}")

#         self.logger.debug(f"Blob uploaded: name={blob_name}, content_type={content_type}")

#         # Construct and return the blob URL
#         blob_url = f"https://media.memedeck.xyz//{container}/{blob_name}"
#         return blob_url

#     async def add_watermark(
#         self,
#         base_blob_name: str,
#         base_image: bytes
#     ) -> str:
#         """
#         Adds a watermark to the provided image and uploads the watermarked image.

#         Args:
#             base_blob_name (str): Original blob name of the image.
#             base_image (bytes): Image data in bytes.

#         Returns:
#             str: URL of the watermarked image.
#         """
#         # Load the input image
#         try:
#             img = Image.open(BytesIO(base_image)).convert("RGBA")
#         except Exception as e:
#             raise Exception(f"Failed to load image: {e}")

#         # Calculate position for the watermark (bottom right corner with padding)
#         padding = 12
#         x = img.width - WATERMARK_SIZE - padding
#         y = img.height - WATERMARK_SIZE - padding

#         # Analyze background brightness where the watermark will be placed
#         background_brightness = self.analyze_background_brightness(img, x, y, WATERMARK_SIZE)
#         self.logger.info(f"Background brightness: {background_brightness}")

#         # Render SVG watermark to PNG bytes using cairosvg
#         try:
#             watermark_png_bytes = cairosvg.svg2png(bytestring=WATERMARK.encode('utf-8'), output_width=WATERMARK_SIZE, output_height=WATERMARK_SIZE)
#             watermark = Image.open(BytesIO(watermark_png_bytes)).convert("RGBA")
#         except Exception as e:
#             raise Exception(f"Failed to render watermark SVG: {e}")

#         # Determine watermark color based on background brightness
#         if background_brightness > 128:
#             # Dark watermark for light backgrounds
#             watermark_color = (0, 0, 0, int(255 * 0.65))  # Black with 65% opacity
#         else:
#             # Light watermark for dark backgrounds
#             watermark_color = (255, 255, 255, int(255 * 0.65))  # White with 65% opacity

#         # Apply the watermark color by blending
#         solid_color = Image.new("RGBA", watermark.size, watermark_color)
#         watermark = Image.alpha_composite(watermark, solid_color)

#         # Overlay the watermark onto the original image
#         img.paste(watermark, (x, y), watermark)

#         # Save the watermarked image to bytes
#         buffer = BytesIO()
#         img = img.convert("RGB")  # Convert back to RGB for JPEG format
#         img.save(buffer, format="JPEG")
#         buffer.seek(0)
#         jpeg_bytes = buffer.read()

#         # Modify the blob name to include '_watermarked'
#         try:
#             if "memes/" in base_blob_name:
#                 base_blob_name_right = base_blob_name.split("memes/", 1)[1]
#             else:
#                 base_blob_name_right = base_blob_name
#             base_blob_name_split = base_blob_name_right.rsplit(".", 1)
#             base_blob_name_without_extension = base_blob_name_split[0]
#             extension = base_blob_name_split[1]
#         except Exception as e:
#             raise Exception(f"Failed to process blob name: {e}")

#         watermarked_blob_name = f"{base_blob_name_without_extension}_watermarked.{extension}"

#         # Upload the watermarked image
#         try:
#             watermarked_blob_url = await self.save_image(
#                 watermarked_blob_name,
#                 "image/jpeg",
#                 jpeg_bytes
#             )
#             return watermarked_blob_url
#         except Exception as e:
#             raise Exception(f"Failed to upload watermarked image: {e}")

#     def analyze_background_brightness(
#         self,
#         img: Image.Image,
#         x: int,
#         y: int,
#         size: int
#     ) -> int:
#         """
#         Analyzes the brightness of a specific region in the image.

#         Args:
#             img (Image.Image): The image to analyze.
#             x (int): X-coordinate of the top-left corner of the region.
#             y (int): Y-coordinate of the top-left corner of the region.
#             size (int): Size of the square region to analyze.

#         Returns:
#             int: Average brightness (0-255) of the region.
#         """
#         # Crop the specified region
#         sub_image = img.crop((x, y, x + size, y + size)).convert("RGB")

#         # Calculate average brightness using the luminance formula
#         total_brightness = 0
#         pixel_count = 0
#         for pixel in sub_image.getdata():
#             r, g, b = pixel
#             brightness = (r * 299 + g * 587 + b * 114) // 1000
#             total_brightness += brightness
#             pixel_count += 1

#         if pixel_count == 0:
#             return 0

#         average_brightness = total_brightness // pixel_count
#         return average_brightness

