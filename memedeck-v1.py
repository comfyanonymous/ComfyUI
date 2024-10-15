import asyncio
import base64
from io import BytesIO
import os
import logging
import signal
import struct
from typing import Optional
import uuid
from PIL import Image, ImageOps
from functools import partial

import pika
import json

import requests
import aiohttp

# load from env file
# load from .env file
from dotenv import load_dotenv
load_dotenv()

amqp_addr = os.getenv('AMQP_ADDR') or 'amqp://api:gacdownatravKekmy9@51.8.120.154:5672/dev'

# define the enum in python
from enum import Enum

class QueueProgressKind(Enum):
    # make json serializable
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
        # set logging level to info 
        logging.getLogger().setLevel(logging.INFO)
        self.active_tasks_map = {}
        self.current_task = None
        
        self.client_id = None
        self.ws_id = None
        self.websocket_node_id = None
        self.current_node = None
        self.current_progress = 0
        self.current_context = None
        
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
        self.channel.basic_consume(queue=self.queue_name, on_message_callback=self.on_message_received)
        
    def start(self, prompt_queue, validate_prompt):
      self.prompt_queue = prompt_queue
      self.validate_prompt = validate_prompt
      
      parameters = pika.URLParameters(self.amqp_url)
      logging.getLogger('pika').setLevel(logging.WARNING) # supress all logs from pika
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
        
        self.current_node = None
        self.current_progress = 0
        self.websocket_node_id = None
        self.ws_id = payload["source_ws_id"]
        self.current_context = payload["req_ctx"]
        
        for node in prompt: # search through prompt nodes for websocket_node_id
            if isinstance(prompt[node], dict) and prompt[node].get("class_type") == "SaveImageWebsocket":
                self.websocket_node_id = node
                break
            
        if valid[0]:
          prompt_id = str(uuid.uuid4())
          outputs_to_execute = valid[2]
          self.active_tasks_map[payload["source_ws_id"]] = {
            "prompt_id": prompt_id,
            "prompt": prompt,
            "outputs_to_execute": outputs_to_execute,
            "client_id": "memedeck-1",
            "is_memedeck": True,
            "websocket_node_id": self.websocket_node_id,
            "ws_id": payload["source_ws_id"],
            "context": payload["req_ctx"],
            "current_node": None,
            "current_progress": 0,
          }
          self.prompt_queue.put((0, prompt_id, prompt, { 
                "client_id": "memedeck-1", 
                'is_memedeck': True, 
                'websocket_node_id': self.websocket_node_id, 
                'ws_id': payload["source_ws_id"], 
                'context': payload["req_ctx"] 
            }, outputs_to_execute))
          self.set_last_prompt_id(prompt_id)
          channel.basic_ack(delivery_tag=method.delivery_tag) # ack the task        
        else:
          channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False) # unack the message
    
    # --------------------------------------------------
    # callbacks for the prompt queue
    # --------------------------------------------------
    def queue_updated(self):
        # print json of the queue info but only print the first 100 lines
        info = self.get_queue_info()
        # update_type = info['']
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
        
    def set_last_prompt_id(self, prompt_id):
        self.last_prompt_id = prompt_id
          
    async def publish_loop(self):
        while True:
            msg = await self.messages.get()
            await self.send(*msg)
            
    async def send(self, event, data, sid=None):
        current_task = self.active_tasks_map.get(sid)
        if current_task is None or current_task['ws_id'] != sid:
            return
        
        if event == MemedeckWorker.BinaryEventTypes.UNENCODED_PREVIEW_IMAGE: # preview and unencoded images are sent here
            self.logger.info(f"[memedeck]: sending image preview for {sid}")
            await self.send_preview(data, sid=current_task['ws_id'], progress=current_task['current_progress'], context=current_task['context'])
        else: # send json data / text data
            if event == "executing":
                current_task['current_node'] = data['node']   
            elif event == "executed":
                self.logger.info(f"---> [memedeck]: executed event for {sid}")
                prompt_id = data['prompt_id']
                if prompt_id in self.active_tasks_map:
                    del self.active_tasks_map[prompt_id]
            elif event == "progress":
                if current_task['current_node'] == current_task['websocket_node_id']: # if the node is the websocket node, then set the progress to 100
                    current_task['current_progress'] = 100
                else: # if the node is not the websocket node, then set the progress to the progress from the node
                    current_task['current_progress'] = data['value'] / data['max'] * 100
                    if current_task['current_progress'] == 100 and current_task['current_node'] != current_task['websocket_node_id']:
                        # in case the progress is 100 but the node is not the websocket node, then set the progress to 95
                        current_task['current_progress'] = 95 # this allows the full resolution image to be sent on the 100 progress event
                
                if data['value'] == 1: # if the value is 1, then send started to api
                    start_data = {
                        "ws_id": current_task['ws_id'],
                        "status": "started",
                        "info": None,
                    }
                    await self.send_to_api(start_data)
                    
            elif event == "status":
                self.logger.info(f"[memedeck]: sending status event: {data}")
            
        self.active_tasks_map[sid] = current_task

        
    async def send_preview(self, image_data, sid=None, progress=None, context=None):
        # if self.current_progress is odd, then don't send the preview
        if progress % 2 == 1:
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
        image.save(bytesIO, format=image_type, quality=100 if progress == 96 else 75, compress_level=1)
        preview_bytes = bytesIO.getvalue()
            
        ai_queue_progress = {
            "ws_id": sid,
            "kind": "image_generating" if progress < 100 else "image_generated",
            "data": list(preview_bytes),
            "progress": int(progress),
            "context": context
        }
       
        await self.send_to_api(ai_queue_progress)
        
    async def send_to_api(self, data):
        if self.websocket_node_id is None: # check if the node is still running
            logging.error(f"[memedeck]: websocket_node_id is None for {data['ws_id']}")
            return
         
        try:
            post_func = partial(requests.post, f"{self.api_url}/generation/update", json=data)        
            await self.loop.run_in_executor(None, post_func)
        except Exception as e:
            self.logger.error(f"[memedeck]: error sending to api: {e}")
