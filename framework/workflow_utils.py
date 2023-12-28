
import os
import io
import uuid
import aiohttp
import imghdr
import base64

import nodes
from framework.app_log import AppLog
from framework.model import object_storage
from config.config import CONFIG



class WorkflowUtils:
    
    @staticmethod
    def get_workflow_input_info(prompt):
        # workflow_nodes = workflow["nodes"] if "nodes" in workflow else []
        # for node in workflow_nodes:
        #     node_type = node["type"]
        #     node_class = nodes.NODE_CLASS_MAPPINGS[node_type]
        #     if hasattr(node_class, "INPUT_NODE") and node_class.INPUT_NODE:
        wf_inputs = {}
        prompt_nodes = prompt["prompt"]
        for node_id, node_info in prompt_nodes.items():
            node_type = node_info["class_type"]
            node_class = nodes.NODE_CLASS_MAPPINGS[node_type]
            if hasattr(node_class, "INPUT_NODE") and node_class.INPUT_NODE:
                flow_input_name = node_info["inputs"]["name"]
                flow_input_type = node_class.INPUT_NODE_TYPE
                if flow_input_name in wf_inputs:
                    AppLog.warning(f"[GetWorkflowInput] the same workflow input name was found: {flow_input_name}")
                wf_inputs[flow_input_name] = flow_input_type
        return wf_inputs
    
    
    
    @staticmethod
    def apply_workflow_inputs(prompt, flow_inputs):
        for node_id, node_info in prompt.items():
            node_type = node_info["class_type"]
            node_class = nodes.NODE_CLASS_MAPPINGS[node_type]
            if hasattr(node_class, "INPUT_NODE") and node_class.INPUT_NODE:
                print(f"has INPUT_NODE: {node_type}")
                flow_input_name = node_info["inputs"]["name"]
                if flow_input_name in flow_inputs:
                    node_info["inputs"]["data"] = flow_inputs[flow_input_name]
                    if "data" in node_info["is_input_linked"]:
                        node_info["is_input_linked"]["data"] = False
        return prompt
    
    
                
    @staticmethod
    def is_resource_type(type_name:str):
        return type_name == "IMAGE"
    
    
    @staticmethod
    async def save_image(session, image_file, save_path):
        async with session.get(image_file) as response:
            with open(save_path, 'wb') as f:
                while True:
                    chunk = await response.content.read(1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    
    
    @staticmethod
    async def upload_resource(filefield):
        filename = filefield.filename
        new_name = str(uuid.uuid4())
        file_ext = os.path.splitext(filename)[1]
        if file_ext is not None and file_ext!="":
            new_name = f"{new_name}{file_ext}"
        remote_dir = CONFIG["resource"]["in_img_path_cloud"]
        new_name = f"{remote_dir}/{new_name}"
        print(f"image file name: {new_name}")
        
        file_data = filefield.file.read()

        object_storage.MinIOConnection().put_object(new_name, io.BytesIO(file_data), len(file_data))
    
        return True, new_name
        
        