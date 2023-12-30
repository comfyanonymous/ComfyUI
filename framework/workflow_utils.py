
import os
import io
import uuid
import aiohttp
import imghdr
import urllib

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
                    inp_name = node_class.INPUT_NODE_DATA
                    node_info["inputs"][inp_name] = flow_inputs[flow_input_name]
                    if inp_name in node_info["is_input_linked"]:
                        node_info["is_input_linked"][inp_name] = False
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
    def upload_file(filename):
        basename = os.path.basename(filename)
        # upload
        remote_dir = CONFIG["resource"]["in_img_path_cloud"]
        remote_name = f"{remote_dir}/{basename}"

        object_storage.MinIOConnection().fput_object(remote_name, filename)
        return remote_name        
        
        
    @staticmethod
    def extract_filename_from_url(url):
        parsed_url = urllib.parse.urlparse(url)
        filename = os.path.basename(parsed_url.path)
        return filename