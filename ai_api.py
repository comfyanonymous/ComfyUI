import os, json, io
import urllib
import magic
from requests_toolbelt import MultipartEncoder


# Assume mime type
def get_mime_type(file_path):
    mime = magic.Magic(mime=True)
    mime_type = mime.from_file(file_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'
    return mime_type

def get_mime_type_from_binary(binary_data):
    mime = magic.Magic(mime=True)
    mime_type = mime.from_buffer(binary_data)
    if mime_type is None:
        mime_type = 'application/octet-stream'
    return mime_type


# API
def queue_prompt(prompt, client_id, server_address):
    p = {"prompt": prompt, "client_id": client_id}
    headers = {'Content-Type': 'application/json'}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request(f"http://{server_address}/prompt", data=data, headers=headers)
    return json.loads(urllib.request.urlopen(req).read())

def get_queue_state(server_address):
    with urllib.request.urlopen(f"http://{server_address}/queue") as response:
        return json.loads(response.read())

def get_history(client_id, server_address):
    with urllib.request.urlopen(f"http://{server_address}/history/{client_id}") as response:
        return json.loads(response.read())

def upload_image(file_data, name, server_address, image_type="input", overwrite=False):

    mime_type = get_mime_type_from_binary(file_data)
    multipart_data = MultipartEncoder(
      fields={
        'image': (name, file_data, mime_type),
        'type': image_type,
        'overwrite': str(overwrite).lower()
      }
    )
    headers = {'Content-Type': multipart_data.content_type}
    
    request = urllib.request.Request(f"http://{server_address}/upload/image", data=multipart_data, headers=headers)
    with urllib.request.urlopen(request) as response:
      return response.read()

# Parsing text
def get_parsed_input_nodes(workflow_json):
    if isinstance(workflow_json, str):
        with open(workflow_json, mode="r") as f:
            workflow_json = json.load(f)

    parsed_input_nodes = {}

    for node_number in workflow_json:
        favorfit_inputs = []

        cur_node = workflow_json[node_number]
        meta_data = cur_node["_meta"].get("Favorfitinput", None)
        if meta_data is not None: favorfit_inputs.extend(meta_data.split(","))
        # if "seed" in cur_node["inputs"].keys(): favorfit_inputs.append("seed")
        
        for favorfit_input in favorfit_inputs:
            if favorfit_input is not None:
                input_value = cur_node["inputs"].get(favorfit_input, None)
                input_type = type(input_value)

                if (input_type == None) or (input_value == ''):
                    raise ValueError(f"{node_number}:{favorfit_input}, is wrong data('{input_type}', '{input_value}')")
                
                parsed_input_nodes[f"{node_number}/{favorfit_input}"] = {
                    "type":input_type.__name__, 
                    "title":cur_node["_meta"]["title"], 
                    "class":cur_node["class_type"],
                    "default":cur_node["inputs"][favorfit_input]
                }
    return parsed_input_nodes
    
def parse_workflow_prompt(workflow_path, **kwargs):

    with open(workflow_path, mode="r") as f:
        workflow_json = json.load(f)
    
    parsed_input_nodes = get_parsed_input_nodes(workflow_json)
    prompt = workflow_json

    for node_id_and_key, node_info in parsed_input_nodes.items():
        node_id, input_key = node_id_and_key.split("/")
        
        input_value = kwargs.get(node_id_and_key, None)
        if input_value is not None:
            if type(input_value).__name__ != node_info["type"]:
                raise ValueError(f"'{node_id_and_key}' need to have type of {node_info["type"]} but got {type(input_value)} from {input_value})")
        else:
            input_value = node_info.get("default", None)
        prompt[node_id]["inputs"][input_key] = input_value
 
    return prompt

def parse_outputs(outputs:dict):
    file_paths, mime_types, file_contents = [], [], []

    output_nodes = list(outputs.values())
    for output_node in output_nodes:
        for key, values in output_node.items():
            for value in values:
                if not isinstance(value, dict): continue
                if value.get("filename", None) ==  None: continue

                file_path = os.path.join("./output", value["filename"])
                if not os.path.isfile(file_path): continue
                mime_type = get_mime_type(file_path)
                with open(file_path, 'rb') as f:
                    file_content = f.read()

                file_paths.append(file_path)
                mime_types.append(mime_type)
                file_contents.append(file_content)

    return file_paths, mime_types, file_contents


# File system
def save_binary_file(data, file_name, directory='./input', return_root_dir=False):
    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Split the file name into base name and extension
    base_name, extension = os.path.splitext(file_name)
    new_file_name = file_name

    # Check if the file exists and find a new name if it does
    counter = 1
    while os.path.exists(os.path.join(directory, new_file_name)):
        new_file_name = f"{base_name} ({counter}){extension}"
        counter += 1

    # Save the binary data to the file
    save_path = os.path.join(directory, new_file_name)
    with open(save_path, 'wb') as file:
        file.write(data)

    if return_root_dir==True:
        return save_path
    else:
        return new_file_name
