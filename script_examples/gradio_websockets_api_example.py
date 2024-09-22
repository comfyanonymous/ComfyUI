# This is a Gradio example demonstrating using the websocket api and that also decodes preview images
# Gradio has a lot of idiosyncrasies and I'm definitely not an expert at coding for it
# I'm sure there are a million and one better ways to code this, but this works pretty well and should get you started
# I suggest taking the time to check any relevant comments throughout the code
# For more info on working with Gradio: https://www.gradio.app/docs

# Ensure that ComfyUI has latent previews enabled
# If you use Comfy Manager, make sure to set the preview type there because it will override --preview-method auto/latent2rgb/taesd launch flag settings
# Check or change the preview_method in "/custom_nodes/ComfyUI-Manager/config.ini"

# If you chose to install Gradio to your ComfyUI python venv, open a command prompt in this script_examples directory and run:
# ..\..\python_embeded\python.exe -s ..\script_examples\gradio_websockets_api_example.py
# To launch the app

import websocket #NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import json
import urllib.request
import urllib.parse
from PIL import Image
import io
from io import BytesIO
import random

#If you want to use your local ComfyUI python installation, you'll need to navigate to your comfyui/python_embeded folder, open a cmd prompt and run "python.exe -m pip install gradio"
import gradio as gr

# adjust to your ComfyUI API settings
server_address = "127.0.0.1:8188"
client_id = str(uuid.uuid4())

#some globals to store previews, active state and progress
preview_image = None
active = False
interrupted = False
step_current = None
step_total = None

def interrupt_diffusion():
    global interrupted, step_current, step_total
    interrupted = True
    step_current = None
    step_total = None
    req = urllib.request.Request("http://{}/interrupt".format(server_address), method='POST')
    return urllib.request.urlopen(req)

def queue_prompt(prompt):
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request("http://{}/prompt".format(server_address), data=data)
    return json.loads(urllib.request.urlopen(req).read())

def get_image(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen("http://{}/view?{}".format(server_address, url_values)) as response:
        return response.read()

def get_history(prompt_id):
    with urllib.request.urlopen("http://{}/history/{}".format(server_address, prompt_id)) as response:
        return json.loads(response.read())

def get_images(ws, prompt):
    global preview_image, active, step_current, step_total
    prompt_id = queue_prompt(prompt)['prompt_id']
    output_images = {}
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    preview_image = None #clear these globals on completion just in case
                    step_current = None
                    step_total = None
                    active = False
                    break #Execution is done
            elif message['type'] == 'progress':
                data = message['data']
                step_current = data['value']
                step_total = data['max']
        else:
            bytesIO = BytesIO(out[8:])
            preview_image = Image.open(bytesIO) # This is your preview in PIL image format

    history = get_history(prompt_id)[prompt_id]
    for node_id in history['outputs']:
        node_output = history['outputs'][node_id]
        images_output = []
        if 'images' in node_output:
            for image in node_output['images']:
                image_data = get_image(image['filename'], image['subfolder'], image['type'])
                images_output.append(image_data)
        output_images[node_id] = images_output

    return output_images

def get_prompt_images(prompt):
    global preview_image
    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))
    images = get_images(ws, prompt)
    outputs = []
    for node_id in images:
        for image_data in images[node_id]:
            image = Image.open(io.BytesIO(image_data))
            outputs.append(image)
    ws.close()
    return outputs

############################################################################################################################
# Edit or add your own api workflow here. Make sure to enable dev mode in ComfyUI and to use the "Save(API Format)" option #
############################################################################################################################
prompt_text = """
{
    "3": {
        "class_type": "KSampler",
        "inputs": {
            "cfg": 8,
            "denoise": 1,
            "latent_image": [
                "5",
                0
            ],
            "model": [
                "4",
                0
            ],
            "negative": [
                "7",
                0
            ],
            "positive": [
                "6",
                0
            ],
            "sampler_name": "euler",
            "scheduler": "normal",
            "seed": -1,
            "steps": 25
        }
    },
    "4": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {
            "ckpt_name": "sdxl_base_1.0_0.9vae.safetensors"
        }
    },
    "5": {
        "class_type": "EmptyLatentImage",
        "inputs": {
            "batch_size": 1,
            "height": 1024,
            "width": 1024
        }
    },
    "6": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "clip": [
                "4",
                1
            ],
            "text": ""
        }
    },
    "7": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "clip": [
                "4",
                1
            ],
            "text": ""
        }
    },
    "8": {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": [
                "3",
                0
            ],
            "vae": [
                "4",
                2
            ]
        }
    },
    "9": {
        "class_type": "SaveImage",
        "inputs": {
            "filename_prefix": "ComfyUI",
            "images": [
                "8",
                0
            ]
        }
    }
}
"""

prompt = json.loads(prompt_text)

# You can also use the following if you'd rather just load a json, make sure to comment out or remove the line above
# with open("/path/to/workflow.json", "r", encoding="utf-8") as f:
    # prompt = json.load(f)

# start and stop timer are used for live updating the preview and progress
# no point in keeping the timer ticking if it's not currently generating
def start_timer(): 
    global active
    active = True
    return gr.Timer(active=True)
    
def stop_timer():
    global active
    active = False
    return gr.Timer(active=False)

def update_preview():
    return gr.Image(value=preview_image)

# Gradio is somewhat finicky about multiple things trying to change the same output, so we switch between preview and image, while hiding the other
def window_preview():
    return gr.Image(visible=False, value=None), gr.Image(visible=True, value=None), gr.Button(visible=False),  gr.Button(visible=True, value="Stop: Busy")

def window_final():
    if interrupted: #if we interrupted during the process, put things back to normal
        return gr.Image(visible=True, value=None), gr.Image(visible=False), gr.Button(visible=True), gr.Button(visible=False)
    else:
        return gr.Image(visible=True), gr.Image(visible=False, value=None), gr.Button(visible=True), gr.Button(visible=False)

# Puts the progress on the stop button
def update_progress():
    if step_current == 0 or step_current == None:
        x = 0
    else:
        x = int(100 * (step_current / step_total))
    if step_current == None or active == False:
        message = "Stop: Busy"
    else:
        message = f"Stop: {step_current} / {step_total} steps {x}%"
    return gr.Button(value=message)

# You will need to do a lot of editing here to match your workflow
def process(pos, neg, width, height, cfg, seed):
    if seed <= -1:
        seed = random.randint(0, 999999999)
    prompt["4"]["inputs"]["ckpt_name"] = "sdxl_base_1.0_0.9vae.safetensors" #if you want to change the model, do it here
    prompt["6"]["inputs"]["text"] = pos
    prompt["7"]["inputs"]["text"] = neg
    prompt["3"]["inputs"]["seed"] = seed
    prompt["3"]["inputs"]["cfg"] = cfg
    prompt["5"]["inputs"]["height"] = height
    prompt["5"]["inputs"]["width"] = width
    
    global interrupted
    interrupted = False
    
    images = get_prompt_images(prompt)
    
    global active
    active = False

    try:
      return gr.Image(value=images[0]) #not covering batch generations in this example because it requires setting the image output to a gr.Gallery, along with some other changes
    except:
      return gr.Image()

with gr.Blocks(analytics_enabled=False, fill_width=True, fill_height=True,) as example:
    preview_timer = gr.Timer(value=1, active=False) # You can also lower the timer to something like 0.5 to get more frequent updates, but there's not really much point to it
    with gr.Row():
        with gr.Column():
            with gr.Group():
                user_prompt = gr.Textbox(label="Positive Prompt: ", value="orange cat, full moon, vibrant impressionistic painting, bright vivid rainbow of colors", lines=5, max_lines=20)
                user_negativeprompt = gr.Textbox(label="Negative Prompt: ", value="text, watermark", lines=2, max_lines=10,)
            with gr.Group():
                with gr.Row():
                    user_width = gr.Slider(label="Width", minimum=512, maximum=1600, step=64, value=1152,)
                    user_height = gr.Slider(label="Height", minimum=512, maximum=1600, step=64, value=896,)
                with gr.Row():
                    user_cfg = gr.Slider(label="CFG: ", minimum=1.0, maximum=16.0, step=0.1, value=4.5,)
                    user_seed = gr.Slider(label="Seed: (-1 for random)", minimum=-1, maximum=999999999, step=1, value=-1,)   
            generate = gr.Button("Generate", variant="primary")
            stop = gr.Button("Stop", variant="stop", visible=False)
        with gr.Column():    
            output_image = gr.Image(label="Image: ", type="pil", format="jpeg", interactive=False, visible=True)            
            output_preview = gr.Image(label="Preview: ", type="pil", format="jpeg", interactive=False, visible=False)
    
    # On tick, we update the preview and then the progress
    preview_timer.tick(
        fn=update_preview, outputs=output_preview, show_progress="hidden").then(
        fn=update_progress, outputs=stop, show_progress="hidden")
    
    # On generate we switch windows/buttons, start the update tick, diffuse the image, stop the update tick and then finally, swap the image outputs/buttons back
    generate.click(
        fn=window_preview, outputs=[output_image, output_preview, generate, stop], show_progress="hidden").then(
        fn=start_timer, outputs=preview_timer, show_progress="hidden").then(
        fn=process, inputs=[user_prompt, user_negativeprompt, user_width, user_height, user_cfg, user_seed], outputs=output_image).then(
        fn=stop_timer, outputs=preview_timer, show_progress="hidden").then(
        fn=window_final, outputs=[output_image, output_preview, generate, stop], show_progress="hidden") 
    
    stop.click(fn=interrupt_diffusion, show_progress="hidden")

# Adjust settings to your needs https://www.gradio.app/docs/gradio/blocks#blocks-launch for more info
example.queue(max_size=2,) # how many users can queue up in line
example.launch(share=False, inbrowser=True, server_name="0.0.0.0", server_port=7860, enable_monitoring=False) # good for LAN-only setups
