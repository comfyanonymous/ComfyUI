# Using ComfyUI as an API / Programmatically

There are multiple ways to use this ComfyUI package to run workflows programmatically:

### Embedded

Start ComfyUI by creating an ordinary Python object. This does not create a web server. It runs ComfyUI as a library, like any other package you are familiar with:

```python
from comfy.client.embedded_comfy_client import Comfy

async with Comfy() as client:
    # This will run your prompt
    # To get the prompt JSON, visit the ComfyUI interface, design your workflow and click **Save (API Format)**. This JSON is what you will use as your workflow.
    outputs = await client.queue_prompt(prompt)
    # At this point, your prompt is finished and all the outputs, like saving images, have been completed.
    # Now the outputs will contain the same thing that the Web UI expresses: a file path for each output.
    # Let's find the node ID of the first SaveImage node. This will work when you change your workflow JSON from
    # the example above.
    save_image_node_id = next(key for key in prompt if prompt[key].class_type == "SaveImage")
    # Now let's print the absolute path to the image.
    print(outputs[save_image_node_id]["images"][0]["abs_path"])
# At this point, all the models have been unloaded from VRAM, and everything has been cleaned up.
```

See [script_examples/basic_api_example.py](examples/script_examples/basic_api_example.py) for a complete example.

Preview can be retrieved too:

```python
prompt_dict = copy.deepcopy(_PROMPT_FROM_WEB_UI)
prompt_dict["6"]["inputs"]["text"] = "masterpiece best quality man"
...
preview_frames = []
async with Comfy() as client:
    task = client.queue_with_progress(prompt)
    async for notification in task.progress():
        if notification.event == BinaryEventTypes.PREVIEW_IMAGE_WITH_METADATA:
            image_data: PreviewImageWithMetadataMessage = notification.data
            unencoded_preview, _ = image_data
            
            preview_frames.append(unencoded_preview.pil_image.copy())
```

See more in Colab: https://colab.research.google.com/drive/1Gd9F8iYRJW-LG8JLiwGTKLAcXLJ5eH78#scrollTo=mP_72JH6v1BK


### Remote

Visit the ComfyUI interface, design your workflow and click **Save (API Format)**. This JSON is what you will use as your workflow.

You can use the built-in Python client library by installing this package without its dependencies.

```shell
pip install aiohttp
pip install --no-deps git+https://github.com/hiddenswitch/ComfyUI.git
```

Then the following idiomatic pattern is available:

```python
from comfy.client.aio_client import AsyncRemoteComfyClient

client = AsyncRemoteComfyClient(server_address="http://localhost:8188")
# Now let's get the bytes of the PNG image saved by the SaveImage node:
png_image_bytes = await client.queue_prompt(prompt)
# You can save these bytes wherever you need!
with open("image.png", "rb") as f:
    f.write(png_image_bytes)
```

See [script_examples/remote_api_example.py](examples/script_examples/remote_api_example.py) for a complete example.

##### REST API

First, install this package using the [Installation Instructions](#installing). Then, run `comfyui`.

Visit the ComfyUI interface, design your workflow and click **Save (API Format)**. This JSON is what you will use as your workflow.

Then, send a request to `api/v1/prompts`. Here are some examples:

**`curl`**:

```shell
curl -X POST "http://localhost:8188/api/v1/prompts" \
     -H "Content-Type: application/json" \
     -H "Accept: image/png" \
     -o output.png \
     -d '{
       "prompt": {
         # ... (include the rest of the workflow)
       }
     }'
```

**Python**:

```python
import requests

url = "http://localhost:8188/api/v1/prompts"
headers = {
    "Content-Type": "application/json",
    "Accept": "image/png"
}
workflow = {
    "4": {
        "inputs": {
            "ckpt_name": "sd_xl_base_1.0.safetensors"
        },
        "class_type": "CheckpointLoaderSimple"
    },
    # ... (include the rest of the workflow)
}

payload = {"prompt": workflow}

response = requests.post(url, json=payload, headers=headers)
```

**Javascript (Browser)**:

```javascript
async function generateImage() {
    const prompt = "a man walking on the beach";
    const workflow = {
        "4": {
            "inputs": {
                "ckpt_name": "sd_xl_base_1.0.safetensors"
            },
            "class_type": "CheckpointLoaderSimple"
        },
        // ... (include the rest of the workflow)
    };

    const response = await fetch('http://localhost:8188/api/v1/prompts', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Accept': 'image/png'
        },
        body: JSON.stringify({prompt: workflow})
    });

    const blob = await response.blob();
    const imageUrl = URL.createObjectURL(blob);
    const img = document.createElement('img');
    // load image into the DOM
    img.src = imageUrl;
    document.body.appendChild(img);
}

generateImage().catch(console.error);
```

You can use the OpenAPI specification file to learn more about all the supported API methods.

### OpenAPI Spec for Vanilla API, Typed Clients

Use a typed, generated API client for your programming language and access ComfyUI server remotely as an API.

You can generate the client from [comfy/api/openapi.yaml](../comfy/api/openapi.yaml).

### RabbitMQ / AMQP Support

Submit jobs directly to a distributed work queue. This package supports AMQP message queues like RabbitMQ. You can submit workflows to the queue, including from the web using RabbitMQ's STOMP support, and receive realtime progress updates from multiple workers. Continue to the next section for more details.
