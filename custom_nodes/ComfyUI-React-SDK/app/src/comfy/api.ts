
export const WS_MESSAGE_TYPE_EXECUTING="executing"
export const WS_MESSAGE_TYPE_EXECUTED="executed"
export const WS_MESSAGE_TYPE_STATUS="status"
export const WS_MESSAGE_TYPE_PROGRESS="progress"
export const WS_MESSAGE_TYPE_EXECUTION_START="execution_start"
export const WS_MESSAGE_TYPE_EXECUTION_CACHED="execution_cached"

interface Callbacks {
    [key: string]: (message: any) => void;
}
const subscribers: Callbacks = {}

let webseocket: WebSocket

export function Subscribe(key: string, callback:(message: any) => void){
    subscribers[key] = callback
}

export function UnSubscribe(key: string){
    delete subscribers[key];
}
export function GetWebSocket(){
    if (webseocket) {
        return webseocket
    }
    let { hostname, port } = window.location;

    if(process.env.NODE_ENV === "development"){ // temp fix until more normal way to proxy web socket.
        hostname = "localhost"
        port = "8188"
    }

    webseocket = new WebSocket("ws://"+hostname+":"+port+"/ws?clientId=1122");
    // Define event handlers for the WebSocket connection
    webseocket.onopen = () => {
        console.log('WebSocket connected');
    };

    webseocket.onmessage = (event) =>{
        Object.entries(subscribers).forEach(([key, callback]) => {
            callback(event); // Call the function
        });
    }

    webseocket.onclose = () => {
        console.log('WebSocket disconnected');
    };
    return webseocket
}

export interface DashboardGenParams {
    cfg: number
    steps: number
    seed: number
    checkpoint: string
    height: number
    width: number
    positivePrompt: string
    negativePrompt: string
}

export interface Root {
    CheckpointLoaderSimple: CheckpointLoaderSimple
}

export interface Input {
    required: Required
}

export interface Required {
    ckpt_name: string[][]
}

export interface CheckpointLoaderSimple {
    input: Input
    output: string[]
    output_is_list: boolean[]
    output_name: string[]
    name: string
    display_name: string
    description: string
    category: string
    output_node: boolean
}

// #This is the ComfyUI api prompt format.
//
//     #If you want it for a specific workflow you can "enable dev mode options"
// #in the settings of the UI (gear beside the "Queue Size: ") this will enable
// #a button on the UI to save workflows in api format.
//
//     #keep in mind ComfyUI is pre alpha software so this format will change a bit.
//
//     #this is the one for the default workflow
export const WORKFLOW =
    {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "cfg": 3,
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
                "seed": Math.round(Math.random()*100000),
                "steps": 5
            }
        },
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": "DreamShaperXL_Turbo_v2_1.safetensors"
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
                "text": "masterpiece best quality fish"
            }
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": [
                    "4",
                    1
                ],
                "text": "bad hands"
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
    };




