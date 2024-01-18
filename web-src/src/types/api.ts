import {ComfyError} from "./many";

export type ComfyExtensionsResponse = string[];

export type ViewFileResponse = ArrayBuffer | string;

export interface UploadFileResponse {
    name: string,
    type: string,
    subfolder: string,
}

export interface ViewMetadataResponse {
    [key: string]: any
}

export interface GetPromptResponse {
    queue_remaining: number,
    exec_info: {
        [key: string]: any
    },
}

export interface ObjectInfoResponse {
    [key: string]: NodeInfo
}

export interface HistoryResponse {
    [key: string]: {
        "prompt": {
            [key: string]: any
        }
        "outputs": {
            [key: string]: any
        },
        status: string,
    }
}

export type QueueDataTypes = number | string | object;
export type QueueData = QueueDataTypes | Array<QueueDataTypes>;


export interface QueueResponse {
    queue_running: QueueData[];
    queue_pending: QueueData[];
}

export interface QueuePromptResponse {
    prompt_id: string,
    number: number,
    "node_errors": Record<string, ComfyError>
}

export interface SystemStatsResponse {
    system: {
        os: string,
        python_version: string,
        embedded_python: boolean
    },
    devices: SystemDeviceStat[],
}

interface SystemDeviceStat {
    name: string,
    type: string,
    index: number,
    vram_total: number,
    vram_free: number,
    torch_vram_total: number,
    torch_vram_free: number,
}

interface NodeInfo {
    input: Record<string, any>
    output: string[]
    output_is_list: boolean[]
    output_name: string[]
    name: string
    display_name: string
    description: string
    category: string
    output_node: boolean
}