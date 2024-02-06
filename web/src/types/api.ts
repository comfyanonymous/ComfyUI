import { ComfyError } from './many';
import { WorkflowStep } from '../../autogen_web_ts/comfy_request.v1';
import { ComfyObjectInfo } from './comfy';

export type EmbeddingsResponse = string[];

export interface UploadFileResponse {
    name: string;
    type: string;
    subfolder: string;
}

export interface ViewMetadataResponse {
    [key: string]: any;
}

export interface GetPromptResponse {
    queue_remaining?: number;
    exec_info: {
        [key: string]: any;
    };
}

export interface ObjectInfoResponse {
    [key: string]: NodeInfo;
}

export interface HistoryResponse {
    [key: string]: {
        prompt: {
            [key: string]: any;
        };
        outputs: {
            [key: string]: any;
        };
        status: string;
    };
}

export type QueueDataTypes = number | string | object;

export type QueueData = QueueDataTypes | Array<QueueDataTypes>;

export interface QueueResponse {
    queue_running: QueueData[];
    queue_pending: QueueData[];
}

export interface QueuePromptResponse {
    prompt_id: string;
    number: number;
    node_errors: Record<string, ComfyError>;
}

export interface SystemStatsResponse {
    system: {
        os: string;
        python_version: string;
        embedded_python: boolean;
    };
    devices: SystemDeviceStat[];
}

export interface UserConfigResponse {
    storage: 'server';
    migrated: boolean;
    users: {
        [key: string]: any;
    }[];
}

export interface SettingsResponse {
    [key: string]: any;
}

interface SystemDeviceStat {
    name: string;
    type: string;
    index: number;
    vram_total: number;
    vram_free: number;
    torch_vram_total: number;
    torch_vram_free: number;
}

interface NodeInfo {
    input: Record<string, any>;
    output: string[];
    output_is_list: boolean[];
    output_name: string[];
    name: string;
    display_name: string;
    description: string;
    category: string;
    output_node: boolean;
}

export interface ComfyQueueItems {
    Running: {
        prompt: QueueData;
        remove: {
            name: string;
            cb: () => Promise<void>;
        };
    }[];
    Pending: {
        prompt: QueueData;
    }[];
}

export interface ComfyHistoryItems {
    History: {
        prompt: {};
        outputs: {};
        status: string;
    }[];
}

export type ComfyItems = ComfyQueueItems | ComfyHistoryItems;

export interface IComfyApi {
    socket: WebSocket | null;
    api_host: string;
    api_base: string;
    user: string | undefined;
    clientId: string | undefined;

    apiURL(route: string): string;
    fetchApi(route: string, options?: RequestInit): Promise<Response>;
    addEventListener(
        type: string,
        callback: EventListenerOrEventListenerObject,
        options?: boolean | AddEventListenerOptions
    ): void;
    init(): void;
    getExtensions(): Promise<string[]>;
    getEmbeddings(): Promise<EmbeddingsResponse>;
    getNodeDefs(): Promise<Record<string, ComfyObjectInfo>>;
    queuePrompt(
        number: number,
        prompt: { output: Record<string, WorkflowStep>; workflow: any }
    ): Promise<QueuePromptResponse>;
    getItems(type: string): Promise<any>; // Replace 'any' with the actual return type if known
    getQueue(): Promise<ComfyQueueItems>;
    getHistory(max_items?: number): Promise<ComfyHistoryItems>;
    getSystemStats(): Promise<SystemStatsResponse>;
    deleteItem(type: string, id: number): Promise<void>;
    clearItems(type: string): Promise<void>;
    interrupt(): Promise<void>;
    getUserConfig(): Promise<UserConfigResponse>;
    createUser(username: string): Promise<Response>;
    getSettings(): Promise<SettingsResponse>;
    getSetting(id: string): Promise<unknown>; // Replace 'unknown' with the actual return type if known
    storeSettings(settings: Record<string, unknown>): Promise<Response>;
    storeSetting(id: string, value: Record<string, any>): Promise<Response>;
    getUserData(file: string, options?: RequestInit): Promise<unknown>; // Replace 'unknown' with the actual return type if known
    storeUserData(
        file: string,
        data: BodyInit,
        options?: { stringify?: boolean; throwOnError?: boolean }
    ): Promise<void>;
}
