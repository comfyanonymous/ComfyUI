export type ComfyObjectInfo = {
    name: string;
    display_name?: string;
    description?: string;
    category: string;
    input?: {
        required?: Record<string, ComfyObjectInfoConfig>;
        optional?: Record<string, ComfyObjectInfoConfig>;
    };
    output?: string[];
    output_name: string[];
};

export type ComfyObjectInfoConfig = [string | any[]] | [string | any[], any];

export interface ComfyNodeConfig {
    minWidth: number;
    minHeight: number;
    widget?: {
        options?: {
            forceInput?: boolean;
            defaultInput?: string;
        };
    };
}

export interface ComfyPromptStatus {
    queue_remaining?: number;
    exec_info: {
        [key: string]: any;
    };
}
