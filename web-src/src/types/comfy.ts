import { ComfyNode } from '../scripts/comfyNode.ts';

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

interface ComfyOptionsHost {
    el: Element;
    updateImages: (imgs: (HTMLImageElement | string)[]) => void;
    getHeight: () => void;
    onDraw: () => void;
}

export interface AddDOMWidgetOptions {
    host?: ComfyOptionsHost;
    getHeight?: ComfyOptionsHost['getHeight'];
    onDraw?: ComfyOptionsHost['onDraw'];
    hideOnZoom?: boolean;
    selectOn?: string[];
    getValue?: () => string | undefined;
    setValue?: (value: any) => string | undefined | void;
    beforeResize?: (node: ComfyNode) => void;
    afterResize?: (node: ComfyNode) => void;
}

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
