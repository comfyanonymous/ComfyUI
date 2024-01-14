export type QueueItem = {
    number: number;
    batchCount: number;
};

export interface ComfyButtonWidget {
    name: string;
    type: 'button';
    value: HTMLButtonElement;
    callback: (value: any) => void;
}

export interface ComfyImageWidget {
    name: string;
    type: 'image';
    last_y: number;
    computedHeight: number;
    value: HTMLImageElement;
    computeSize: () => number[];
    callback: (value: any) => void;
}

export interface ComfyFileWidget {
    name: string;
    type: 'file';
    value: ComfyFile;
    callback: (value: any) => void;
}

export interface ComfyTextWidget {
    name: string;
    type: 'text';
    value: string;
    callback: (value: any) => void;
}

export type ComfyWidget = ComfyImageWidget | ComfyButtonWidget | ComfyFileWidget | ComfyTextWidget;

export interface ComfyFile {
    type: string;
    filename: string;
    subfolder: string;
}

export type ComfyImages = HTMLImageElement[] | ComfyFile[];

export interface SerializedNodeObject {
    imgs?: ComfyImages;
    images?: ComfyImages;
    selectedIndex: number;
    img_paste_mode: string;
    original_imgs?: ComfyImages;
    widgets?: ComfyWidget[] | null;
}

export class ComfyNode {
    id: string;
    imgs: ComfyImages;
    imageIndex?: number;
    imageOffset?: number;
    images?: ComfyImages;
    widgets: ComfyWidget[];
    computeSize: () => number[];
}

export type ClassMethod<T> = {
    [K in keyof T]: T[K] extends (...args: any[]) => any ? T[K] : never;
};
