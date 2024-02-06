import { widgetTypes, IWidget } from 'litegraph.js';
import { IComfyNode } from './interfaces';

export type comfyWidgetTypes = widgetTypes & ('string' | 'converted-widget' | 'hidden');

// TO DO: I think this should be a class, rather than an interface?
// Also what are the generics for?
export interface ComfyWidget<TValue = any, TOption = any> extends IWidget<TValue, TOption> {
    type: any;
    computedHeight?: number;
    label?: string;
    element: HTMLElement;
    onRemove?: () => void;
    callback?: (value: TValue) => void;
    serializeValue?: () => undefined;
    afterQueued?: () => void;
    beforeQueued?: () => void;
    serialize?: boolean;
    dynamicPrompts?: boolean;
    linkedWidgets?: ComfyWidget[];
    inputEl?: HTMLInputElement | HTMLTextAreaElement;
    draw?(ctx: CanvasRenderingContext2D, node: IComfyNode, width: number, posY: number, height: number): void;

    [key: symbol]: boolean;
}
