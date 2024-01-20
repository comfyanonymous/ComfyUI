import { ComfyNode } from './comfyNode';
import { widgetTypes, IWidget } from 'litegraph.js';

export type comfyWidgetTypes = widgetTypes & ('converted-widget' | 'hidden');

export interface ComfyWidget<TValue = any, TOption = any> extends IWidget<TValue, TOption> {
    type: comfyWidgetTypes;
    computedHeight?: number;
    element: HTMLElement;
    onRemove?: () => void;
    callback?: (value: TValue) => void;
    serializeValue?: () => undefined;
    afterQueued?: () => void;
    draw?(ctx: CanvasRenderingContext2D, node: ComfyNode, width: number, posY: number, height: number): void;
}
