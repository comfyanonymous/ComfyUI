import { ANIM_PREVIEW_WIDGET, ComfyApp, app } from './app.js';
import { LGraphCanvas, LGraphNode, LiteGraph } from 'litegraph.js';
import { ComfyNode } from './comfyNode';

const SIZE = Symbol();

interface Point {
    x: number;
    y: number;
    width: number;
    height: number;
}

function intersect(a: Point, b: Point) {
    const x = Math.max(a.x, b.x);
    const num1 = Math.min(a.x + a.width, b.x + b.width);
    const y = Math.max(a.y, b.y);
    const num2 = Math.min(a.y + a.height, b.y + b.height);
    if (num1 >= x && num2 >= y) return [x, y, num1 - x, num2 - y];
    else return null;
}

export function getClipPath(node: ComfyNode, element: Element, elRect: Point) {
    if (app.canvas) {
        const selectedNode = Object.values(app.canvas.selected_nodes)[0];
        if (selectedNode && selectedNode !== node) {
            const MARGIN = 7;
            const scale = app.canvas?.ds.scale;

            const bounding = selectedNode.getBounding();
            const intersection = intersect(
                {
                    x: elRect.x / scale,
                    y: elRect.y / scale,
                    width: elRect.width / scale,
                    height: elRect.height / scale,
                },
                {
                    x: selectedNode.pos[0] + app.canvas.ds.offset[0] - MARGIN,
                    y: selectedNode.pos[1] + app.canvas.ds.offset[1] - LiteGraph.NODE_TITLE_HEIGHT - MARGIN,
                    width: bounding[2] + MARGIN + MARGIN,
                    height: bounding[3] + MARGIN + MARGIN,
                }
            );

            if (!intersection) {
                return '';
            }

            const widgetRect = element.getBoundingClientRect();
            const clipX = intersection[0] - widgetRect.x / scale + 'px';
            const clipY = intersection[1] - widgetRect.y / scale + 'px';
            const clipWidth = intersection[2] + 'px';
            const clipHeight = intersection[3] + 'px';
            const path = `polygon(0% 0%, 0% 100%, ${clipX} 100%, ${clipX} ${clipY}, calc(${clipX} + ${clipWidth}) ${clipY}, calc(${clipX} + ${clipWidth}) calc(${clipY} + ${clipHeight}), ${clipX} calc(${clipY} + ${clipHeight}), ${clipX} 100%, 100% 100%, 100% 0%)`;
            return path;
        }
    }
    return '';
}

export function addDomClippingSetting(app: ComfyApp) {
    app.ui.settings.addSetting({
        id: 'Comfy.DOMClippingEnabled',
        name: 'Enable DOM element clipping (enabling may reduce performance)',
        type: 'boolean',
        defaultValue: true,
        onChange(value: boolean) {},
    });
}
