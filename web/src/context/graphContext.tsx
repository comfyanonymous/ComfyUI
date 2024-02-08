import React, { ReactNode, useState } from 'react';
import { createUseContextHook } from './hookCreator';
import { ComfyGraph } from '../litegraph/comfyGraph';
import { ComfyCanvas } from '../litegraph/comfyCanvas';

interface GraphContextType {
    graph: ComfyGraph;
    canvas: ComfyCanvas;
    ctx: CanvasRenderingContext2D | null;
    initGraph: (mainCanvas: HTMLCanvasElement) => void;
    resizeCanvas: (canvasEl: HTMLCanvasElement) => void;
}

const GraphContext = React.createContext<GraphContextType | null>(null);

// Initially we create a graph and canvas, but they're not attached to any canvas
// element on the page. You must use initGraph to attach the ComfyCanvas to a
// canvas-element.
export const GraphContextProvider = ({ children }: { children: ReactNode }) => {
    const [graph, setGraph] = useState<ComfyGraph>(new ComfyGraph());
    const [canvas, setCanvas] = useState<ComfyCanvas>(new ComfyCanvas(undefined, graph));
    const [ctx, setCtx] = useState<CanvasRenderingContext2D | null>(null);

    const initGraph = (mainCanvas: HTMLCanvasElement) => {
        const canvasEl = Object.assign(mainCanvas, { id: 'graph-canvas' });
        canvasEl.tabIndex = 1;
        canvasEl.style.touchAction = 'none';

        canvas.setCanvas(canvasEl);
        const ctx = canvasEl.getContext('2d');

        graph.start();

        // Set the state with the new graph, canvas, and context
        setCtx(ctx);
    };

    const resizeCanvas = (canvasEl?: HTMLCanvasElement) => {
        if (!canvasEl) return;
        // Limit minimal scale to 1, see https://github.com/comfyanonymous/ComfyUI/pull/845
        const scale = Math.max(window.devicePixelRatio, 1);
        const { width, height } = canvasEl.getBoundingClientRect();
        canvasEl.width = Math.round(width * scale);
        canvasEl.height = Math.round(height * scale);
        if (ctx) {
            ctx.scale(scale, scale);
        }

        canvas.draw(true, true);
    };

    return (
        <GraphContext.Provider value={{ graph, canvas, ctx, initGraph, resizeCanvas }}>
            {children}
        </GraphContext.Provider>
    );
};

export const useGraph = createUseContextHook(GraphContext, 'GraphContext not found');
