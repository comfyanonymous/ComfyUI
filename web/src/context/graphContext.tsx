import React, { ReactNode, useState } from 'react';
import { container } from '../inversify.config';
import { createUseContextHook } from './hookCreator';
import { ComfyGraph } from '../litegraph/comfyGraph';
import { ComfyCanvas } from '../litegraph/comfyCanvas';
import { useLoadGraphData } from '../hooks/useLoadGraphData.tsx';

type GraphState = {
    graph: ComfyGraph;
    canvas: ComfyCanvas;
    ctx: CanvasRenderingContext2D | null;
} | null;

interface GraphContextType {
    graphState: GraphState;
    initGraph: (mainCanvas: HTMLCanvasElement) => void;
    resizeCanvas: (canvasEl: HTMLCanvasElement) => void;
}

const GraphContext = React.createContext<GraphContextType | null>(null);

export const GraphContextProvider = ({ children }: { children: ReactNode }) => {
    const [graphState, setGraphState] = useState<GraphState>(null);

    const initGraph = (mainCanvas: HTMLCanvasElement) => {
        const canvasEl = Object.assign(mainCanvas, { id: 'graph-canvas' });
        canvasEl.tabIndex = 1;
        canvasEl.style.touchAction = 'none';

        const graph = new ComfyGraph();
        const canvas = new ComfyCanvas(canvasEl, graph);
        const ctx = canvasEl.getContext('2d');

        graph.start();

        // Set the state with the new graph, canvas, and context
        setGraphState({
            graph,
            canvas,
            ctx,
        });
    };

    const resizeCanvas = (canvasEl: HTMLCanvasElement) => {
        // Limit minimal scale to 1, see https://github.com/comfyanonymous/ComfyUI/pull/845
        const scale = Math.max(window.devicePixelRatio, 1);
        const { width, height } = canvasEl.getBoundingClientRect();
        canvasEl.width = Math.round(width * scale);
        canvasEl.height = Math.round(height * scale);
        if (graphState?.ctx) {
            graphState.ctx.scale(scale, scale);
        }

        if (graphState?.canvas) {
            graphState.canvas.draw(true, true);
        }
    };

    return <GraphContext.Provider value={{ graphState, initGraph, resizeCanvas }}>{children}</GraphContext.Provider>;
};

export const useGraph = createUseContextHook(GraphContext, 'GraphContext not found');
