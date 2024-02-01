import React, { useState, useEffect, ReactNode } from 'react';
import { createUseContextHook } from './hookCreator';
import { loadWorkflow } from '../litegraph/graphUtils';
import { ComfyGraph } from '../litegraph/comfyGraph';
import { ComfyCanvas } from '../litegraph/comfyCanvas';

interface GraphContextType {
    graphState: {
        graph: ComfyGraph;
        canvas: ComfyCanvas;
        ctx: CanvasRenderingContext2D | null;
    } | null;
    mountLiteGraph: (mainCanvas: HTMLCanvasElement) => Promise<void>;
}

const GraphContext = React.createContext<GraphContextType | null>(null);
export const useGraph = createUseContextHook(GraphContext, 'GraphContext not found');

export const GraphContextProvider = ({ children }: { children: ReactNode }) => {
    const [graphState, setGraphState] = useState<{
        graph: ComfyGraph;
        canvas: ComfyCanvas;
        ctx: CanvasRenderingContext2D | null;
    } | null>(null);

    const mountLiteGraph = async (mainCanvas: HTMLCanvasElement) => {
        // Assuming mainCanvas is passed or obtained somehow
        // Initialize your graph and canvas here, similar to the original mountLiteGraph function
        const canvasEl = Object.assign(mainCanvas, { id: 'graph-canvas' });
        canvasEl.tabIndex = 1;

        const graph = new ComfyGraph();
        const canvas = new ComfyCanvas(canvasEl, graph);
        const ctx = canvasEl.getContext('2d');

        // Set the state with the new graph, canvas, and context
        setGraphState({ graph, canvas, ctx });
    };

    return <GraphContext.Provider value={(graphState, mountLiteGraph)}>{children}</GraphContext.Provider>;
};
