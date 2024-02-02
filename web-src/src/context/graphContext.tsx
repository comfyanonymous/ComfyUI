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
}

const GraphContext = React.createContext<GraphContextType | null>(null);

export const GraphContextProvider = ({ children }: { children: ReactNode }) => {
    const { loadGraphData } = useLoadGraphData();
    const [graphState, setGraphState] = useState<GraphState>(null);

    const initGraph = (mainCanvas: HTMLCanvasElement) => {
        const canvasEl = Object.assign(mainCanvas, { id: 'graph-canvas' });
        canvasEl.tabIndex = 1;

        const graph = container.resolve(ComfyGraph);
        const canvas = new ComfyCanvas(canvasEl, graph);
        const ctx = canvasEl.getContext('2d');

        // Set the state with the new graph, canvas, and context
        setGraphState({ graph, canvas, ctx });
    };

    return <GraphContext.Provider value={{ graphState, initGraph }}>{children}</GraphContext.Provider>;
};

export const useGraph = createUseContextHook(GraphContext, 'GraphContext not found');
