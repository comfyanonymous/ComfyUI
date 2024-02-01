import React, {ReactNode, useState} from 'react';
import {createUseContextHook} from './hookCreator';
import {ComfyGraph} from '../litegraph/comfyGraph';
import {ComfyCanvas} from '../litegraph/comfyCanvas';
import {useLoadGraphData} from "../hooks/useLoadGraphData.tsx";

interface GraphContextType {
    graphState: {
        graph: ComfyGraph;
        canvas: ComfyCanvas;
        ctx: CanvasRenderingContext2D | null;
    } | null;
    loadWorkflow: () => Promise<boolean>;
    mountLiteGraph: (mainCanvas: HTMLCanvasElement) => Promise<void>;
}

const GraphContext = React.createContext<GraphContextType | null>(null);
export const useGraph = createUseContextHook(GraphContext, 'GraphContext not found');

export const GraphContextProvider = ({ children }: { children: ReactNode }) => {
    const {loadGraphData} = useLoadGraphData()
    const [graphState, setGraphState] = useState<{
        graph: ComfyGraph;
        canvas: ComfyCanvas;
        ctx: CanvasRenderingContext2D | null;
    } | null>(null);

    const loadWorkflow = async (): Promise<boolean> => {
        try {
            const json = localStorage.getItem('workflow');
            if (json) {
                const workflow = JSON.parse(json);
                await loadGraphData(workflow);
                return true;
            }
        } catch (err) {
            console.error('Error loading previous workflow', err);
        }

        return false;
    };

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

    return (
        <GraphContext.Provider
            value={{
                graphState,
                mountLiteGraph,
                loadWorkflow,
            }}
        >
            {children}
        </GraphContext.Provider>
    );
};

