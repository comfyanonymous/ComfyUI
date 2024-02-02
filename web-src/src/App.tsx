import './App.css';
import React from 'react';
import { useEffect, useRef } from 'react';
import { GraphContextProvider, useGraph } from './context/graphContext';
import { ComfyAppContextProvider, useComfyApp } from './context/appContext.tsx';
import { ComfyDialogContextProvider } from './context/comfyDialogContext.tsx';
import { useLoadGraphData } from './hooks/useLoadGraphData.tsx';
import { loadWorkflow } from './litegraph/graphUtils.ts';
import { PluginProvider, pluginStore } from './pluginStore';

function RenderComponents() {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const { graphState, initGraph } = useGraph();
    const { app } = useComfyApp();
    const { loadGraphData } = useLoadGraphData();

    useEffect(() => {
        if (canvasRef.current) {
            initGraph(canvasRef.current);
        }

        const loadAppData = async () => {
            const restored = await loadWorkflow();

            // We failed to restore a workflow so load the default
            if (!restored) {
                await loadGraphData();
            }

            if (graphState && graphState.graph) {
                app.enableWorkflowAutoSave(graphState.graph);
            }
        };

        if (canvasRef.current) {
            loadAppData();
        }
    }, [canvasRef.current]);

    return (
        <>
            <canvas ref={canvasRef} style={{ width: '100%', height: '100%' }} />
            {/* Other UI componets will go here */}
        </>
    );
}

function App() {
    return (
        <div className="App">
            <PluginProvider pluginStore={pluginStore}>
                <ComfyAppContextProvider>
                    <ComfyDialogContextProvider>
                        <GraphContextProvider>
                            <RenderComponents />
                        </GraphContextProvider>
                    </ComfyDialogContextProvider>
                </ComfyAppContextProvider>
            </PluginProvider>
        </div>
    );
}

export default App;
