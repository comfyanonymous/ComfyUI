import React, { useState } from 'react';
import { useEffect, useRef } from 'react';
import { GraphContextProvider, useGraph } from './context/graphContext';
import { ComfyAppContextProvider, useComfyApp } from './context/appContext.tsx';
import { ComfyDialogContextProvider } from './context/comfyDialogContext.tsx';
import { useLoadGraphData } from './hooks/useLoadGraphData.tsx';
import { SettingsContextProvider, useSettings } from './context/settingsContext.tsx';
import { registerNodes } from './litegraph/registerNodes.ts';
import { PluginProvider } from './context/pluginContext';
import { ApiContextProvider } from './context/apiContext.tsx';
import { LiteGraph } from 'litegraph.js';

function RenderComponents() {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const { graphState, initGraph } = useGraph();
    const { app } = useComfyApp();
    const { load: loadSettings } = useSettings();
    const { loadGraphData, loadWorkflow } = useLoadGraphData();

    // TODO: some stuff here don't actually belong here... we'd remove them later
    // they're here currently for easier debugging and development
    const [isPanning, setIsPanning] = useState(false);
    const [lastPos, setLastPos] = useState({ x: 0, y: 0 });

    useEffect(() => {
        if (canvasRef.current) {
            // Set canvas size to window size
            canvasRef.current.width = window.innerWidth;
            canvasRef.current.height = window.innerHeight;

            // Initialize graph
            initGraph(canvasRef.current);

            // Add event listener for zooming
            canvasRef.current.addEventListener('wheel', event => {
                const scale = event.deltaY < 0 ? 1.1 : 0.9; // Zoom in if scroll up, else zoom out
                const center = [event.clientX, event.clientY]; // Center of zoom is mouse position
                graphState?.graph?.zoom(scale, center);
            });

            // Add event listeners for panning
            canvasRef.current.addEventListener('mousedown', event => {
                setIsPanning(true);
                setLastPos({ x: event.clientX, y: event.clientY });
            });

            canvasRef.current.addEventListener('mousemove', event => {
                if (isPanning && graphState?.graph) {
                    const dx = event.clientX - lastPos.x;
                    const dy = event.clientY - lastPos.y;
                    graphState.graph.offset[0] += dx;
                    graphState.graph.offset[1] += dy;
                    setLastPos({ x: event.clientX, y: event.clientY });
                    graphState.graph.setDirtyCanvas(true, true);
                }
            });

            canvasRef.current.addEventListener('mouseup', () => {
                setIsPanning(false);
            });
        }

        const loadAppData = async () => {
            await registerNodes();
            loadSettings();

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
            <canvas id="graph-canvas" tabIndex={1} ref={canvasRef} />
            {/* Other UI componets will go here */}
        </>
    );
}

function App() {
    return (
        <div className="App">
            <PluginProvider>
                <ApiContextProvider>
                    <ComfyAppContextProvider>
                        <ComfyDialogContextProvider>
                            <GraphContextProvider>
                                <SettingsContextProvider>
                                    <RenderComponents />
                                </SettingsContextProvider>
                            </GraphContextProvider>
                        </ComfyDialogContextProvider>
                    </ComfyAppContextProvider>
                </ApiContextProvider>
            </PluginProvider>
        </div>
    );
}

export default App;
