import React, { useEffect, useRef } from 'react';
import { GraphContextProvider, useGraph } from './context/graphContext';
import { ComfyAppContextProvider, useComfyApp } from './context/appContext.tsx';
import { ComfyDialogContextProvider } from './context/comfyDialogContext.tsx';
import { useLoadGraphData } from './hooks/useLoadGraphData.tsx';
import { SettingsContextProvider, useSettings } from './context/settingsContext.tsx';
import { registerNodes } from './litegraph/registerNodes.ts';
import { PluginProvider } from './pluginSystem/pluginContext.tsx';
import { ApiContextProvider } from './context/apiContext.tsx';
import { ComfyUIContextProvider } from './context/uiContext.tsx';
import { JobQueueContextProvider } from './context/jobQueueContext.tsx';

function RenderComponents() {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const { enableWorkflowAutoSave } = useComfyApp();
    const { graph, initGraph, resizeCanvas } = useGraph();
    const { load: loadSettings } = useSettings();
    const { loadGraphData, loadWorkflow } = useLoadGraphData();

    useEffect(() => {
        if (canvasRef.current) {
            // Initialize graph
            initGraph(canvasRef.current);

            resizeCanvas(canvasRef.current);
            window.addEventListener('resize', () => resizeCanvas(canvasRef.current!));
        }

        const loadAppData = async () => {
            await registerNodes();
            loadSettings();

            const restored = await loadWorkflow();

            // We failed to restore a workflow so load the default
            if (!restored) {
                await loadGraphData();
            }

            enableWorkflowAutoSave(graph);
        };

        if (canvasRef.current) {
            loadAppData().catch((err: unknown) => {
                console.error(err);
            });
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
        <PluginProvider>
            <ApiContextProvider>
                <ComfyAppContextProvider>
                    <ComfyDialogContextProvider>
                        <GraphContextProvider>
                            <JobQueueContextProvider>
                                <SettingsContextProvider>
                                    <ComfyUIContextProvider>
                                        <RenderComponents />
                                    </ComfyUIContextProvider>
                                </SettingsContextProvider>
                            </JobQueueContextProvider>
                        </GraphContextProvider>
                    </ComfyDialogContextProvider>
                </ComfyAppContextProvider>
            </ApiContextProvider>
        </PluginProvider>
    );
}

export default App;
