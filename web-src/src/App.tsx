import React from 'react';
import { useEffect, useRef } from 'react';
import { GraphContextProvider, useGraph } from './context/graphContext';
import { ComfyAppContextProvider, useComfyApp } from './context/appContext.tsx';
import { ComfyDialogContextProvider } from './context/comfyDialogContext.tsx';
import { useLoadGraphData } from './hooks/useLoadGraphData.tsx';
import { SettingsContextProvider, useSettings } from './context/settingsContext.tsx';
import { registerNodes } from './litegraph/registerNodes.ts';
import { PluginProvider } from './context/pluginContext';
import { ApiContextProvider } from './context/apiContext.tsx';

function RenderComponents() {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const { graphState, initGraph } = useGraph();
    const { app } = useComfyApp();
    const { load: loadSettings } = useSettings();
    const { loadGraphData, loadWorkflow } = useLoadGraphData();

    useEffect(() => {
        if (canvasRef.current) {
            initGraph(canvasRef.current);
        }

        const loadAppData = async () => {
            await registerNodes();
            loadSettings();

            const restored = await loadWorkflow();

            // We failed to restore a workflow so load the default
            if (!restored) {
                console.log('sss');
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
