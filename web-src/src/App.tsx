import './App.css';
import {useEffect, useRef} from 'react';
import {GraphContextProvider, useGraph} from './context/graphContext';
import {ComfyAppContextProvider} from "./context/appContext.tsx";
import {ComfyDialogContextProvider} from "./context/comfyDialogContext.tsx";
import {useLoadGraphData} from "./hooks/useLoadGraphData.tsx";

function App() {
    return (
        <div className="App">
            <ComfyAppContextProvider>
                <ComfyDialogContextProvider>
                    <GraphContextProvider>
                        <InnerApp/>
                    </GraphContextProvider>
                </ComfyDialogContextProvider>
            </ComfyAppContextProvider>
        </div>
    );
}

function InnerApp() {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const {mountLiteGraph, loadWorkflow} = useGraph()
    const {loadGraphData} = useLoadGraphData()

    useEffect(() => {
        const loadData = async () => {
            const restored = await loadWorkflow();

            // We failed to restore a workflow so load the default
            if (!restored) {
                await loadGraphData();
            }
        }

        if (canvasRef.current) {
            mountLiteGraph(canvasRef.current);
            loadData();
        }
    });

    return (
        <>
        <canvas ref={canvasRef} style={{width: '100%', height: '100%'}}/>
            {/* Other UI componets will go here */}
        </>
    );
}
export default App;
