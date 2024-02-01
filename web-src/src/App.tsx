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
                        <MainCanvas/>
                        {/* Other UI componets will go here */}
                    </GraphContextProvider>
                </ComfyDialogContextProvider>
            </ComfyAppContextProvider>
        </div>
    );
}

function MainCanvas() {
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
        <canvas ref={canvasRef} style={{width: '100%', height: '100%'}}/>
    );
}
export default App;
