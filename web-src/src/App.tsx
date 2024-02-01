import './App.css';
import {useEffect, useRef} from 'react';
import {GraphContextProvider, useGraph} from './context/graphContext';
import {ComfyAppContextProvider} from "./context/appContext.tsx";

function App() {
    return (
        <div className="App">
            <ComfyAppContextProvider>
                <GraphContextProvider>
                    <MainCanvas/>
                    {/* Other UI componets will go here */}
                </GraphContextProvider>
            </ComfyAppContextProvider>
        </div>
    );
}

function MainCanvas() {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const {mountLiteGraph} = useGraph()

    useEffect(() => {
        if (canvasRef.current) {
            mountLiteGraph(canvasRef.current);
        }
    });

    return (
        <canvas ref={canvasRef} style={{width: '100%', height: '100%'}}/>
    );
}
export default App;
