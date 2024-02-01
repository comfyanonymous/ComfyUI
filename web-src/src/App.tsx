import './App.css';
import { useEffect, useRef } from 'react';
import { app } from './scripts/app';
import { api } from './scripts/api';
import { mountLiteGraph } from './scripts/main';
import { GraphContextProvider } from './context/graphContext';

function App() {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        if (canvasRef.current) {
            mountLiteGraph(canvasRef.current);
        }
    }, []);

    return (
        <div className="App">
            <GraphContextProvider>
                <canvas ref={canvasRef} style={{ width: '100%', height: '100%' }} />
                {/* Other UI componets will go here */}
            </GraphContextProvider>
        </div>
    );
}

export default App;
