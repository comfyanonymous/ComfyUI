import './App.css';
import { useEffect, useRef } from 'react';
import { app } from './scripts/app';
import { api } from './scripts/api';

// We should probably have a ComfyAppContextProvider that wraps the entire app?
function App() {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        if (canvasRef.current) {
            app.setup(canvasRef.current, api);
        }
    }, []);

    return (
        <div className="App">
            <canvas ref={canvasRef} style={{ width: '100%', height: '100%' }} />
            {/* Other UI componets will go here */}
        </div>
    );
}

export default App;
