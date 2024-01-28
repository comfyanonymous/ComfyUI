import './App.css'
import {Canvas} from "./components/Canvas.tsx";
import {ComfyApp} from "./scripts/app.ts";

function App() {
    const app = ComfyApp.getInstance();


    return (
        <>
            <Canvas app={app}/>
        </>
    )
}

export default App
