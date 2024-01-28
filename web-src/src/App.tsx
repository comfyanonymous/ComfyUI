import './App.css'
import {MainCanvas} from "./components/MainCanvas.tsx";
import {ComfyApp} from "./scripts/app.ts";

function App({app}: { app: ComfyApp }) {
    return (
        <>
            <MainCanvas app={app}/>
        </>
    )
}

export default App
