import {useEffect, useRef} from "react";
import {ComfyApp} from "../scripts/app.ts";

export function Canvas({app}: { app: ComfyApp }) {
    const canvasRef = useRef<HTMLCanvasElement | null>(null);

    useEffect(() => {
        if (canvasRef.current) {
            canvasRef.current = app.canvasEl;
        }
    }, []);

    return (
        <canvas ref={canvasRef} style={{width: "100%", height: "100%"}}/>
    )
}
