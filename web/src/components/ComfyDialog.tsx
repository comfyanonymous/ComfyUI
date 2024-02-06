import {ReactNode} from "react";

interface ComfyDialogProps {
    isOpen: boolean,
    content: ReactNode,
    setIsOpen: (isOpen: boolean) => void,
}

export function ComfyDialog({isOpen, setIsOpen, content}: ComfyDialogProps) {
    return (
        <div className="comfy-modal" style={{display: isOpen ? "flex" : "none"}}>
            <div className="comfy-modal-content">
                {content}
                <button type="button" onClick={() => setIsOpen(false)}>Close</button>
            </div>
        </div>
    );
}
