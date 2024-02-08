import { ReactNode, useEffect } from 'react';

interface ComfySettingsDialogProps {
    open: boolean;
    content: ReactNode;
    closeDialog: () => void;
}

export function ComfySettingsDialog({ open, content, closeDialog }: ComfySettingsDialogProps) {
    return (
        <dialog id="comfy-settings-dialog" open={open}>
            <table className="comfy-modal-content comfy-table">
                <caption>Settings</caption>
                <tbody>{content}</tbody>
            </table>

            <button type="button" style={{ cursor: 'pointer' }} onClick={closeDialog}>
                Close
            </button>
        </dialog>
    );
}
