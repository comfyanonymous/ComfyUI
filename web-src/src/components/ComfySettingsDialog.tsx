import { ReactNode } from 'react';

interface ComfySettingsDialogProps {
    open: boolean;
    content: ReactNode;
}

export function ComfySettingsDialog({ open, content }: ComfySettingsDialogProps) {
    return (
        <dialog id="comfy-settings-dialog" open={open}>
            <table className="comfy-modal-content comfy-table">
                <caption>Settings</caption>
                <tbody>{content}</tbody>

                <button type="button" style={{ cursor: 'pointer' }} onClick={() => {}}>
                    Close
                </button>
            </table>
        </dialog>
    );
}
