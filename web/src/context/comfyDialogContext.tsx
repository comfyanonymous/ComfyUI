import React, {ReactNode, useState} from 'react';
import {createUseContextHook} from './hookCreator';
import {ComfyDialog} from "../components/ComfyDialog.tsx";

interface ComfyDialogContextType {
    showDialog: (content: ReactNode) => void;
}

const ComfyDialogContext = React.createContext<ComfyDialogContextType | null>(null);
export const useComfyDialog = createUseContextHook(ComfyDialogContext, 'useComfyDialog must be used within a ComfyDialogContextProvider');


export const ComfyDialogContextProvider = ({children}: { children: ReactNode }) => {
    const [isOpen, setIsOpen] = useState(false);
    const [dialogContent, setDialogContent] = useState<ReactNode | null>(null);

    const show = (content: ReactNode) => {
        setIsOpen(true)
        setDialogContent(() => content)
    }

    return (
        <ComfyDialogContext.Provider value={{showDialog: show}}>
            {children}

            <ComfyDialog
                isOpen={isOpen}
                setIsOpen={setIsOpen}
                content={dialogContent}
            />
        </ComfyDialogContext.Provider>
    );
}
