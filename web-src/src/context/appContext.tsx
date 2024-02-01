import React, {ReactNode, useEffect, useState} from 'react';
import {createUseContextHook} from './hookCreator';
import {IComfyApp} from "../types/interfaces.ts";
import {ComfyApp} from "../scripts/app2.ts";

interface ComfyAppContextType {
    app: IComfyApp | null
}

const ComfyAppContext = React.createContext<ComfyAppContextType | null>(null);
export const useComfyApp = createUseContextHook(ComfyAppContext, 'useComfyApp must be used within a ComfyAppContextProvider');

export const ComfyAppContextProvider = ({children}: { children: ReactNode }) => {
    const [app, setApp] = useState<IComfyApp | null>(null);

    // Though the comfier app is a singleton,
    // I think it makes sense to have it in the context for usage in the web app
    useEffect(() => {
        setApp(ComfyApp.getInstance())
    })

    return (
        <ComfyAppContext.Provider value={{app}}>
            {children}
        </ComfyAppContext.Provider>
    );
};
