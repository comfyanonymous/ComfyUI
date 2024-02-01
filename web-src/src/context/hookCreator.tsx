import {Context, useContext} from 'react';

export const createUseContextHook = <T extends any>(context: Context<T>, error: string) => {
    return () => {
        const contextValue = useContext(context);

        if (!contextValue) {
            throw new Error(error);
        }

        return contextValue;
    };
};
