import { Context, useContext } from 'react';

export const createUseContextHook = (context: Context<any>, error: string) => {
    return () => {
        const contextValue = useContext(context);

        if (!contextValue) {
            throw new Error(error);
        }

        return contextValue;
    };
};
