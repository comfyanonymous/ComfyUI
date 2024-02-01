// Factory function; converts a context to a hook with type checking

import { Context, useContext } from 'react';

export const createUseContextHook = <T,>(context: Context<T>, error: string) => {
    return () => {
        const contextValue = useContext(context);

        if (!contextValue) {
            throw new Error(error);
        }

        return contextValue;
    };
};
