import { useCallback, useState } from 'react';

// Returning a new object reference guarantees that a before-and-after
//   equivalence check will always be false, resulting in a re-render, even
//   when multiple calls to forceUpdate are batched.

export default function useForceUpdate(): () => void {
    const [, dispatch] = useState<object>(Object.create(null));

    // Turn dispatch(required_parameter) into dispatch().
    const memoizedDispatch = useCallback((): void => {
        dispatch(Object.create(null));
    }, [dispatch]);
    return memoizedDispatch;
}
