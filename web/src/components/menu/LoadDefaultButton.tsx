import React from 'react';
import { useLoadGraphData } from '../../hooks/useLoadGraphData.tsx';

interface LoadDefaultButtonProps {
    confirmClear: {
        value: boolean;
    };
}

export function LoadDefaultButton({ confirmClear }: LoadDefaultButtonProps) {
    const { loadGraphData } = useLoadGraphData();

    return (
        <button
            id="comfy-load-default-button"
            onClick={async () => {
                if (!confirmClear.value || confirm('Load default workflow?')) {
                    await loadGraphData();
                }
            }}
        >
            Load Default
        </button>
    );
}
