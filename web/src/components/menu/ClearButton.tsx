import React from 'react';
import { useComfyApp } from '../../context/appContext.tsx';
import { useGraph } from '../../context/graphContext.tsx';

interface ClearButtonProps {
    confirmClear: {
        value: boolean;
    };
}

export function ClearButton({ confirmClear }: ClearButtonProps) {
    const { clean: cleanApp } = useComfyApp();
    const { graph } = useGraph();

    return (
        <button
            id="comfy-clear-button"
            onClick={() => {
                if (!confirmClear.value || confirm('Clear workflow?')) {
                    cleanApp();
                    graph.clear();
                }
            }}
        >
            Clear
        </button>
    );
}
