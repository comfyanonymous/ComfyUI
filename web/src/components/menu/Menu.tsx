import React, { ReactNode, RefObject } from 'react';
import { useSettings } from '../../context/settingsContext.tsx';
import { usePrompt } from '../../hooks/usePrompt.tsx';
import { ExtraOptions } from './ExtraOptions.tsx';
import { Buttons } from './Buttons.tsx';

interface MenuProps {
    queue: ReactNode;
    history: ReactNode;

    setAutoQueueEnabled: (i: boolean) => void;
    setBatchCount: (i: number) => void;
    batchCount: number;
    autoQueueEnabled: boolean;
    autoQueueModeElRef: RefObject<HTMLDivElement>;
    autoQueueModeEl: ReactNode;

    promptFilename: { value: boolean };
    confirmClear: { value: boolean };

    menuContainerEl: RefObject<HTMLDivElement>;
    queueSizeEl: RefObject<HTMLSpanElement>;
}

export function ComfyMenu({
    menuContainerEl,
    queueSizeEl,
    batchCount,
    setBatchCount,
    autoQueueEnabled,
    autoQueueModeEl,
    setAutoQueueEnabled,
    autoQueueModeElRef,
    promptFilename,
    confirmClear,
    queue,
    history,
}: MenuProps) {
    const { show: showSettings } = useSettings();
    const { queuePrompt, graphToPrompt } = usePrompt();

    return (
        <div className="comfy-menu" ref={menuContainerEl}>
            <div
                className="drag-handle"
                style={{
                    overflow: 'hidden',
                    position: 'relative',
                    cursor: 'default',
                    width: '100%',
                }}
            >
                <span className="drag-handle" />
                <span ref={queueSizeEl} />
                <button className="comfy-settings-btn" onClick={showSettings}>
                    ⚙️
                </button>
            </div>

            <button id="queue-button" className="comfy-queue-btn" onClick={() => queuePrompt(0)}>
                Queue Prompt
            </button>

            <ExtraOptions
                setAutoQueueEnabled={setAutoQueueEnabled}
                setBatchCount={setBatchCount}
                batchCount={batchCount}
                autoQueueEnabled={autoQueueEnabled}
                autoQueueModeElRef={autoQueueModeElRef}
                autoQueueModeEl={autoQueueModeEl}
            />

            {queue}
            {history}

            <Buttons promptFilename={promptFilename} confirmClear={confirmClear} />
        </div>
    );
}
