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

    queueButtonRef: RefObject<HTMLButtonElement>;
    historyButtonRef: RefObject<HTMLButtonElement>;

    setShowQueue: (i: boolean) => void;
    setShowHistory: (i: boolean) => void;
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
    queueButtonRef,
    historyButtonRef,
    setShowQueue,
    setShowHistory,
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

            {/*

            $el("button", {
					$: (b) => (this.queue.button = b),
					id: "comfy-view-queue-button",
					textContent: "View Queue",
					onclick: () => {
						this.history.hide();
						this.queue.toggle();
					},
				}),
				$el("button", {
					$: (b) => (this.history.button = b),
					id: "comfy-view-history-button",
					textContent: "View History",
					onclick: () => {
						this.queue.hide();
						this.history.toggle();
					},
				}),
            */}

            <div className="comfy-menu-btns">
                <button id="queue-front-button" onClick={() => queuePrompt(-1)}>
                    Queue Front
                </button>
                <button
                    id="comfy-view-queue-button"
                    ref={queueButtonRef}
                    onClick={() => {
                        setShowHistory(false);
                        setShowQueue(i => !i);
                    }}
                >
                    View Queue
                </button>
                <button
                    id="comfy-view-history-button"
                    ref={historyButtonRef}
                    onClick={() => {
                        setShowQueue(false);
                        setShowHistory(i => !i);
                    }}
                >
                    View History
                </button>
            </div>

            {queue}
            {history}

            <Buttons promptFilename={promptFilename} confirmClear={confirmClear} />
        </div>
    );
}
