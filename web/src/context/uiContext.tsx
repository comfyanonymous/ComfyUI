import React, { ReactNode, useEffect, useRef, useState } from 'react';
import { createUseContextHook } from './hookCreator';
import { useSettings } from './settingsContext.tsx';
import { api } from '../scripts/api.tsx';
import { ComfyPromptStatus } from '../types/comfy.ts';
import { useComfyApp } from './appContext.tsx';
import { usePrompt } from '../hooks/usePrompt.tsx';
import { dragElement, toggleSwitch } from '../utils/ui.tsx';
import { ComfyMenu } from '../components/menu/Menu.tsx';
import { ComfyList } from '../components/ComfyList.tsx';

type AutoQueueMode =
    | {
          text: string;
          value?: string;
          tooltip?: string;
      }
    | string
    | null;

interface ComfyUIContextType {}

const ComfyUIContext = React.createContext<ComfyUIContextType | null>(null);

export const ComfyUIContextProvider = ({ children }: { children: ReactNode }) => {
    const { addSetting } = useSettings();
    const { queuePrompt } = usePrompt();
    const { lastExecutionError } = useComfyApp();

    const [batchCount, setBatchCount] = useState(1);
    const [lastQueueSize, setLastQueueSize] = useState(0);
    const [queue, setQueue] = useState<ReactNode>([]);
    const [history, setHistory] = useState<ReactNode>([]);
    const [autoQueueMode, setAutoQueueMode] = useState<AutoQueueMode>(null);
    const [graphHasChanged, setGraphHasChanged] = useState(false);
    const [autoQueueEnabled, setAutoQueueEnabled] = useState(false);
    const [confirmClear, setConfirmClear] = useState<{ value: boolean }>({ value: false });
    const [promptFilename, setPromptFilename] = useState<{ value: boolean }>({ value: false });

    const [showQueue, setShowQueue] = useState(false);
    const [showHistory, setShowHistory] = useState(false);

    const menuContainerEl = useRef<HTMLDivElement>(null);
    const queueButtonRef = useRef<HTMLButtonElement>(null);
    const historyButtonRef = useRef<HTMLButtonElement>(null);
    const queueSizeEl = useRef<HTMLSpanElement>(null);
    const autoQueueModeElRef = useRef<HTMLDivElement>(null);

    const setStatus = (status: ComfyPromptStatus) => {
        if (!queueSizeEl.current) return;
        queueSizeEl.current.textContent = 'Queue size: ' + (status ? status.exec_info.queue_remaining : 'ERR');

        if (status) {
            if (
                lastQueueSize != 0 &&
                status.exec_info.queue_remaining == 0 &&
                autoQueueEnabled &&
                (autoQueueMode === 'instant' || graphHasChanged) &&
                !lastExecutionError
            ) {
                queuePrompt(0);
                status.exec_info.queue_remaining += batchCount;
                setGraphHasChanged(false);
            }
            setLastQueueSize(status.exec_info.queue_remaining);
        }
    };

    useEffect(() => {
        api.addEventListener('status', () => {
            // this.queue.update();
            // this.history.update();
        });

        const confirmClear = addSetting({
            id: 'Comfy.ConfirmClear',
            name: 'Require confirmation when clearing workflow',
            type: 'boolean',
            defaultValue: true,
            onChange: () => undefined,
        });
        setConfirmClear(() => confirmClear);

        const promptFilename = addSetting({
            id: 'Comfy.PromptFilename',
            name: 'Prompt for filename when saving workflow',
            type: 'boolean',
            defaultValue: false,
            onChange: () => undefined,
        });
        setPromptFilename(() => promptFilename);

        /* file format for preview
         *
         * format;quality
         *
         * ex)
         * webp;50 -> webp, quality 50
         * jpeg;80 -> rgb, jpeg, quality 80
         *
         * @type {string}
         */

        addSetting({
            id: 'Comfy.PreviewFormat',
            name: 'When displaying a preview in the image widget, convert it to a lightweight image, e.g. webp, jpeg, webp;50, etc.',
            type: 'text',
            defaultValue: '',
            onChange: () => undefined,
        });

        addSetting({
            id: 'Comfy.DisableSliders',
            name: 'Disable sliders.',
            type: 'boolean',
            defaultValue: false,
            onChange: () => undefined,
        });

        addSetting({
            id: 'Comfy.DisableFloatRounding',
            name: 'Disable rounding floats (requires page reload).',
            type: 'boolean',
            defaultValue: false,
            onChange: () => undefined,
        });

        addSetting({
            id: 'Comfy.FloatRoundingPrecision',
            name: 'Decimal places [0 = auto] (requires page reload).',
            type: 'slider',
            attrs: {
                min: 0,
                max: 6,
                step: 1,
            },
            defaultValue: 0,
            onChange: () => undefined,
        });

        api.addEventListener('graphChanged', () => {
            if (autoQueueMode === 'change' && autoQueueEnabled) {
                if (lastQueueSize === 0) {
                    setGraphHasChanged(false);
                    queuePrompt(0);
                } else {
                    setGraphHasChanged(true);
                }
            }
        });

        addSetting({
            id: 'Comfy.DevMode',
            name: 'Enable Dev mode Options',
            type: 'boolean',
            defaultValue: false,
            onChange: function (value: string) {
                const devSaveApiButton = document.getElementById('comfy-dev-save-api-button');
                if (devSaveApiButton) {
                    devSaveApiButton.style.display = value ? 'block' : 'none';
                }
            },
        });
    }, []);

    useEffect(() => {
        setQueue(<ComfyList text="Queue" show={showQueue} buttonRef={queueButtonRef} />);
        setHistory(<ComfyList text="History" show={showHistory} reverse={true} buttonRef={historyButtonRef} />);
    }, [showQueue, showHistory]);

    const autoQueueModeEl = toggleSwitch(
        'autoQueueMode',
        [
            { text: 'instant', tooltip: 'A new prompt will be queued as soon as the queue reaches 0' },
            {
                text: 'change',
                tooltip: 'A new prompt will be queued when the queue is at 0 and the graph is/has changed',
            },
        ],
        {
            ref: autoQueueModeElRef,
            onChange: value => {
                setAutoQueueMode(value.item.value);
            },
        }
    );

    if (autoQueueModeElRef.current) {
        autoQueueModeElRef.current.style.display = 'none';
    }

    const fileInput = (
        <input
            id="comfy-file-input"
            type="file"
            accept=".json,image/png,.latent,.safetensors,image/webp"
            style={{ display: 'none' }}
            onChange={() => {
                if ('files' in fileInput && Array.isArray(fileInput.files)) {
                    // app.handleFile(fileInput.files[0]);
                }
            }}
        />
    );

    useEffect(() => {
        setStatus({ exec_info: { queue_remaining: 'X' } });
    }, [queueSizeEl]);

    useEffect(() => {
        if (menuContainerEl.current) {
            dragElement(menuContainerEl.current, addSetting);
        }
    }, [menuContainerEl]);

    return (
        <ComfyUIContext.Provider value={{}}>
            <ComfyMenu
                queue={queue}
                history={history}
                setAutoQueueEnabled={setAutoQueueEnabled}
                setBatchCount={setBatchCount}
                batchCount={batchCount}
                autoQueueEnabled={autoQueueEnabled}
                autoQueueModeElRef={autoQueueModeElRef}
                autoQueueModeEl={autoQueueModeEl}
                promptFilename={promptFilename}
                confirmClear={confirmClear}
                menuContainerEl={menuContainerEl}
                queueSizeEl={queueSizeEl}
                queueButtonRef={queueButtonRef}
                historyButtonRef={historyButtonRef}
                setShowQueue={setShowQueue}
                setShowHistory={setShowHistory}
            />

            {children}
        </ComfyUIContext.Provider>
    );
};

export const useComfyUI = createUseContextHook(ComfyUIContext, 'ComfyUIContext not found');
