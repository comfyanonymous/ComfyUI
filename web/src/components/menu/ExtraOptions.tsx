import React, { ReactNode, RefObject } from 'react';
import { usePrompt } from '../../hooks/usePrompt.tsx';

interface ExtraOptionsProps {
    setAutoQueueEnabled: (i: boolean) => void;
    setBatchCount: (i: number) => void;
    batchCount: number;
    autoQueueEnabled: boolean;
    autoQueueModeElRef: RefObject<HTMLDivElement>;
    autoQueueModeEl: ReactNode;
}

export function ExtraOptions({
    batchCount,
    setBatchCount,
    autoQueueEnabled,
    autoQueueModeEl,
    setAutoQueueEnabled,
    autoQueueModeElRef,
}: ExtraOptionsProps) {
    return (
        <>
            <div>
                <label>
                    Extra Options
                    <input
                        type="checkbox"
                        onChange={i => {
                            let extraOptions = document.getElementById('extraOptions');
                            if (extraOptions) {
                                // extraOptions.style.display = i.srcElement.checked ? 'block' : 'none';
                                extraOptions.style.display = i.target.checked ? 'block' : 'none';

                                let batchCountInputRange = document.getElementById(
                                    'batchCountInputRange'
                                ) as HTMLInputElement;
                                // this.batchCount = i.srcElement.checked ? Number(batchCountInputRange.value) : 1;
                                setBatchCount(i.target.checked ? Number(batchCountInputRange.value) : 1);

                                let autoQueueCheckbox = document.getElementById(
                                    'autoQueueCheckbox'
                                ) as HTMLInputElement;
                                if (autoQueueCheckbox) {
                                    autoQueueCheckbox.checked = false;
                                }

                                setAutoQueueEnabled(false);
                            }
                        }}
                    />
                </label>
            </div>

            <div id="extraOptions" style={{ width: '100%', display: 'none' }}>
                <div>
                    <label>Batch Count</label>
                    <input
                        min={1}
                        type="number"
                        value={batchCount}
                        id="batchCountInputNumber"
                        style={{ width: '35%', marginLeft: '0.4em' }}
                        onInput={i => {
                            setBatchCount((i.target as any).value);
                            let batchCountInputRange = document.getElementById(
                                'batchCountInputRange'
                            ) as HTMLInputElement | null;

                            if (batchCountInputRange) {
                                batchCountInputRange.value = batchCount.toString();
                            }
                        }}
                    />

                    <input
                        type="range"
                        min={1}
                        max={100}
                        value={batchCount}
                        id="batchCountInputRange"
                        onInput={i => {
                            setBatchCount((i.target as any).value);
                            let batchCountInputNumber = document.getElementById(
                                'batchCountInputNumber'
                            ) as HTMLInputElement | null;
                            if (batchCountInputNumber) {
                                batchCountInputNumber.value = (i.target as any).value;
                            }
                        }}
                    />
                </div>

                <div>
                    <label htmlFor="autoQueueCheckbox">Auto Queue</label>
                    <input
                        id="autoQueueCheckbox"
                        type="checkbox"
                        checked={autoQueueEnabled}
                        title="Automatically queue prompt when the queue size hits 0"
                        onChange={e => {
                            setAutoQueueEnabled(e.target.checked);
                            if (autoQueueModeElRef.current) {
                                autoQueueModeElRef.current.style.display = autoQueueEnabled ? '' : 'none';
                            }
                        }}
                    />
                    {autoQueueModeEl}
                </div>
            </div>
        </>
    );
}
