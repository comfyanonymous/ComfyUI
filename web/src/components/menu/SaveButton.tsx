import React, { useEffect, useRef } from 'react';
import { usePrompt } from '../../hooks/usePrompt.tsx';

interface SaveButtonProps {
    promptFilename: {
        value: boolean;
    };
}

export function SaveButton({ promptFilename }: SaveButtonProps) {
    const { graphToPrompt } = usePrompt();

    return (
        <button
            id="comfy-save-button"
            onClick={() => {
                let filename: string | null = 'workflow.json';
                if (promptFilename.value) {
                    filename = prompt('Save workflow as:', filename);
                    if (!filename) return;
                    if (!filename.toLowerCase().endsWith('.json')) {
                        filename += '.json';
                    }
                }

                graphToPrompt().then((p: any) => {
                    const json = JSON.stringify(p.workflow, null, 2); // convert the data to a JSON string
                    const blob = new Blob([json], { type: 'application/json' });
                    const url = URL.createObjectURL(blob);

                    function Link() {
                        const ref = useRef<HTMLAnchorElement>(null);

                        useEffect(() => {
                            ref.current?.click();

                            return () => {
                                setTimeout(function () {
                                    ref.current?.remove();
                                    window.URL.revokeObjectURL(url);
                                }, 0);
                            };
                        }, []);

                        return <a ref={ref} href={url} download={filename} style={{ display: 'none' }} />;
                    }

                    return <Link />;
                });
            }}
        >
            Save
        </button>
    );
}
