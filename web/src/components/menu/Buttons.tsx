import React, { useEffect, useRef } from 'react';
import { usePrompt } from '../../hooks/usePrompt.tsx';
import { useComfyApp } from '../../context/appContext.tsx';
import { useLoadGraphData } from '../../hooks/useLoadGraphData.tsx';
import { useGraph } from '../../context/graphContext.tsx';

interface ButtonsProps {
    promptFilename: {
        value: boolean;
    };
    confirmClear: {
        value: boolean;
    };
}

export function Buttons({ promptFilename, confirmClear }: ButtonsProps) {
    const { clean: cleanApp } = useComfyApp();
    const { graphToPrompt } = usePrompt();
    const { graph } = useGraph();
    const { loadGraphData } = useLoadGraphData();

    return (
        <>
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

            <button
                id="comfy-dev-save-api-button"
                style={{ width: '100%', display: 'none' }}
                onClick={() => {
                    let filename: string | null = 'workflow_api.json';
                    if (promptFilename.value) {
                        filename = prompt('Save workflow (API) as:', filename);
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
                Save (API Format)
            </button>

            <button
                id="comfy-load-button"
                onClick={() => {
                    // TODO: handle this
                    // fileInput.click();
                }}
            >
                Load
            </button>

            <button
                id="comfy-refresh-button"
                onClick={() => {
                    /*app.refreshComboInNodes()*/
                }}
            >
                Refresh
            </button>
            <button id="comfy-clipspace-button" onClick={() => {}}>
                Clipspace
            </button>
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
        </>
    );
}
