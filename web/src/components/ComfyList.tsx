import React, { RefObject, useEffect, useState } from 'react';
import { api } from '../scripts/api.tsx';
import { ComfyItems } from '../types/api.ts';
import { useLoadGraphData } from '../hooks/useLoadGraphData.tsx';

interface ComfyListProps {
    text: string;
    type?: string;
    reverse?: boolean;

    show: boolean;
    buttonRef: RefObject<HTMLButtonElement>;
}

export const ComfyList = ({ show, buttonRef, text, type, reverse = false }: ComfyListProps) => {
    const [items, setItems] = useState({});

    const { loadGraphData } = useLoadGraphData();

    const load = async () => {
        const items = await api.getItems(type || text.toLowerCase());
        setItems(items);
    };

    useEffect(() => {
        if (show) {
            if (buttonRef.current) {
                buttonRef.current.textContent = 'Close';
            }

            load();
        } else {
            if (buttonRef.current) {
                buttonRef.current.textContent = 'View ' + text;
            }
        }
    }, [show]);

    return (
        <div className="comfy-list" style={{ display: show ? 'block' : 'none' }}>
            {Object.keys(items).flatMap(section => [
                <h4 key={section}>{section}</h4>,
                <div className="comfy-list-items" key={`${section}-items`}>
                    {(reverse
                        ? Object.values((items as ComfyItems[])[section as keyof typeof items]).reverse()
                        : Object.values(items[section as keyof typeof items])
                    ).map((item, index) => {
                        const removeAction = item.remove || {
                            name: 'Delete',
                            cb: () => api.deleteItem(type || text.toLowerCase(), item.prompt[1]),
                        };

                        return (
                            <div key={index}>
                                {item.prompt[0]}:
                                <button
                                    onClick={async () => {
                                        await loadGraphData(item.prompt[3].extra_pnginfo.workflow);
                                        if (item.outputs) {
                                            // app.nodeOutputs = item.outputs;
                                        }
                                    }}
                                >
                                    Load
                                </button>
                                <button
                                    onClick={async () => {
                                        await removeAction.cb();
                                        await load();
                                    }}
                                >
                                    {removeAction.name}
                                </button>
                            </div>
                        );
                    })}
                </div>,
            ])}

            <div className="comfy-list-actions">
                <button
                    onClick={async () => {
                        await api.clearItems(type ?? text.toLowerCase());
                        await load();
                    }}
                >
                    Clear {text}
                </button>

                <button onClick={load}>Refresh</button>
            </div>
        </div>
    );
};
