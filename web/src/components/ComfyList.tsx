import React, { useEffect, useRef, useState } from 'react';
import { api } from '../scripts/api.tsx';
import { ComfyItems } from '../types/api.ts';
import { useComfyApp } from '../context/appContext.tsx';

interface ComfyListProps {
    text: string;
    type: string;
    reverse?: boolean;
}

export const ComfyList = ({ text, type, reverse = false }: ComfyListProps) => {
    const [visible, setVisible] = useState(false);
    const [items, setItems] = useState({});

    const button = useRef<HTMLButtonElement>(null);
    const { app } = useComfyApp();

    const load = async () => {
        const items = await api.getItems(type || text.toLowerCase());
        setItems(items);
    };

    useEffect(() => {
        if (visible) {
            load();
        }
    }, [visible]);

    const show = async () => {
        setVisible(true);
        if (button.current) {
            button.current.textContent = 'Close';
        }

        await load();
    };

    const hide = () => {
        setVisible(false);
        if (button.current) {
            button.current.textContent = 'View ' + text;
        }
    };

    const toggle = () => {
        if (visible) {
            hide();
            return false;
        } else {
            show();
            return true;
        }
    };

    return (
        <div className="comfy-list" style={{ display: 'none' }}>
            <div style={{ display: visible ? 'block' : 'none' }}>
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
                                            await app.loadGraphData(item.prompt[3].extra_pnginfo.workflow);
                                            if (item.outputs) {
                                                app.nodeOutputs = item.outputs;
                                            }
                                        }}
                                    >
                                        Load
                                    </button>
                                    <button
                                        onClick={async () => {
                                            await removeAction.cb();
                                            load();
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
                            load();
                        }}
                    >
                        Clear {text}
                    </button>
                    <button onClick={load}>Refresh</button>
                </div>
            </div>
        </div>
    );
};

