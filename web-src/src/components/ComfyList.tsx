import React, { useEffect, useState } from 'react';
import { api } from '../context/api.tsx';

const ComfyList = ({ text, type, reverse }) => {
    const [visible, setVisible] = useState(false);
    const [items, setItems] = useState({});

    const load = async () => {
        const items = await api.getItems(type || text.toLowerCase());
        setItems(items);
    };

    useEffect(() => {
        if (visible) {
            load();
        }
    }, [visible]);

    const toggle = () => {
        setVisible(prev => !prev);
    };

    return (
        <div style={{ display: visible ? 'block' : 'none' }}>
            {Object.keys(items).flatMap(section => [
                <h4 key={section}>{section}</h4>,
                <div key={`${section}-items`}>
                    {(reverse ? Object.values(items[section]).reverse() : Object.values(items[section])).map(
                        (item, index) => {
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
                        }
                    )}
                    ,
                </div>,
            ])}
            <div>
                <button
                    onClick={async () => {
                        await api.clearItems(type || text.toLowerCase());
                        load();
                    }}
                >
                    Clear {text}
                </button>
                <button onClick={load}>Refresh</button>
            </div>
        </div>
    );
};

export default ComfyList;
