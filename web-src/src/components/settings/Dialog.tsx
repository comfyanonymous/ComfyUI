import React, { useEffect, useState } from 'react';
import { app } from '../../scripts/app.ts';
import { api } from '../../scripts/api.tsx';

interface IAddSetting {
    id: string;
    name: string;
    type: any;
    defaultValue: any;
    onChange?: Function;
    attrs?: Object;
    tooltip?: string;
    options?: any[] | Function;
}

enum StorageLocation {
    Browser = 'browser',
    Server = 'server',
}

export function ComfySettingsDialog() {
    const [settingsValues, setSettingsValues] = useState<Record<string, any>>({});
    const [settingsLookup, setSettingsLookup] = useState<Record<string, any>>({});
    const [visible, setVisible] = useState(false);

    useEffect(() => {
        const load = async () => {
            let values;
            const comfySettings = localStorage.getItem('Comfy.Settings');
            if (comfySettings) {
                values = JSON.parse(comfySettings);
            } else {
                values = await api.getSettings();
            }
            setSettingsValues(values);

            for (const id in settingsLookup) {
                settingsLookup[id].onChange?.(values[id]);
            }
        };

        load();
    }, [settingsLookup]);

    const setSettingValue = async (id: string, value: any) => {
        const values = { ...settingsValues, [id]: value };
        setSettingsValues(values);

        if (id in settingsLookup) {
            settingsLookup[id].onChange?.(value, settingsValues[id]);
        }

        await api.storeSetting(id, value);
    };

    const getId = (id: string) => {
        if (app.storageLocation === StorageLocation.Browser) {
            id = 'Comfy.Settings.' + id;
        }

        return id;
    };

    const getSettingValue = (id: string, defaultValue?: any) => {
        let value = settingsValues[getId(id)];
        if (!!value) {
            if (app.storageLocation === StorageLocation.Browser) {
                try {
                    value = JSON.parse(value);
                } catch (error) {
                    console.log('An error occurred while parsing the setting value', { error });
                }
            }
        }

        return value ?? defaultValue;
    };

    const addSetting = ({
        id,
        name,
        type,
        defaultValue,
        onChange,
        attrs = {},
        tooltip = '',
        options = [],
    }: IAddSetting) => {
        let value = settingsValues[id];
        if (value == null) {
            value = defaultValue;
        }

        onChange?.(value, undefined);

        setSettingsLookup(() => ({
            ...settingsLookup,
            [id]: {
                id,
                onChange,
                name,
                type,
                defaultValue,
                attrs,
                tooltip,
                options,
            },
        }));
    };

    const show = () => {
        setVisible(true);
    };

    return (
        <dialog id={'comfy-settings-dialog'}>
            <table className={'comfy-modal-content comfy-table'}>
                <caption>Settings</caption>
                <tbody>
                    {Object.values(settingsLookup).map((setting: any) => {
                        const { id, name, type, defaultValue, attrs, tooltip, options } = setting;
                        const value = getSettingValue(id, defaultValue);
                        let input;
                        switch (type) {
                            case 'boolean':
                                input = (
                                    <input
                                        type={'checkbox'}
                                        checked={value}
                                        onChange={event => {
                                            setSettingValue(id, event.target.checked);
                                        }}
                                        {...attrs}
                                    />
                                );
                                break;
                            case 'select':
                                input = (
                                    <select
                                        value={value}
                                        onChange={event => {
                                            setSettingValue(id, event.target.value);
                                        }}
                                        {...attrs}
                                    >
                                        {options.map((option: any) => {
                                            const { value, label } = option;
                                            return (
                                                <option key={value} value={value}>
                                                    {label}
                                                </option>
                                            );
                                        })}
                                    </select>
                                );
                                break;
                            case 'slider':
                                input = (
                                    <>
                                        <input
                                            {...attrs}
                                            value={value}
                                            type="range"
                                            onChange={event => {
                                                setSettingValue(id, event.target.value);
                                            }}
                                        />
                                        <input
                                            {...attrs}
                                            value={value}
                                            id={id}
                                            type="number"
                                            style={{ maxWidth: '4rem' }}
                                            onInput={event => {
                                                setSettingValue(id, (event.target as HTMLInputElement).value);
                                            }}
                                        />
                                    </>
                                );
                                break;
                            case 'number':
                                input = (
                                    <input
                                        type={'number'}
                                        value={value}
                                        onChange={event => {
                                            setSettingValue(id, event.target.value);
                                        }}
                                        {...attrs}
                                    />
                                );
                                break;
                            case 'text':
                                input = (
                                    <input
                                        type={'text'}
                                        value={value}
                                        onChange={event => {
                                            setSettingValue(id, event.target.value);
                                        }}
                                        {...attrs}
                                    />
                                );
                                break;
                            case 'textarea':
                                input = (
                                    <textarea
                                        value={value}
                                        onChange={event => {
                                            setSettingValue(id, event.target.value);
                                        }}
                                        {...attrs}
                                    />
                                );
                                break;
                            default:
                                input = (
                                    <input
                                        type={'text'}
                                        value={value}
                                        onChange={event => {
                                            setSettingValue(id, event.target.value);
                                        }}
                                        {...attrs}
                                    />
                                );
                                break;
                        }

                        return (
                            <tr key={id}>
                                <td>
                                    {name}
                                    {tooltip && (
                                        <span className={'comfy-tooltip'}>
                                            <span className={'comfy-tooltip-text'}>{tooltip}</span>
                                        </span>
                                    )}
                                </td>
                                <td>{input}</td>
                            </tr>
                        );
                    })}
                </tbody>

                <button type={'button'} style={{ cursor: 'pointer' }} onClick={() => setVisible(false)}>
                    Close
                </button>
            </table>
        </dialog>
    );
}
