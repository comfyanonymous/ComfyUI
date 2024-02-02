// The container is used to provider dependency resolution for plugins

import React, { ReactNode, useState } from 'react';
import { createUseContextHook } from './hookCreator';
import { ComfySettingsDialog } from '../components/ComfySettingsDialog.tsx';
import { api } from '../scripts/api.tsx';
import { useComfyApp } from './appContext.tsx';

interface ISettingsContext {
    settings: any[];
    show: () => void;
    load: () => Promise<void>;
    getId: (id: string) => string;
    addSetting: (setting: IAddSetting) => any;
    setSettingValue: (id: string, value: any) => void;
    getSettingValue: (id: string, defaultValue?: any) => any;
}

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

const SettingsContext = React.createContext<ISettingsContext | null>(null);

export const SettingsContextProvider: React.FC = ({ children }) => {
    const [settingsValues, setSettingsValues] = useState<Record<string, any>>({});
    const [settingsLookup, setSettingsLookup] = useState<Record<string, any>>({});
    const [content, setContent] = useState<ReactNode>([]);
    const [openModal, setOpenModal] = useState<boolean>(false);

    const { app } = useComfyApp();

    const settings = Object.values(settingsLookup);

    const load = async () => {
        const settingsVal = app.storageLocation === 'browser' ? localStorage : await api.getSettings();
        setSettingsValues(settingsVal);

        // Trigger onChange for any settings added before load
        for (const id in settingsLookup) {
            settingsLookup[id].onChange?.(settingsValues[getId(id)]);
        }
    };

    const getId = (id: string) => {
        if (app.storageLocation === 'browser') {
            id = 'Comfy.Settings.' + id;
        }

        return id;
    };

    const getSettingValue = (id: string, defaultValue?: any) => {
        let value = settingsValues[getId(id)];
        if (value != null) {
            if (app.storageLocation === 'browser') {
                try {
                    value = JSON.parse(value);
                } catch (error) {}
            }
        }

        return value ?? defaultValue;
    };

    const setSettingValueAsync = async (id: string, value: any) => {
        const json = JSON.stringify(value);
        localStorage['Comfy.Settings.' + id] = json; // backwards compatibility for extensions keep setting in storage

        let oldValue = getSettingValue(id, undefined);
        setSettingsValues(prev => ({
            ...prev,
            [getId(id)]: value,
        }));

        if (id in settingsLookup) {
            settingsLookup[id].onChange?.(value, oldValue);
        }

        await api.storeSetting(id, value);
    };

    const setSettingValue = (id: string, value: any) => {
        setSettingValueAsync(id, value).catch(err => {
            alert(`Error saving setting '${id}'`);
            console.error(err);
        });
    };

    const show = () => {
        setContent(() => [
            <tr style={{ display: 'none' }}>
                <th />
                <th style={{ width: '33%' }} />
            </tr>,
            ...settings.sort((a, b) => a.name.localeCompare(b.name)).map(s => s.render()),
        ]);

        setOpenModal(true);
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
        if (!id) {
            throw new Error('Settings must have an ID');
        }

        if (id in settingsLookup) {
            throw new Error(`Setting ${id} of type ${type} must have a unique ID.`);
        }

        let skipOnChange = false;
        let value = getSettingValue(id);
        if (value == null) {
            if (app.isNewUserSession) {
                // Check if we have a localStorage value but not a setting value and we are a new user
                const localValue = localStorage['Comfy.Settings.' + id];
                if (localValue) {
                    value = JSON.parse(localValue);
                    setSettingValue(id, value); // Store on the server
                }
            }
            if (value == null) {
                value = defaultValue;
            }
        }

        // Trigger initial setting of value
        if (!skipOnChange) {
            onChange?.(value, undefined);
        }

        const setting = {
            id,
            onChange,
            name,
            render: () => {
                const setter = (v: any) => {
                    if (onChange) {
                        onChange(v, value);
                    }

                    setSettingValue(id, v);
                    value = v;
                };
                value = getSettingValue(id, defaultValue);

                let element: ReactNode;
                const htmlID = id.replaceAll('.', '-');

                const labelCell = (
                    <td>
                        <label htmlFor={htmlID} className={tooltip !== '' ? 'comfy-tooltip-indicator' : ''}>
                            {name}
                        </label>
                    </td>
                );

                if (typeof type === 'function') {
                    element = type(name, setter, value, attrs);
                } else {
                    switch (type) {
                        case 'boolean':
                            element = (
                                <tr>
                                    {labelCell}
                                    <td>
                                        <input
                                            id={htmlID}
                                            type="checkbox"
                                            checked={value}
                                            onChange={event => {
                                                const isChecked = event.target.checked;
                                                if (onChange !== undefined) {
                                                    onChange(isChecked);
                                                }
                                                setSettingValue(id, isChecked);
                                            }}
                                        />
                                    </td>
                                </tr>
                            );
                            break;
                        case 'number':
                            element = (
                                <tr>
                                    {labelCell}
                                    <td>
                                        <input
                                            type={type}
                                            value={value}
                                            id={htmlID}
                                            onInput={e => {
                                                const target = e.target as HTMLInputElement;
                                                setter(target.value);
                                            }}
                                            {...attrs}
                                        />
                                    </td>
                                </tr>
                            );
                            break;
                        case 'slider':
                            element = (
                                <tr>
                                    {labelCell}
                                    <td>
                                        <div style={{ display: 'grid', gridAutoFlow: 'column' }}>
                                            <input
                                                {...attrs}
                                                value={value}
                                                type="range"
                                                onInput={e => {
                                                    const target = e.target as HTMLInputElement;
                                                    setter(target.value);
                                                    if (target.nextElementSibling instanceof HTMLInputElement) {
                                                        target.nextElementSibling.value = target.value;
                                                    }
                                                }}
                                            />
                                            <input
                                                {...attrs}
                                                value={value}
                                                id={htmlID}
                                                type="number"
                                                style={{ maxWidth: '4rem' }}
                                                onInput={e => {
                                                    const target = e.target as HTMLInputElement;
                                                    setter(target.value);
                                                    if (target.previousElementSibling instanceof HTMLInputElement) {
                                                        target.previousElementSibling.value = target.value;
                                                    }
                                                }}
                                            />
                                        </div>
                                    </td>
                                </tr>
                            );
                            break;
                        case 'combo':
                            element = (
                                <tr>
                                    {labelCell}
                                    <td>
                                        <select
                                            onInput={e => {
                                                const target = e.target as HTMLSelectElement;
                                                setter(target.value);
                                            }}
                                        >
                                            {(typeof options === 'function' ? options(value) : options || []).map(
                                                (opt: any) => {
                                                    if (typeof opt === 'string') {
                                                        opt = { text: opt };
                                                    }
                                                    const v = opt.value ?? opt.text;
                                                    return (
                                                        <option value={v} selected={value + '' === v + ''}>
                                                            {opt.text}
                                                        </option>
                                                    );
                                                }
                                            )}
                                        </select>
                                    </td>
                                </tr>
                            );
                            break;
                        case 'text':
                        default:
                            if (type !== 'text') {
                                console.warn(`Unsupported setting type '${type}, defaulting to text`);
                            }

                            element = (
                                <tr>
                                    {labelCell}
                                    <td>
                                        <input
                                            value={value}
                                            id={htmlID}
                                            onInput={e => {
                                                const target = e.target as HTMLInputElement;
                                                setter(target.value);
                                            }}
                                            {...attrs}
                                        />
                                    </td>
                                </tr>
                            );
                            break;
                    }
                }
                if (tooltip) {
                    element = (
                        <div title={tooltip} style={{ display: 'inline-block' }}>
                            {element}
                        </div>
                    );
                }

                return element;
            },
        };

        setSettingsLookup(prev => ({
            ...prev,
            [id]: setting,
        }));

        return {
            get value() {
                return getSettingValue(id, defaultValue);
            },
            set value(v) {
                setSettingValue(id, v);
            },
        };
    };

    return (
        <SettingsContext.Provider
            value={{
                load,
                getId,
                show,
                settings,
                addSetting,
                setSettingValue,
                getSettingValue,
            }}
        >
            {children}
            <ComfySettingsDialog open={openModal} content={content} />
        </SettingsContext.Provider>
    );
};

export const useSettings = createUseContextHook(
    SettingsContext,
    'useSettings must be used within a SettingsContextProvider'
);
