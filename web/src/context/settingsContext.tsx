// The container is used to provider dependency resolution for plugins

import React, { ReactNode, useState } from 'react';
import { createUseContextHook } from './hookCreator';
import { ComfySettingsDialog } from '../components/ComfySettingsDialog.tsx';
import { api } from '../scripts/api.tsx';
import { useComfyApp } from './appContext.tsx';
import { BooleanInput, ComboInput, NumberInput, SliderInput, TextInput } from '../components/SettingInputs.tsx';
import { ComboOption } from '../types/many.ts';

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
    type: any;
    id: string;
    name: string;
    attrs?: Object;
    tooltip?: string;
    defaultValue?: any;
    onChange?: (...arg: any[]) => any;
    options?: ComboOption[] | ((value: string) => (ComboOption | string)[]);
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

                function buildSettingInput(element: ReactNode) {
                    return (
                        <tr>
                            <td>
                                <label htmlFor={htmlID} className={tooltip !== '' ? 'comfy-tooltip-indicator' : ''}>
                                    {name}
                                </label>
                            </td>
                            
                            <td>{element}</td>
                        </tr>
                    );
                }

                if (typeof type === 'function') {
                    element = type(name, setter, value, attrs);
                } else {
                    switch (type) {
                        case 'boolean':
                            element = buildSettingInput(
                                <BooleanInput
                                    id={htmlID}
                                    value={value}
                                    onChange={onChange}
                                    setSettingValue={setSettingValue}
                                />
                            );
                            break;
                        case 'number':
                            element = buildSettingInput(
                                <NumberInput id={htmlID} value={value} setter={setter} attrs={attrs} />
                            );
                            break;
                        case 'slider':
                            element = buildSettingInput(
                                <SliderInput id={htmlID} value={value} setter={setter} attrs={attrs} />
                            );
                            break;
                        case 'combo':
                            element = buildSettingInput(<ComboInput value={value} setter={setter} options={options} />);
                            break;
                        case 'text':
                        default:
                            if (type !== 'text') {
                                console.warn(`Unsupported setting type '${type}, defaulting to text`);
                            }

                            element = buildSettingInput(
                                <TextInput id={htmlID} value={value} setter={setter} attrs={attrs} />
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
