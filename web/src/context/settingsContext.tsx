// The container is used to provider dependency resolution for plugins

import React, { ReactNode, useEffect, useState } from 'react';
import { createUseContextHook } from './hookCreator';
import { ComfySettingsDialog } from '../components/ComfySettingsDialog.tsx';
import { api } from '../scripts/api.tsx';
import { useComfyApp } from './appContext.tsx';
import { BooleanInput, ComboInput, NumberInput, SliderInput, TextInput } from '../components/SettingInputs.tsx';
import { ComboOption } from '../types/many.ts';

interface ISettingsContext {
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
    const [openDialog, setOpenDialog] = useState<boolean>(false);

    const { storageLocation, isNewUserSession } = useComfyApp();

    const load = async () => {
        const settingsVal = storageLocation === 'browser' ? localStorage : await api.getSettings();
        setSettingsValues(settingsVal);

        // Trigger onChange for any settings added before load
        for (const id in settingsLookup) {
            settingsLookup[id].onChange?.(settingsValues[getId(id)]);
        }
    };

    const getId = (id: string) => {
        if (storageLocation === 'browser') {
            id = 'Comfy.Settings.' + id;
        }

        return id;
    };

    const getSettingValue = (id: string, defaultValue?: any) => {
        let value = settingsValues[getId(id)];
        if (!value) {
            setSettingsValues(prev => ({
                ...prev,
                [getId(id)]: defaultValue,
            }));
        }

        if (value) {
            if (storageLocation === 'browser') {
                try {
                    value = JSON.parse(value);
                } catch (error) {}
            }
        }

        return value ?? defaultValue;
    };

    const setSettingValueAsync = async (id: string, value: any) => {
        localStorage['Comfy.Settings.' + id] = JSON.stringify(value); // backwards compatibility for extensions keep setting in storage

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
        // TODO: 👇
        // for some weird reasons, reading `settingsLookup` directly in the `show` function doesn't work,
        // it keeps returning empty object, so I'm using this hack to get around it for now.
        // We should properly look into this later

        setSettingsLookup(settings => {
            setContent(() => {
                return [
                    <tr key={0} style={{ display: 'nones' }}>
                        <th />
                        <th style={{ width: '33%' }} />
                    </tr>,
                    ...Object.values(settings)
                        .sort((a, b) => a.name.localeCompare(b.name))
                        .map((s, i) => s.render(i + 1)),
                ];
            });

            setOpenDialog(true);
            return settings;
        });
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
            if (isNewUserSession) {
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
            render: (i: any) => {
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
                        <tr key={i}>
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
                                    onChange={onChange}
                                    setSettingValue={setSettingValue}
                                    value={getSettingValue(htmlID)}
                                />
                            );
                            break;
                        case 'number':
                            element = buildSettingInput(
                                <NumberInput
                                    id={htmlID}
                                    attrs={attrs}
                                    setter={setter}
                                    value={getSettingValue(htmlID)}
                                />
                            );
                            break;
                        case 'slider':
                            element = buildSettingInput(
                                <SliderInput
                                    id={htmlID}
                                    attrs={attrs}
                                    setter={setter}
                                    value={getSettingValue(htmlID)}
                                />
                            );
                            break;
                        case 'combo':
                            element = buildSettingInput(
                                <ComboInput setter={setter} options={options} value={getSettingValue(htmlID)} />
                            );
                            break;
                        case 'text':
                        default:
                            if (type !== 'text') {
                                console.warn(`Unsupported setting type '${type}, defaulting to text`);
                            }

                            element = buildSettingInput(
                                <TextInput id={htmlID} setter={setter} attrs={attrs} value={getSettingValue(htmlID)} />
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

        setSettingsLookup(prev => {
            const updated = {
                ...prev,
                [id]: setting,
            };

            console.log('Updated settingsLookup:', updated); // Log the updated settingsLookup
            return updated;
        });

        console.log('Added settings...');

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
                addSetting,
                setSettingValue,
                getSettingValue,
            }}
        >
            <ComfySettingsDialog closeDialog={() => setOpenDialog(false)} open={openDialog} content={content} />
            {children}
        </SettingsContext.Provider>
    );
};

export const useSettings = createUseContextHook(
    SettingsContext,
    'useSettings must be used within a SettingsContextProvider'
);
