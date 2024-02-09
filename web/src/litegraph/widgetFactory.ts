// "Widgets" are input-fields that can exist inside of nodes. This module allows the creation of
// new widget-types, and manages their state.

import { api } from '../scripts/api.js';
import { ComfyWidget, comfyWidgetTypes } from '../types/comfyWidget.js';
import { ComfyNode } from './comfyNode.js';
import { ComfyFile } from '../types/many.js';
import { GetCustomWidgetResponse } from '../types/interfaces.js';

interface WidgetReturnType {
    minWidth?: number;
    minHeight?: number;
    widget?: ComfyWidget;
}

// Define the type for the widget creation functions
type WidgetCreationFunction =
    | ((node: ComfyNode, inputName: string, inputData: any) => WidgetReturnType)
    | ((node: ComfyNode, inputName: string, inputData: any, widgetName: string) => WidgetReturnType);

// Define the structure of the widgets collection
export interface IWidgetFactory {
    [key: string]: WidgetCreationFunction;
}

let controlValueRunBefore = false;

export function updateControlWidgetLabel(widget: ComfyWidget) {
    let replacement = 'after';
    let find = 'before';
    if (controlValueRunBefore) {
        [find, replacement] = [replacement, find];
    }
    widget.label = (widget.label ?? widget.name)?.replace(find, replacement);
}

const IS_CONTROL_WIDGET = Symbol();
const HAS_EXECUTED = Symbol();

interface InputData {
    min?: number;
    max?: number;
    step?: number;
    round?: boolean | number;
    control_prefix?: string;
    control_after_generate?: string;

    [key: string]: any;
}

interface InputOptions {
    addFilterList?: boolean;
    controlAfterGenerateName?: string;

    [key: string]: any;
}

function getNumberDefaults(
    inputData: InputData[],
    defaultStep: number,
    precision: number | undefined,
    enable_rounding: boolean
) {
    let defaultVal = inputData[1]['default'];
    let { min, max, step, round } = inputData[1];

    if (defaultVal == undefined) defaultVal = 0;
    if (min == undefined) min = 0;
    if (max == undefined) max = 2048;
    if (step == undefined) step = defaultStep;
    // precision is the number of decimal places to show.
    // by default, display the the smallest number of decimal places such that changes of size step are visible.
    if (precision == undefined) {
        precision = Math.max(-Math.floor(Math.log10(step)), 0);
    }

    if (enable_rounding && (round == undefined || round === true)) {
        // by default, round the value to those decimal places shown.
        round = Math.round(1000000 * Math.pow(0.1, precision)) / 1000000;
    }

    return { val: defaultVal, config: { min, max, step: 10.0 * step, round, precision } };
}

export function addValueControlWidget(
    node: ComfyNode,
    targetWidget: ComfyWidget,
    defaultValue = 'randomize',
    values?: any,
    widgetName?: string,
    inputData?: InputData[]
) {
    let name = inputData?.[1]?.control_after_generate;
    if (typeof name !== 'string') {
        name = widgetName;
    }
    const widgets = addValueControlWidgets(
        node,
        targetWidget,
        defaultValue,
        {
            addFilterList: false,
            controlAfterGenerateName: name,
        },
        inputData
    );
    return widgets[0];
}

export function addValueControlWidgets(
    node: ComfyNode,
    targetWidget: ComfyWidget,
    defaultValue = 'randomize',
    options?: InputOptions,
    inputData?: InputData[]
) {
    if (!defaultValue) defaultValue = 'randomize';
    if (!options) options = {};

    const getName = (defaultName: string, optionName: string) => {
        let name = defaultName;
        if (options?.[optionName]) {
            name = options[optionName];
        } else if (typeof inputData?.[1]?.[defaultName] === 'string') {
            name = inputData?.[1]?.[defaultName];
        } else if (inputData?.[1]?.control_prefix) {
            name = inputData?.[1]?.control_prefix + ' ' + name;
        }
        return name;
    };

    const widgets = [];
    const valueControl = node.addWidget<ComfyWidget>(
        'combo',
        getName('control_after_generate', 'controlAfterGenerateName'),
        defaultValue,
        function () {},
        {
            values: ['fixed', 'increment', 'increment-wrap', 'decrement', 'randomize'],
            serialize: false, // Don't include this in prompt.
        }
    );
    valueControl[IS_CONTROL_WIDGET] = true;
    updateControlWidgetLabel(valueControl);
    widgets.push(valueControl);

    const isCombo = targetWidget.type === 'combo';
    let comboFilter: ComfyWidget;
    if (isCombo) {
        valueControl.options.values.push('increment-wrap');
    }
    if (isCombo && options.addFilterList !== false) {
        comboFilter = node.addWidget<ComfyWidget>(
            'string' as comfyWidgetTypes,
            getName('control_filter_list', 'controlFilterListName'),
            '',
            function () {},
            {
                serialize: false, // Don't include this in prompt.
            }
        );
        updateControlWidgetLabel(comboFilter);

        widgets.push(comboFilter);
    }

    const applyWidgetControl = () => {
        var v = valueControl.value;

        if (isCombo && v !== 'fixed') {
            let values = targetWidget.options.values;
            const filter = comboFilter?.value;
            if (filter) {
                let check: ((item: string) => boolean) | undefined;
                if (filter.startsWith('/') && filter.endsWith('/')) {
                    try {
                        const regex = new RegExp(filter.substring(1, filter.length - 1));
                        check = item => regex.test(item);
                    } catch (error) {
                        console.error('Error constructing RegExp filter for node ' + node.id, filter, error);
                    }
                }
                if (!check) {
                    const lower = filter.toLocaleLowerCase();
                    check = item => item.toLocaleLowerCase().includes(lower);
                }
                values = values.filter((item: string) => check?.(item));
                if (!values.length && targetWidget.options.values.length) {
                    console.warn('Filter for node ' + node.id + ' has filtered out all items', filter);
                }
            }
            let current_index = values.indexOf(targetWidget.value);
            let current_length = values.length;

            switch (v) {
                case 'increment':
                    current_index += 1;
                    break;
                case 'increment-wrap':
                    current_index += 1;
                    if (current_index >= current_length) {
                        current_index = 0;
                    }
                    break;
                case 'decrement':
                    current_index -= 1;
                    break;
                case 'randomize':
                    current_index = Math.floor(Math.random() * current_length);
                    break;
                default:
                    break;
            }
            current_index = Math.max(0, current_index);
            current_index = Math.min(current_length - 1, current_index);
            if (current_index >= 0) {
                let value = values[current_index];
                targetWidget.value = value;
                targetWidget.callback?.(value);
            }
        } else {
            //number
            let min = targetWidget.options.min;
            let max = targetWidget.options.max;
            // limit to something that javascript can handle
            max = Math.min(1125899906842624, max);
            min = Math.max(-1125899906842624, min);
            let range = (max - min) / (targetWidget.options.step / 10);

            //adjust values based on valueControl Behaviour
            switch (v) {
                case 'fixed':
                    break;
                case 'increment':
                    targetWidget.value += targetWidget.options.step / 10;
                    break;
                case 'decrement':
                    targetWidget.value -= targetWidget.options.step / 10;
                    break;
                case 'randomize':
                    targetWidget.value = Math.floor(Math.random() * range) * (targetWidget.options.step / 10) + min;
                    break;
                default:
                    break;
            }
            /*check if values are over or under their respective
             * ranges and set them to min or max.*/
            if (targetWidget.value < min) targetWidget.value = min;

            if (targetWidget.value > max) targetWidget.value = max;
            targetWidget.callback?.(targetWidget.value);
        }
    };

    valueControl.beforeQueued = () => {
        if (controlValueRunBefore) {
            // Don't run on first execution
            if (valueControl[HAS_EXECUTED]) {
                applyWidgetControl();
            }
        }
        valueControl[HAS_EXECUTED] = true;
    };

    valueControl.afterQueued = () => {
        if (!controlValueRunBefore) {
            applyWidgetControl();
        }
    };

    return widgets;
}

function seedWidget(node: ComfyNode, inputName: string, inputData: InputData[], widgetName?: string) {
    const seed = createIntWidget(node, inputName, inputData, true);
    const seedControl = addValueControlWidget(node, seed.widget, 'randomize', undefined, widgetName, inputData);

    seed.widget.linkedWidgets = [seedControl];
    return seed;
}

function createIntWidget(
    node: ComfyNode,
    inputName: string,
    inputData: InputData[],
    isSeedInput?: boolean
): {
    widget: ComfyWidget;
} {
    const control: string | undefined = inputData[1]?.control_after_generate;
    if (!isSeedInput) {
        return seedWidget(node, inputName, inputData, typeof control === 'string' ? control : undefined);
    }

    let widgetType = isSlider(inputData[1]['display']) as comfyWidgetTypes;
    const { val, config } = getNumberDefaults(inputData, 1, 0, true);
    Object.assign(config, { precision: 0 });
    return {
        widget: node.addWidget<ComfyWidget>(
            widgetType,
            inputName,
            val,
            function (v) {
                const s = this.options.step / 10;
                this.value = Math.round(v / s) * s;
            },
            config
        ),
    };
}

function addMultilineWidget(node: ComfyNode, name: string, opts: InputOptions) {
    const inputEl = document.createElement('textarea');
    inputEl.className = 'comfy-multiline-input';
    inputEl.value = opts.defaultVal;
    inputEl.placeholder = opts.placeholder || name;

    const widget = node.addDOMWidget(name, 'customtext', inputEl, {
        getValue() {
            return inputEl.value;
        },
        setValue(v: any) {
            inputEl.value = v;
        },
    });
    if (widget) {
        widget.inputEl = inputEl;

        inputEl.addEventListener('input', () => {
            widget.callback?.(widget.value);
        });
    }
    return { minWidth: 400, minHeight: 200, widget };
}

function isSlider(display: string) {
    if (app.ui.settings.getSettingValue('Comfy.DisableSliders')) {
        return 'number';
    }

    return display === 'slider' ? 'slider' : 'number';
}

export function initWidgets(app: ComfyApp) {
    app.ui.settings.addSetting({
        id: 'Comfy.WidgetControlMode',
        name: 'Widget Value Control Mode',
        type: 'combo',
        defaultValue: 'after',
        options: ['before', 'after'],
        tooltip:
            'Controls when widget values are updated (randomize/increment/decrement), either before the prompt is queued or after.',
        onChange(value: string) {
            controlValueRunBefore = value === 'before';
            for (const n of app.graph?.nodes || []) {
                if (!n.widgets) continue;
                for (const w of n.widgets) {
                    if (w[IS_CONTROL_WIDGET]) {
                        updateControlWidgetLabel(w);
                        if (w.linkedWidgets) {
                            for (const l of w.linkedWidgets) {
                                updateControlWidgetLabel(l);
                            }
                        }
                    }
                }
            }
            app.graph?.setDirtyCanvas(true, false);
        },
    });
}

/** Collection of factory-functions that add widgets to nodes */
export const WidgetFactory: IWidgetFactory = {
    'INT:seed': seedWidget,
    'INT:noise_seed': seedWidget,
    FLOAT(node: ComfyNode, inputName: string, inputData: InputData[], app: ComfyApp): { widget: ComfyWidget } {
        let widgetType = isSlider(inputData[1]['display'], app) as comfyWidgetTypes;
        let precision = app.ui.settings.getSettingValue('Comfy.FloatRoundingPrecision');
        let disable_rounding = app.ui.settings.getSettingValue('Comfy.DisableFloatRounding');
        if (precision == 0) precision = undefined;
        const { val, config } = getNumberDefaults(inputData, 0.5, precision, !disable_rounding);
        return {
            widget: node.addWidget(
                widgetType,
                inputName,
                val,
                function (v: number) {
                    if (config.round) {
                        this.value = Math.round(v / <number>config.round) * <number>config.round;
                    } else {
                        this.value = v;
                    }
                },
                config
            ),
        };
    },
    INT(node: ComfyNode, inputName: string, inputData: InputData[], app: ComfyApp) {
        return createIntWidget(node, inputName, inputData, app);
    },
    BOOLEAN(node: ComfyNode, inputName: string, inputData: InputData[]): { widget: ComfyWidget } {
        let defaultVal = false;
        let options: InputOptions = {};
        if (inputData[1]) {
            if (inputData[1].default) defaultVal = inputData[1].default;
            if (inputData[1].label_on) options['on'] = inputData[1].label_on;
            if (inputData[1].label_off) options['off'] = inputData[1].label_off;
        }
        return {
            widget: node.addWidget<ComfyWidget>('toggle', inputName, defaultVal, () => {}, options),
        };
    },
    STRING(
        node: ComfyNode,
        inputName: string,
        inputData: InputData[]
    ): {
        minWidth?: number;
        minHeight?: number;
        widget?: ComfyWidget;
    } {
        const defaultVal = inputData[1].default || '';
        const multiline = !!inputData[1].multiline;

        let res;
        if (multiline) {
            res = addMultilineWidget(node, inputName, { defaultVal, ...inputData[1] });
        } else {
            res = {
                widget: node.addWidget<ComfyWidget>('text', inputName, defaultVal, () => {}, {}),
            };
        }

        if (inputData[1].dynamicPrompts != undefined && res.widget?.dynamicPrompts) {
            res.widget.dynamicPrompts = inputData[1].dynamicPrompts;
        }

        return res;
    },
    COMBO(node: ComfyNode, inputName: string, inputData: InputData[]) {
        const type = inputData[0];
        let defaultValue = type[0];
        if (inputData[1] && inputData[1].default) {
            defaultValue = inputData[1].default;
        }
        const res = {
            widget: node.addWidget<ComfyWidget>('combo', inputName, defaultValue, () => {}, { values: type }),
        };
        if (inputData[1]?.control_after_generate) {
            res.widget.linkedWidgets = addValueControlWidgets(node, res.widget, undefined, undefined, inputData);
        }
        return res;
    },
    IMAGEUPLOAD(node: ComfyNode, inputName: string, inputData: InputData, app: ComfyApp) {
        const imageWidget = node.widgets.find(w => w.name === (inputData[1]?.widget ?? 'image'));
        let uploadWidget;

        function showImage(name: string) {
            const img = new Image();
            img.onload = () => {
                node.imgs = [img];
                app.graph?.setDirtyCanvas(true, false);
            };
            let folder_separator = name.lastIndexOf('/');
            let subfolder = '';
            if (folder_separator > -1) {
                subfolder = name.substring(0, folder_separator);
                name = name.substring(folder_separator + 1);
            }
            img.src = api.apiURL(
                `/view?filename=${encodeURIComponent(
                    name
                )}&type=input&subfolder=${subfolder}${app.getPreviewFormatParam()}${app.getRandParam()}`
            );
            node.setSizeForImage?.();
        }

        var default_value = imageWidget?.value;
        Object.defineProperty(imageWidget, 'value', {
            set: function (value) {
                this._real_value = value;
            },

            get: function () {
                let value: string | ComfyFile = '';
                if (this._real_value) {
                    value = this._real_value;
                } else {
                    return default_value;
                }

                if (typeof value !== 'string' && value.filename) {
                    let real_value = value;
                    value = '';
                    if (real_value.subfolder) {
                        value = real_value.subfolder + '/';
                    }

                    value += real_value.filename;

                    if (real_value.type && real_value.type !== 'input') value += ` [${real_value.type}]`;
                }
                return value;
            },
        });

        // Add our own callback to the combo widget to render an image when it changes
        const cb = node.callback;
        if (imageWidget) {
            imageWidget.callback = function (...args) {
                showImage(imageWidget.value);
                if (cb) {
                    return cb.apply(this, args);
                }
            };
        }

        // On load if we have a value then render the image
        // The value isnt set immediately so we need to wait a moment
        // No change callbacks seem to be fired on initial setting of the value
        requestAnimationFrame(() => {
            if (imageWidget?.value) {
                showImage(imageWidget.value);
            }
        });

        async function uploadFile(file: File, updateNode: boolean, pasted = false) {
            try {
                // Wrap file in formdata so it includes filename
                const body = new FormData();
                body.append('image', file);
                if (pasted) body.append('subfolder', 'pasted');
                const resp = await api.fetchApi('/upload/image', {
                    method: 'POST',
                    body,
                });

                if (resp.status === 200) {
                    const data = await resp.json();
                    // Add the file to the dropdown list and update the widget value
                    let path = data.name;
                    if (data.subfolder) path = data.subfolder + '/' + path;

                    if (imageWidget) {
                        if (!imageWidget.options.values.includes(path)) {
                            imageWidget.options.values.push(path);
                        }

                        if (updateNode) {
                            showImage(path);
                            imageWidget.value = path;
                        }
                    }
                } else {
                    alert(resp.status + ' - ' + resp.statusText);
                }
            } catch (error) {
                alert(error);
            }
        }

        const fileInput = document.createElement('input');
        Object.assign(fileInput, {
            type: 'file',
            accept: 'image/jpeg,image/png,image/webp',
            style: 'display: none',
            onchange: async () => {
                if (fileInput.files?.length) {
                    await uploadFile(fileInput.files[0], true);
                }
            },
        });
        document.body.append(fileInput);

        // Create the button widget for selecting the files
        uploadWidget = node.addWidget<ComfyWidget>('button', inputName, 'image', () => {
            fileInput.click();
        });
        uploadWidget.label = 'choose file to upload';
        uploadWidget.serialize = false;

        // Add handler to check if an image is being dragged over our node
        node.onDragOver = function (e: DragEvent) {
            if (e.dataTransfer && e.dataTransfer.items) {
                const image = [...e.dataTransfer.items].find(f => f.kind === 'file');
                return !!image;
            }

            return false;
        };

        // On drop upload files
        node.onDragDrop = function (e: DragEvent) {
            console.log('onDragDrop called');
            let handled = false;
            if (e.dataTransfer) {
                for (const file of e.dataTransfer.files) {
                    if (file.type.startsWith('image/')) {
                        uploadFile(file, !handled); // Dont await these, any order is fine, only update on first one
                        handled = true;
                    }
                }
            }

            return handled;
        };

        node.pasteFile = function (file: File) {
            if (file.type.startsWith('image/')) {
                const is_pasted = file.name === 'image.png' && file.lastModified - Date.now() < 2000;
                uploadFile(file, true, is_pasted);
                return true;
            }
            return false;
        };

        return { widget: uploadWidget };
    },
};

/** Utility Function */
export function getWidgetType(inputData: string | string[], inputName: string): string | null {
    const type = inputData[0];

    if (Array.isArray(type)) {
        return 'COMBO';
    } else if (widgetState.widgets && `${type}:${inputName}` in widgetState.widgets) {
        return `${type}:${inputName}`;
    } else if (widgetState.widgets && type in widgetState.widgets) {
        return type;
    } else {
        return null;
    }
}

/** Singleton state holder; holds the complete list of available widgets */
export class WidgetState {
    private static instance: WidgetState;
    widgets: IWidgetFactory = WidgetFactory;

    private constructor() {}
    static getInstance() {
        if (!WidgetState.instance) {
            WidgetState.instance = new WidgetState();
            WidgetState.instance.refresh().catch(err => console.error); // TO DO: not awaiting
        }
        return WidgetState.instance;
    }

    // TO DO: we should make this synchronous by forcing invokeExtensions to be a synchronous function instead
    async refresh() {
        // const customWidgetsResponses = await extensionManager.invokeExtensionsAsync('getCustomWidgets');
        // const customWidgetsMerged = customWidgetsResponses.reduce((acc, widgetResponse) => {
        //     return Object.assign(acc, widgetResponse);
        // }, {} as GetCustomWidgetResponse);
        //
        // this.widgets = {
        //     ...WidgetFactory,
        //     ...customWidgetsMerged,
        // };
    }
}

export const widgetState = WidgetState.getInstance();

/** Legacy name, for backwards compatability */
export { WidgetFactory as ComfyWidgets };
