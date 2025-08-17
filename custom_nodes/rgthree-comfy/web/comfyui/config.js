import { app } from "../../scripts/app.js";
import { RgthreeDialog } from "../../rgthree/common/dialog.js";
import { createElement as $el, queryAll as $$ } from "../../rgthree/common/utils_dom.js";
import { checkmark, logoRgthree } from "../../rgthree/common/media/svgs.js";
import { rgthree } from "./rgthree.js";
import { SERVICE as CONFIG_SERVICE } from "./services/config_service.js";
var ConfigType;
(function (ConfigType) {
    ConfigType[ConfigType["UNKNOWN"] = 0] = "UNKNOWN";
    ConfigType[ConfigType["BOOLEAN"] = 1] = "BOOLEAN";
    ConfigType[ConfigType["STRING"] = 2] = "STRING";
    ConfigType[ConfigType["NUMBER"] = 3] = "NUMBER";
    ConfigType[ConfigType["ARRAY"] = 4] = "ARRAY";
})(ConfigType || (ConfigType = {}));
var ConfigInputType;
(function (ConfigInputType) {
    ConfigInputType[ConfigInputType["UNKNOWN"] = 0] = "UNKNOWN";
    ConfigInputType[ConfigInputType["CHECKLIST"] = 1] = "CHECKLIST";
})(ConfigInputType || (ConfigInputType = {}));
const TYPE_TO_STRING = {
    [ConfigType.UNKNOWN]: "unknown",
    [ConfigType.BOOLEAN]: "boolean",
    [ConfigType.STRING]: "string",
    [ConfigType.NUMBER]: "number",
    [ConfigType.ARRAY]: "array",
};
const CONFIGURABLE = {
    features: [
        {
            key: "features.progress_bar.enabled",
            type: ConfigType.BOOLEAN,
            label: "Prompt Progress Bar",
            description: `Shows a minimal progress bar for nodes and steps at the top of the app.`,
            subconfig: [
                {
                    key: "features.progress_bar.height",
                    type: ConfigType.NUMBER,
                    label: "Height of the bar",
                },
                {
                    key: "features.progress_bar.position",
                    type: ConfigType.STRING,
                    label: "Position at top or bottom of window",
                    options: ["top", "bottom"],
                },
            ],
        },
        {
            key: "features.import_individual_nodes.enabled",
            type: ConfigType.BOOLEAN,
            label: "Import Individual Nodes Widgets",
            description: "Dragging & Dropping a similar image/JSON workflow onto (most) current workflow nodes" +
                "will allow you to import that workflow's node's widgets when it has the same " +
                "id and type. This is useful when you have several images and you'd like to import just " +
                "one part of a previous iteration, like a seed, or prompt.",
        },
    ],
    menus: [
        {
            key: "features.comfy_top_bar_menu.enabled",
            type: ConfigType.BOOLEAN,
            label: "Enable Top Bar Menu",
            description: "Have quick access from ComfyUI's new top bar to rgthree-comfy bookmarks, settings " +
                "(and more to come).",
        },
        {
            key: "features.menu_queue_selected_nodes",
            type: ConfigType.BOOLEAN,
            label: "Show 'Queue Selected Output Nodes'",
            description: "Will show a menu item in the right-click context menus to queue (only) the selected " +
                "output nodes.",
        },
        {
            key: "features.menu_auto_nest.subdirs",
            type: ConfigType.BOOLEAN,
            label: "Auto Nest Subdirectories in Menus",
            description: "When a large, flat list of values contain sub-directories, auto nest them. (Like, for " +
                "a large list of checkpoints).",
            subconfig: [
                {
                    key: "features.menu_auto_nest.threshold",
                    type: ConfigType.NUMBER,
                    label: "Number of items needed to trigger nesting.",
                },
            ],
        },
        {
            key: "features.menu_bookmarks.enabled",
            type: ConfigType.BOOLEAN,
            label: "Show Bookmarks in context menu",
            description: "Will list bookmarks in the rgthree-comfy right-click context menu.",
        },
    ],
    groups: [
        {
            key: "features.group_header_fast_toggle.enabled",
            type: ConfigType.BOOLEAN,
            label: "Show fast toggles in Group Headers",
            description: "Show quick toggles in Groups' Headers to quickly mute, bypass or queue.",
            subconfig: [
                {
                    key: "features.group_header_fast_toggle.toggles",
                    type: ConfigType.ARRAY,
                    label: "Which toggles to show.",
                    inputType: ConfigInputType.CHECKLIST,
                    options: [
                        { value: "queue", label: "queue" },
                        { value: "bypass", label: "bypass" },
                        { value: "mute", label: "mute" },
                    ],
                },
                {
                    key: "features.group_header_fast_toggle.show",
                    type: ConfigType.STRING,
                    label: "When to show them.",
                    options: [
                        { value: "hover", label: "on hover" },
                        { value: "always", label: "always" },
                    ],
                },
            ],
        },
    ],
    advanced: [
        {
            key: "features.show_alerts_for_corrupt_workflows",
            type: ConfigType.BOOLEAN,
            label: "Detect Corrupt Workflows",
            description: "Will show a message at the top of the screen when loading a workflow that has " +
                "corrupt linking data.",
        },
        {
            key: "log_level",
            type: ConfigType.STRING,
            label: "Log level for browser dev console.",
            description: "Further down the list, the more verbose logs to the console will be. For instance, " +
                "selecting 'IMPORTANT' means only important message will be logged to the browser " +
                "console, while selecting 'WARN' will log all messages at or higher than WARN, including " +
                "'ERROR' and 'IMPORTANT' etc.",
            options: ["IMPORTANT", "ERROR", "WARN", "INFO", "DEBUG", "DEV"],
            isDevOnly: true,
            onSave: function (value) {
                rgthree.setLogLevel(value);
            },
        },
        {
            key: "features.invoke_extensions_async.node_created",
            type: ConfigType.BOOLEAN,
            label: "Allow other extensions to call nodeCreated on rgthree-nodes.",
            isDevOnly: true,
            description: "Do not disable unless you are having trouble (and then file an issue at rgthree-comfy)." +
                "Prior to Apr 2024 it was not possible for other extensions to invoke their nodeCreated " +
                "event on some rgthree-comfy nodes. Now it's possible and this option is only here in " +
                "for easy if something is wrong.",
        },
    ],
};
function fieldrow(item) {
    var _a;
    const initialValue = CONFIG_SERVICE.getConfigValue(item.key);
    const container = $el(`div.fieldrow.-type-${TYPE_TO_STRING[item.type]}`, {
        dataset: {
            name: item.key,
            initial: initialValue,
            type: item.type,
        },
    });
    $el(`label[for="${item.key}"]`, {
        children: [
            $el(`span[text="${item.label}"]`),
            item.description ? $el("small", { html: item.description }) : null,
        ],
        parent: container,
    });
    let input;
    if ((_a = item.options) === null || _a === void 0 ? void 0 : _a.length) {
        if (item.inputType === ConfigInputType.CHECKLIST) {
            const initialValueList = initialValue || [];
            input = $el(`fieldset.rgthree-checklist-group[id="${item.key}"]`, {
                parent: container,
                children: item.options.map((o) => {
                    const label = o.label || String(o);
                    const value = o.value || o;
                    const id = `${item.key}_${value}`;
                    return $el(`span.rgthree-checklist-item`, {
                        children: [
                            $el(`input[type="checkbox"][value="${value}"]`, {
                                id,
                                checked: initialValueList.includes(value),
                            }),
                            $el(`label`, {
                                for: id,
                                text: label,
                            })
                        ]
                    });
                }),
            });
        }
        else {
            input = $el(`select[id="${item.key}"]`, {
                parent: container,
                children: item.options.map((o) => {
                    const label = o.label || String(o);
                    const value = o.value || o;
                    const valueSerialized = JSON.stringify({ value: value });
                    return $el(`option[value="${valueSerialized}"]`, {
                        text: label,
                        selected: valueSerialized === JSON.stringify({ value: initialValue }),
                    });
                }),
            });
        }
    }
    else if (item.type === ConfigType.BOOLEAN) {
        container.classList.toggle("-checked", !!initialValue);
        input = $el(`input[type="checkbox"][id="${item.key}"]`, {
            parent: container,
            checked: initialValue,
        });
    }
    else {
        input = $el(`input[id="${item.key}"]`, {
            parent: container,
            value: initialValue,
        });
    }
    $el("div.fieldrow-value", { children: [input], parent: container });
    return container;
}
export class RgthreeConfigDialog extends RgthreeDialog {
    constructor() {
        const content = $el("div");
        content.appendChild(RgthreeConfigDialog.buildFieldset(CONFIGURABLE["features"], "Features"));
        content.appendChild(RgthreeConfigDialog.buildFieldset(CONFIGURABLE["menus"], "Menus"));
        content.appendChild(RgthreeConfigDialog.buildFieldset(CONFIGURABLE["groups"], "Groups"));
        content.appendChild(RgthreeConfigDialog.buildFieldset(CONFIGURABLE["advanced"], "Advanced"));
        content.addEventListener("input", (e) => {
            const changed = this.getChangedFormData();
            $$(".save-button", this.element)[0].disabled =
                !Object.keys(changed).length;
        });
        content.addEventListener("change", (e) => {
            const changed = this.getChangedFormData();
            $$(".save-button", this.element)[0].disabled =
                !Object.keys(changed).length;
        });
        const dialogOptions = {
            class: "-iconed -settings",
            title: logoRgthree + `<h2>Settings - rgthree-comfy</h2>`,
            content,
            onBeforeClose: () => {
                const changed = this.getChangedFormData();
                if (Object.keys(changed).length) {
                    return confirm("Looks like there are unsaved changes. Are you sure you want close?");
                }
                return true;
            },
            buttons: [
                {
                    label: "Save",
                    disabled: true,
                    className: "rgthree-button save-button -blue",
                    callback: async (e) => {
                        var _a, _b;
                        const changed = this.getChangedFormData();
                        if (!Object.keys(changed).length) {
                            this.close();
                            return;
                        }
                        const success = await CONFIG_SERVICE.setConfigValues(changed);
                        if (success) {
                            for (const key of Object.keys(changed)) {
                                (_b = (_a = Object.values(CONFIGURABLE)
                                    .flat()
                                    .find((f) => f.key === key)) === null || _a === void 0 ? void 0 : _a.onSave) === null || _b === void 0 ? void 0 : _b.call(_a, changed[key]);
                            }
                            this.close();
                            rgthree.showMessage({
                                id: "config-success",
                                message: `${checkmark} Successfully saved rgthree-comfy settings!`,
                                timeout: 4000,
                            });
                            $$(".save-button", this.element)[0].disabled = true;
                        }
                        else {
                            alert("There was an error saving rgthree-comfy configuration.");
                        }
                    },
                },
            ],
        };
        super(dialogOptions);
    }
    static buildFieldset(datas, label) {
        const fieldset = $el(`fieldset`, { children: [$el(`legend[text="${label}"]`)] });
        for (const data of datas) {
            if (data.isDevOnly && !rgthree.isDevMode()) {
                continue;
            }
            const container = $el("div.formrow");
            container.appendChild(fieldrow(data));
            if (data.subconfig) {
                for (const subfeature of data.subconfig) {
                    container.appendChild(fieldrow(subfeature));
                }
            }
            fieldset.appendChild(container);
        }
        return fieldset;
    }
    getChangedFormData() {
        return $$("[data-name]", this.contentElement).reduce((acc, el) => {
            const name = el.dataset["name"];
            const type = el.dataset["type"];
            const initialValue = CONFIG_SERVICE.getConfigValue(name);
            let currentValueEl = $$("fieldset.rgthree-checklist-group, input, textarea, select", el)[0];
            let currentValue = null;
            if (type === String(ConfigType.BOOLEAN)) {
                currentValue = currentValueEl.checked;
                el.classList.toggle("-checked", currentValue);
            }
            else {
                currentValue = currentValueEl === null || currentValueEl === void 0 ? void 0 : currentValueEl.value;
                if (currentValueEl.nodeName === "SELECT") {
                    currentValue = JSON.parse(currentValue).value;
                }
                else if (currentValueEl.classList.contains('rgthree-checklist-group')) {
                    currentValue = [];
                    for (const check of $$('input[type="checkbox"]', currentValueEl)) {
                        if (check.checked) {
                            currentValue.push(check.value);
                        }
                    }
                }
                else if (type === String(ConfigType.NUMBER)) {
                    currentValue = Number(currentValue) || initialValue;
                }
            }
            if (JSON.stringify(currentValue) !== JSON.stringify(initialValue)) {
                acc[name] = currentValue;
            }
            return acc;
        }, {});
    }
}
app.ui.settings.addSetting({
    id: "rgthree.config",
    defaultValue: null,
    name: "Open rgthree-comfy config",
    type: () => {
        return $el("tr.rgthree-comfyui-settings-row", {
            children: [
                $el("td", {
                    child: `<div>${logoRgthree} [rgthree-comfy] configuration / settings</div>`,
                }),
                $el("td", {
                    child: $el('button.rgthree-button.-blue[text="rgthree-comfy settings"]', {
                        events: {
                            click: (e) => {
                                new RgthreeConfigDialog().show();
                            },
                        },
                    }),
                }),
            ],
        });
    },
});
