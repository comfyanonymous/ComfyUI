import { app } from "../../scripts/app.js";
import { rgthree } from "./rgthree.js";
import { SERVICE as CONFIG_SERVICE } from "./services/config_service.js";
const SPECIAL_ENTRIES = [/^(CHOOSE|NONE|DISABLE|OPEN)(\s|$)/i, /^\p{Extended_Pictographic}/gu];
app.registerExtension({
    name: "rgthree.ContextMenuAutoNest",
    async setup() {
        const logger = rgthree.newLogSession("[ContextMenuAutoNest]");
        const existingContextMenu = LiteGraph.ContextMenu;
        LiteGraph.ContextMenu = function (values, options) {
            var _a, _b, _c, _d, _e, _f, _g, _h;
            const threshold = CONFIG_SERVICE.getConfigValue("features.menu_auto_nest.threshold", 20);
            const enabled = CONFIG_SERVICE.getConfigValue("features.menu_auto_nest.subdirs", false);
            let incompatible = !enabled || !!((_a = options === null || options === void 0 ? void 0 : options.extra) === null || _a === void 0 ? void 0 : _a.rgthree_doNotNest);
            if (!incompatible) {
                if (values.length <= threshold) {
                    incompatible = `Skipping context menu auto nesting b/c threshold is not met (${threshold})`;
                }
                if (!((_c = (_b = options.parentMenu) === null || _b === void 0 ? void 0 : _b.options) === null || _c === void 0 ? void 0 : _c.rgthree_originalCallback)) {
                    if (!(options === null || options === void 0 ? void 0 : options.callback)) {
                        incompatible = `Skipping context menu auto nesting b/c a callback was expected.`;
                    }
                    else if (values.some((i) => typeof i !== "string")) {
                        incompatible = `Skipping context menu auto nesting b/c not all values were strings.`;
                    }
                }
            }
            if (incompatible) {
                if (enabled) {
                    const [n, v] = logger.infoParts("Skipping context menu auto nesting for incompatible menu.");
                    (_d = console[n]) === null || _d === void 0 ? void 0 : _d.call(console, ...v);
                }
                return existingContextMenu.apply(this, [...arguments]);
            }
            const folders = {};
            const specialOps = [];
            const folderless = [];
            for (const value of values) {
                if (!value) {
                    folderless.push(value);
                    continue;
                }
                const newValue = typeof value === "string" ? { content: value } : Object.assign({}, value);
                newValue.rgthree_originalValue = value.rgthree_originalValue || value;
                const valueContent = newValue.content || "";
                const splitBy = valueContent.indexOf("/") > -1 ? "/" : "\\";
                const valueSplit = valueContent.split(splitBy);
                if (valueSplit.length > 1) {
                    const key = valueSplit.shift();
                    newValue.content = valueSplit.join(splitBy);
                    folders[key] = folders[key] || [];
                    folders[key].push(newValue);
                }
                else if (SPECIAL_ENTRIES.some((r) => r.test(valueContent))) {
                    specialOps.push(newValue);
                }
                else {
                    folderless.push(newValue);
                }
            }
            const foldersCount = Object.values(folders).length;
            if (foldersCount > 0) {
                options.rgthree_originalCallback =
                    options.rgthree_originalCallback ||
                        ((_f = (_e = options.parentMenu) === null || _e === void 0 ? void 0 : _e.options) === null || _f === void 0 ? void 0 : _f.rgthree_originalCallback) ||
                        options.callback;
                const oldCallback = options === null || options === void 0 ? void 0 : options.rgthree_originalCallback;
                options.callback = undefined;
                const newCallback = (item, options, event, parentMenu, node) => {
                    oldCallback === null || oldCallback === void 0 ? void 0 : oldCallback(item === null || item === void 0 ? void 0 : item.rgthree_originalValue, options, event, undefined, node);
                };
                const [n, v] = logger.infoParts(`Nested folders found (${foldersCount}).`);
                (_g = console[n]) === null || _g === void 0 ? void 0 : _g.call(console, ...v);
                const newValues = [];
                for (const [folderName, folderValues] of Object.entries(folders)) {
                    newValues.push({
                        content: `📁 ${folderName}`,
                        has_submenu: true,
                        callback: () => {
                        },
                        submenu: {
                            options: folderValues.map((value) => {
                                value.callback = newCallback;
                                return value;
                            }),
                        },
                    });
                }
                values = [].concat(specialOps.map((f) => {
                    if (typeof f === "string") {
                        f = { content: f };
                    }
                    f.callback = newCallback;
                    return f;
                }), newValues, folderless.map((f) => {
                    if (typeof f === "string") {
                        f = { content: f };
                    }
                    f.callback = newCallback;
                    return f;
                }));
            }
            if (options.scale == null) {
                options.scale = Math.max(((_h = app.canvas.ds) === null || _h === void 0 ? void 0 : _h.scale) || 1, 1);
            }
            const oldCtrResponse = existingContextMenu.call(this, values, options);
            if (oldCtrResponse === null || oldCtrResponse === void 0 ? void 0 : oldCtrResponse.constructor) {
                oldCtrResponse.constructor = LiteGraph.ContextMenu;
            }
            return this;
        };
    },
});
