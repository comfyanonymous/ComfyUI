import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { SERVICE as CONFIG_SERVICE } from "./services/config_service.js";
import { SERVICE as BOOKMARKS_SERVICE } from "./services/bookmarks_services.js";
import { SERVICE as KEY_EVENT_SERVICE } from "./services/key_events_services.js";
import { WorkflowLinkFixer } from "../../rgthree/common/link_fixer.js";
import { injectCss, wait } from "../../rgthree/common/shared_utils.js";
import { replaceNode, waitForCanvas, waitForGraph } from "./utils.js";
import { NodeTypesString, addRgthree, getNodeTypeStrings } from "./constants.js";
import { RgthreeProgressBar } from "../../rgthree/common/progress_bar.js";
import { RgthreeConfigDialog } from "./config.js";
import { iconGear, iconNode, iconReplace, iconStarFilled, logoRgthree, } from "../../rgthree/common/media/svgs.js";
import { createElement, queryAll, query } from "../../rgthree/common/utils_dom.js";
export var LogLevel;
(function (LogLevel) {
    LogLevel[LogLevel["IMPORTANT"] = 1] = "IMPORTANT";
    LogLevel[LogLevel["ERROR"] = 2] = "ERROR";
    LogLevel[LogLevel["WARN"] = 3] = "WARN";
    LogLevel[LogLevel["INFO"] = 4] = "INFO";
    LogLevel[LogLevel["DEBUG"] = 5] = "DEBUG";
    LogLevel[LogLevel["DEV"] = 6] = "DEV";
})(LogLevel || (LogLevel = {}));
const LogLevelKeyToLogLevel = {
    IMPORTANT: LogLevel.IMPORTANT,
    ERROR: LogLevel.ERROR,
    WARN: LogLevel.WARN,
    INFO: LogLevel.INFO,
    DEBUG: LogLevel.DEBUG,
    DEV: LogLevel.DEV,
};
const LogLevelToMethod = {
    [LogLevel.IMPORTANT]: "log",
    [LogLevel.ERROR]: "error",
    [LogLevel.WARN]: "warn",
    [LogLevel.INFO]: "info",
    [LogLevel.DEBUG]: "log",
    [LogLevel.DEV]: "log",
};
const LogLevelToCSS = {
    [LogLevel.IMPORTANT]: "font-weight: bold; color: blue;",
    [LogLevel.ERROR]: "",
    [LogLevel.WARN]: "",
    [LogLevel.INFO]: "font-style: italic; color: blue;",
    [LogLevel.DEBUG]: "font-style: italic; color: #444;",
    [LogLevel.DEV]: "color: #004b68;",
};
let GLOBAL_LOG_LEVEL = LogLevel.ERROR;
const apiURL = api.apiURL;
api.apiURL = function (route) {
    if (route.includes("rgthree/")) {
        return (this.api_base + "/" + route).replace(/\/\//g, "/");
    }
    return apiURL.apply(this, arguments);
};
const INVOKE_EXTENSIONS_BLOCKLIST = [
    {
        name: "Comfy.WidgetInputs",
        reason: "Major conflict with rgthree-comfy nodes' inputs causing instability and " +
            "repeated link disconnections.",
    },
    {
        name: "efficiency.widgethider",
        reason: "Overrides value getter before widget getter is prepared. Can be lifted if/when " +
            "https://github.com/jags111/efficiency-nodes-comfyui/pull/203 is pulled.",
    },
];
class Logger {
    log(level, message, ...args) {
        var _a;
        const [n, v] = this.logParts(level, message, ...args);
        (_a = console[n]) === null || _a === void 0 ? void 0 : _a.call(console, ...v);
    }
    logParts(level, message, ...args) {
        if (level <= GLOBAL_LOG_LEVEL) {
            const css = LogLevelToCSS[level] || "";
            if (level === LogLevel.DEV) {
                message = `🔧 ${message}`;
            }
            return [LogLevelToMethod[level], [`%c${message}`, css, ...args]];
        }
        return ["none", []];
    }
}
class LogSession {
    constructor(name) {
        this.name = name;
        this.logger = new Logger();
        this.logsCache = {};
    }
    logParts(level, message, ...args) {
        message = `${this.name || ""}${message ? " " + message : ""}`;
        return this.logger.logParts(level, message, ...args);
    }
    logPartsOnceForTime(level, time, message, ...args) {
        message = `${this.name || ""}${message ? " " + message : ""}`;
        const cacheKey = `${level}:${message}`;
        const cacheEntry = this.logsCache[cacheKey];
        const now = +new Date();
        if (cacheEntry && cacheEntry.lastShownTime + time > now) {
            return ["none", []];
        }
        const parts = this.logger.logParts(level, message, ...args);
        if (console[parts[0]]) {
            this.logsCache[cacheKey] = this.logsCache[cacheKey] || {};
            this.logsCache[cacheKey].lastShownTime = now;
        }
        return parts;
    }
    debugParts(message, ...args) {
        return this.logParts(LogLevel.DEBUG, message, ...args);
    }
    infoParts(message, ...args) {
        return this.logParts(LogLevel.INFO, message, ...args);
    }
    warnParts(message, ...args) {
        return this.logParts(LogLevel.WARN, message, ...args);
    }
    newSession(name) {
        return new LogSession(`${this.name}${name}`);
    }
}
class Rgthree extends EventTarget {
    constructor() {
        var _a, _b, _c, _d;
        super();
        this.api = api;
        this.settingsDialog = null;
        this.progressBarEl = null;
        this.queueNodeIds = null;
        this.version = CONFIG_SERVICE.getConfigValue("version");
        this.logger = new LogSession("[rgthree]");
        this.monitorBadLinksAlerted = false;
        this.monitorLinkTimeout = null;
        this.processingQueue = false;
        this.loadingApiJson = false;
        this.replacingReroute = null;
        this.processingMouseDown = false;
        this.processingMouseUp = false;
        this.processingMouseMove = false;
        this.lastCanvasMouseEvent = null;
        this.canvasCurrentlyCopyingToClipboard = false;
        this.canvasCurrentlyCopyingToClipboardWithMultipleNodes = false;
        this.initialGraphToPromptSerializedWorkflowBecauseComfyUIBrokeStuff = null;
        this.isMac = !!(((_a = navigator.platform) === null || _a === void 0 ? void 0 : _a.toLocaleUpperCase().startsWith("MAC")) ||
            ((_c = (_b = navigator.userAgentData) === null || _b === void 0 ? void 0 : _b.platform) === null || _c === void 0 ? void 0 : _c.toLocaleUpperCase().startsWith("MAC")));
        const logLevel = (_d = LogLevelKeyToLogLevel[CONFIG_SERVICE.getConfigValue("log_level")]) !== null && _d !== void 0 ? _d : GLOBAL_LOG_LEVEL;
        this.setLogLevel(logLevel);
        this.initializeGraphAndCanvasHooks();
        this.initializeComfyUIHooks();
        this.initializeContextMenu();
        this.rgthreeCssPromise = injectCss("extensions/rgthree-comfy/rgthree.css");
        this.initializeProgressBar();
        CONFIG_SERVICE.addEventListener("config-change", ((e) => {
            var _a, _b;
            if ((_b = (_a = e.detail) === null || _a === void 0 ? void 0 : _a.key) === null || _b === void 0 ? void 0 : _b.includes("features.progress_bar")) {
                this.initializeProgressBar();
            }
        }));
        if (CONFIG_SERVICE.getConfigValue("debug.keys_down.enabled")) {
            const elDebugKeydowns = createElement("div.rgthree-debug-keydowns", {
                parent: document.body,
            });
            const updateDebugKeyDown = () => {
                elDebugKeydowns.innerText = Object.keys(KEY_EVENT_SERVICE.downKeys).join(" ");
            };
            KEY_EVENT_SERVICE.addEventListener("keydown", updateDebugKeyDown);
            KEY_EVENT_SERVICE.addEventListener("keyup", updateDebugKeyDown);
        }
    }
    async initializeProgressBar() {
        var _a;
        if (CONFIG_SERVICE.getConfigValue("features.progress_bar.enabled")) {
            await this.rgthreeCssPromise;
            if (!this.progressBarEl) {
                this.progressBarEl = RgthreeProgressBar.create();
                this.progressBarEl.setAttribute("title", "Progress Bar by rgthree. right-click for rgthree menu.");
                this.progressBarEl.addEventListener("contextmenu", async (e) => {
                    e.stopPropagation();
                    e.preventDefault();
                });
                this.progressBarEl.addEventListener("pointerdown", async (e) => {
                    var _a;
                    LiteGraph.closeAllContextMenus();
                    if (e.button == 2) {
                        const canvas = await waitForCanvas();
                        new LiteGraph.ContextMenu(this.getRgthreeIContextMenuValues(), {
                            title: `<div class="rgthree-contextmenu-item rgthree-contextmenu-title-rgthree-comfy">${logoRgthree} rgthree-comfy</div>`,
                            left: e.clientX,
                            top: 5,
                        });
                        return;
                    }
                    if (e.button == 0) {
                        const nodeId = (_a = this.progressBarEl) === null || _a === void 0 ? void 0 : _a.currentNodeId;
                        if (nodeId) {
                            const [canvas, graph] = await Promise.all([waitForCanvas(), waitForGraph()]);
                            const node = graph.getNodeById(Number(nodeId));
                            if (node) {
                                canvas.centerOnNode(node);
                                e.stopPropagation();
                                e.preventDefault();
                            }
                        }
                        return;
                    }
                });
            }
            const isUpdatedComfyBodyClasses = !!query(".comfyui-body-top");
            const position = CONFIG_SERVICE.getConfigValue("features.progress_bar.position");
            this.progressBarEl.classList.toggle("rgthree-pos-bottom", position === "bottom");
            if (isUpdatedComfyBodyClasses) {
                if (position === "bottom") {
                    query(".comfyui-body-bottom").appendChild(this.progressBarEl);
                }
                else {
                    query(".comfyui-body-top").appendChild(this.progressBarEl);
                }
            }
            else {
                document.body.appendChild(this.progressBarEl);
            }
            const height = CONFIG_SERVICE.getConfigValue("features.progress_bar.height") || 14;
            this.progressBarEl.style.height = `${height}px`;
            const fontSize = Math.max(10, Number(height) - 10);
            this.progressBarEl.style.fontSize = `${fontSize}px`;
            this.progressBarEl.style.fontWeight = fontSize <= 12 ? "bold" : "normal";
        }
        else {
            (_a = this.progressBarEl) === null || _a === void 0 ? void 0 : _a.remove();
        }
    }
    async initializeGraphAndCanvasHooks() {
        const rgthree = this;
        const graphSerialize = LGraph.prototype.serialize;
        LGraph.prototype.serialize = function () {
            const response = graphSerialize.apply(this, [...arguments]);
            rgthree.initialGraphToPromptSerializedWorkflowBecauseComfyUIBrokeStuff = response;
            return response;
        };
        const processMouseDown = LGraphCanvas.prototype.processMouseDown;
        LGraphCanvas.prototype.processMouseDown = function (e) {
            rgthree.processingMouseDown = true;
            const returnVal = processMouseDown.apply(this, [...arguments]);
            rgthree.dispatchCustomEvent("on-process-mouse-down", { originalEvent: e });
            rgthree.processingMouseDown = false;
            return returnVal;
        };
        const adjustMouseEvent = LGraphCanvas.prototype.adjustMouseEvent;
        LGraphCanvas.prototype.adjustMouseEvent = function (e) {
            adjustMouseEvent.apply(this, [...arguments]);
            rgthree.lastCanvasMouseEvent = e;
        };
        const copyToClipboard = LGraphCanvas.prototype.copyToClipboard;
        LGraphCanvas.prototype.copyToClipboard = function (nodes) {
            rgthree.canvasCurrentlyCopyingToClipboard = true;
            rgthree.canvasCurrentlyCopyingToClipboardWithMultipleNodes =
                Object.values(nodes || this.selected_nodes || []).length > 1;
            copyToClipboard.apply(this, [...arguments]);
            rgthree.canvasCurrentlyCopyingToClipboard = false;
            rgthree.canvasCurrentlyCopyingToClipboardWithMultipleNodes = false;
        };
        const onGroupAdd = LGraphCanvas.onGroupAdd;
        LGraphCanvas.onGroupAdd = function (...args) {
            const graph = app.canvas.getCurrentGraph();
            onGroupAdd.apply(this, [...args]);
            LGraphCanvas.onShowPropertyEditor({}, null, null, null, graph._groups[graph._groups.length - 1]);
        };
    }
    async invokeExtensionsAsync(method, ...args) {
        var _a;
        const comfyapp = app;
        if (CONFIG_SERVICE.getConfigValue("features.invoke_extensions_async.node_created") === false) {
            const [m, a] = this.logParts(LogLevel.INFO, `Skipping invokeExtensionsAsync for applicable rgthree-comfy nodes`);
            (_a = console[m]) === null || _a === void 0 ? void 0 : _a.call(console, ...a);
            return Promise.resolve();
        }
        return await Promise.all(comfyapp.extensions.map(async (ext) => {
            var _a, _b;
            if (ext === null || ext === void 0 ? void 0 : ext[method]) {
                try {
                    const blocked = INVOKE_EXTENSIONS_BLOCKLIST.find((block) => ext.name.toLowerCase().startsWith(block.name.toLowerCase()));
                    if (blocked) {
                        const [n, v] = this.logger.logPartsOnceForTime(LogLevel.WARN, 5000, `Blocked extension '${ext.name}' method '${method}' for rgthree-nodes because: ${blocked.reason}`);
                        (_a = console[n]) === null || _a === void 0 ? void 0 : _a.call(console, ...v);
                        return Promise.resolve();
                    }
                    return await ext[method](...args, comfyapp);
                }
                catch (error) {
                    const [n, v] = this.logParts(LogLevel.ERROR, `Error calling extension '${ext.name}' method '${method}' for rgthree-node.`, { error }, { extension: ext }, { args });
                    (_b = console[n]) === null || _b === void 0 ? void 0 : _b.call(console, ...v);
                }
            }
        }));
    }
    dispatchCustomEvent(event, detail) {
        if (detail != null) {
            return this.dispatchEvent(new CustomEvent(event, { detail }));
        }
        return this.dispatchEvent(new CustomEvent(event));
    }
    async initializeContextMenu() {
        const that = this;
        setTimeout(async () => {
            const getCanvasMenuOptions = LGraphCanvas.prototype.getCanvasMenuOptions;
            LGraphCanvas.prototype.getCanvasMenuOptions = function (...args) {
                let existingOptions = getCanvasMenuOptions.apply(this, [...args]);
                const options = [];
                options.push(null);
                options.push(null);
                options.push(null);
                options.push({
                    content: logoRgthree + `rgthree-comfy`,
                    className: "rgthree-contextmenu-item rgthree-contextmenu-main-item-rgthree-comfy",
                    submenu: {
                        options: that.getRgthreeIContextMenuValues(),
                    },
                });
                options.push(null);
                options.push(null);
                let idx = null;
                idx = idx || existingOptions.findIndex((o) => { var _a, _b; return (_b = (_a = o === null || o === void 0 ? void 0 : o.content) === null || _a === void 0 ? void 0 : _a.startsWith) === null || _b === void 0 ? void 0 : _b.call(_a, "Queue Group"); }) + 1;
                idx =
                    idx || existingOptions.findIndex((o) => { var _a, _b; return (_b = (_a = o === null || o === void 0 ? void 0 : o.content) === null || _a === void 0 ? void 0 : _a.startsWith) === null || _b === void 0 ? void 0 : _b.call(_a, "Queue Selected"); }) + 1;
                idx = idx || existingOptions.findIndex((o) => { var _a, _b; return (_b = (_a = o === null || o === void 0 ? void 0 : o.content) === null || _a === void 0 ? void 0 : _a.startsWith) === null || _b === void 0 ? void 0 : _b.call(_a, "Convert to Group"); });
                idx = idx || existingOptions.findIndex((o) => { var _a, _b; return (_b = (_a = o === null || o === void 0 ? void 0 : o.content) === null || _a === void 0 ? void 0 : _a.startsWith) === null || _b === void 0 ? void 0 : _b.call(_a, "Arrange ("); });
                idx = idx || existingOptions.findIndex((o) => !o) + 1;
                idx = idx || 3;
                existingOptions.splice(idx, 0, ...options);
                for (let i = existingOptions.length; i > 0; i--) {
                    if (existingOptions[i] === null && existingOptions[i + 1] === null) {
                        existingOptions.splice(i, 1);
                    }
                }
                return existingOptions;
            };
        }, 1016);
    }
    getRgthreeIContextMenuValues() {
        const [canvas, graph] = [app.canvas, app.canvas.getCurrentGraph()];
        const selectedNodes = Object.values(canvas.selected_nodes || {});
        let rerouteNodes = [];
        if (selectedNodes.length) {
            rerouteNodes = selectedNodes.filter((n) => n.type === "Reroute");
        }
        else {
            rerouteNodes = graph._nodes.filter((n) => n.type == "Reroute");
        }
        const rerouteLabel = selectedNodes.length ? "selected" : "all";
        const showBookmarks = CONFIG_SERVICE.getFeatureValue("menu_bookmarks.enabled");
        const bookmarkMenuItems = showBookmarks ? getBookmarks() : [];
        return [
            {
                content: "Nodes",
                disabled: true,
                className: "rgthree-contextmenu-item rgthree-contextmenu-label",
            },
            {
                content: iconNode + "All",
                className: "rgthree-contextmenu-item",
                has_submenu: true,
                submenu: {
                    options: getNodeTypeStrings(),
                    callback: (value, options, event) => {
                        const node = LiteGraph.createNode(addRgthree(value));
                        if (node) {
                            node.pos = [
                                rgthree.lastCanvasMouseEvent.canvasX,
                                rgthree.lastCanvasMouseEvent.canvasY,
                            ];
                            canvas.graph.add(node);
                            canvas.selectNode(node);
                            graph.setDirtyCanvas(true, true);
                        }
                    },
                    extra: { rgthree_doNotNest: true },
                },
            },
            {
                content: "Actions",
                disabled: true,
                className: "rgthree-contextmenu-item rgthree-contextmenu-label",
            },
            {
                content: iconGear + "Settings (rgthree-comfy)",
                disabled: !!this.settingsDialog,
                className: "rgthree-contextmenu-item",
                callback: (...args) => {
                    this.settingsDialog = new RgthreeConfigDialog().show();
                    this.settingsDialog.addEventListener("close", (e) => {
                        this.settingsDialog = null;
                    });
                },
            },
            {
                content: iconReplace + ` Convert ${rerouteLabel} Reroutes`,
                disabled: !rerouteNodes.length,
                className: "rgthree-contextmenu-item",
                callback: (...args) => {
                    const msg = `Convert ${rerouteLabel} ComfyUI Reroutes to Reroute (rgthree) nodes? \n` +
                        `(First save a copy of your workflow & check reroute connections afterwards)`;
                    if (!window.confirm(msg)) {
                        return;
                    }
                    (async () => {
                        for (const node of [...rerouteNodes]) {
                            if (node.type == "Reroute") {
                                this.replacingReroute = node.id;
                                await replaceNode(node, NodeTypesString.REROUTE);
                                this.replacingReroute = null;
                            }
                        }
                    })();
                },
            },
            ...bookmarkMenuItems,
            {
                content: "More...",
                disabled: true,
                className: "rgthree-contextmenu-item rgthree-contextmenu-label",
            },
            {
                content: iconStarFilled + "Star on Github",
                className: "rgthree-contextmenu-item rgthree-contextmenu-github",
                callback: (...args) => {
                    window.open("https://github.com/rgthree/rgthree-comfy", "_blank");
                },
            },
        ];
    }
    async queueOutputNodes(nodeIds) {
        var _a;
        try {
            this.queueNodeIds = nodeIds;
            await app.queuePrompt(0);
        }
        catch (e) {
            const [n, v] = this.logParts(LogLevel.ERROR, `There was an error queuing nodes ${nodeIds}`, e);
            (_a = console[n]) === null || _a === void 0 ? void 0 : _a.call(console, ...v);
        }
        finally {
            this.queueNodeIds = null;
        }
    }
    recursiveAddNodes(nodeId, oldOutput, newOutput) {
        let currentId = nodeId;
        let currentNode = oldOutput[currentId];
        if (newOutput[currentId] == null) {
            newOutput[currentId] = currentNode;
            for (const inputValue of Object.values(currentNode.inputs || [])) {
                if (Array.isArray(inputValue)) {
                    this.recursiveAddNodes(inputValue[0], oldOutput, newOutput);
                }
            }
        }
        return newOutput;
    }
    initializeComfyUIHooks() {
        const rgthree = this;
        const queuePrompt = app.queuePrompt;
        app.queuePrompt = async function (number, batchCount) {
            rgthree.processingQueue = true;
            rgthree.dispatchCustomEvent("queue");
            try {
                return await queuePrompt.apply(app, [...arguments]);
            }
            finally {
                rgthree.processingQueue = false;
                rgthree.dispatchCustomEvent("queue-end");
            }
        };
        const loadApiJson = app.loadApiJson;
        app.loadApiJson = async function () {
            rgthree.loadingApiJson = true;
            try {
                loadApiJson.apply(app, [...arguments]);
            }
            finally {
                rgthree.loadingApiJson = false;
            }
        };
        const graphToPrompt = app.graphToPrompt;
        app.graphToPrompt = async function () {
            rgthree.dispatchCustomEvent("graph-to-prompt");
            let promise = graphToPrompt.apply(app, [...arguments]);
            await promise;
            rgthree.dispatchCustomEvent("graph-to-prompt-end");
            return promise;
        };
        const apiQueuePrompt = api.queuePrompt;
        api.queuePrompt = async function (index, prompt) {
            var _a;
            if (((_a = rgthree.queueNodeIds) === null || _a === void 0 ? void 0 : _a.length) && prompt.output) {
                const oldOutput = prompt.output;
                let newOutput = {};
                for (const queueNodeId of rgthree.queueNodeIds) {
                    rgthree.recursiveAddNodes(String(queueNodeId), oldOutput, newOutput);
                }
                prompt.output = newOutput;
            }
            rgthree.dispatchCustomEvent("comfy-api-queue-prompt-before", {
                workflow: prompt.workflow,
                output: prompt.output,
            });
            const response = apiQueuePrompt.apply(app, [index, prompt]);
            rgthree.dispatchCustomEvent("comfy-api-queue-prompt-end");
            return response;
        };
        const clean = app.clean;
        app.clean = function () {
            rgthree.clearAllMessages();
            clean && clean.apply(app, [...arguments]);
        };
        const loadGraphData = app.loadGraphData;
        app.loadGraphData = function (graph) {
            if (rgthree.monitorLinkTimeout) {
                clearTimeout(rgthree.monitorLinkTimeout);
                rgthree.monitorLinkTimeout = null;
            }
            rgthree.clearAllMessages();
            let graphCopy;
            try {
                graphCopy = JSON.parse(JSON.stringify(graph));
            }
            catch (e) {
                graphCopy = null;
            }
            setTimeout(() => {
                var _a, _b, _c;
                const wasLoadingAborted = (_b = (_a = document
                    .querySelector(".comfy-modal-content")) === null || _a === void 0 ? void 0 : _a.textContent) === null || _b === void 0 ? void 0 : _b.includes("Loading aborted due");
                const graphToUse = wasLoadingAborted ? graphCopy || graph : app.graph;
                const fixer = WorkflowLinkFixer.create(graphToUse);
                const fixBadLinksResult = fixer.check();
                if (fixBadLinksResult.hasBadLinks) {
                    const [n, v] = rgthree.logParts(LogLevel.WARN, `The workflow you've loaded has corrupt linking data. Open ${new URL(location.href).origin}/rgthree/link_fixer to try to fix.`);
                    (_c = console[n]) === null || _c === void 0 ? void 0 : _c.call(console, ...v);
                    if (CONFIG_SERVICE.getConfigValue("features.show_alerts_for_corrupt_workflows")) {
                        rgthree.showMessage({
                            id: "bad-links",
                            type: "warn",
                            message: "The workflow you've loaded has corrupt linking data that may be able to be fixed.",
                            actions: [
                                {
                                    label: "Open fixer",
                                    href: "/rgthree/link_fixer",
                                },
                                {
                                    label: "Fix in place",
                                    href: "/rgthree/link_fixer",
                                    callback: (event) => {
                                        event.stopPropagation();
                                        event.preventDefault();
                                        if (confirm("This will attempt to fix in place. Please make sure to have a saved copy of your workflow.")) {
                                            try {
                                                const fixBadLinksResult = fixer.fix();
                                                if (!fixBadLinksResult.hasBadLinks) {
                                                    rgthree.hideMessage("bad-links");
                                                    alert("Success! It's possible some valid links may have been affected. Please check and verify your workflow.");
                                                    wasLoadingAborted && app.loadGraphData(fixBadLinksResult.graph);
                                                    if (CONFIG_SERVICE.getConfigValue("features.monitor_for_corrupt_links") ||
                                                        CONFIG_SERVICE.getConfigValue("features.monitor_bad_links")) {
                                                        rgthree.monitorLinkTimeout = setTimeout(() => {
                                                            rgthree.monitorBadLinks();
                                                        }, 5000);
                                                    }
                                                }
                                            }
                                            catch (e) {
                                                console.error(e);
                                                alert("Unsuccessful at fixing corrupt data. :(");
                                                rgthree.hideMessage("bad-links");
                                            }
                                        }
                                    },
                                },
                            ],
                        });
                    }
                }
                else if (CONFIG_SERVICE.getConfigValue("features.monitor_for_corrupt_links") ||
                    CONFIG_SERVICE.getConfigValue("features.monitor_bad_links")) {
                    rgthree.monitorLinkTimeout = setTimeout(() => {
                        rgthree.monitorBadLinks();
                    }, 5000);
                }
            }, 100);
            return loadGraphData && loadGraphData.apply(app, [...arguments]);
        };
    }
    getNodeFromInitialGraphToPromptSerializedWorkflowBecauseComfyUIBrokeStuff(node) {
        var _a, _b, _c;
        return ((_c = (_b = (_a = this.initialGraphToPromptSerializedWorkflowBecauseComfyUIBrokeStuff) === null || _a === void 0 ? void 0 : _a.nodes) === null || _b === void 0 ? void 0 : _b.find((n) => n.id === node.id)) !== null && _c !== void 0 ? _c : null);
    }
    async showMessage(data) {
        let container = document.querySelector(".rgthree-top-messages-container");
        if (!container) {
            container = document.createElement("div");
            container.classList.add("rgthree-top-messages-container");
            document.body.appendChild(container);
        }
        const dialogs = queryAll("dialog[open]");
        if (dialogs.length) {
            let dialog = dialogs[dialogs.length - 1];
            dialog.appendChild(container);
            dialog.addEventListener("close", (e) => {
                document.body.appendChild(container);
            });
        }
        await this.hideMessage(data.id);
        const messageContainer = document.createElement("div");
        messageContainer.setAttribute("type", data.type || "info");
        const message = document.createElement("span");
        message.innerHTML = data.message;
        messageContainer.appendChild(message);
        for (let a = 0; a < (data.actions || []).length; a++) {
            const action = data.actions[a];
            if (a > 0) {
                const sep = document.createElement("span");
                sep.innerHTML = "&nbsp;|&nbsp;";
                messageContainer.appendChild(sep);
            }
            const actionEl = document.createElement("a");
            actionEl.innerText = action.label;
            if (action.href) {
                actionEl.target = "_blank";
                actionEl.href = action.href;
            }
            if (action.callback) {
                actionEl.onclick = (e) => {
                    return action.callback(e);
                };
            }
            messageContainer.appendChild(actionEl);
        }
        const messageAnimContainer = document.createElement("div");
        messageAnimContainer.setAttribute("msg-id", data.id);
        messageAnimContainer.appendChild(messageContainer);
        container.appendChild(messageAnimContainer);
        await wait(64);
        messageAnimContainer.style.marginTop = `-${messageAnimContainer.offsetHeight}px`;
        await wait(64);
        messageAnimContainer.classList.add("-show");
        if (data.timeout) {
            await wait(data.timeout);
            this.hideMessage(data.id);
        }
    }
    async hideMessage(id) {
        const msg = document.querySelector(`.rgthree-top-messages-container > [msg-id="${id}"]`);
        if (msg === null || msg === void 0 ? void 0 : msg.classList.contains("-show")) {
            msg.classList.remove("-show");
            await wait(750);
        }
        msg && msg.remove();
    }
    async clearAllMessages() {
        let container = document.querySelector(".rgthree-top-messages-container");
        container && (container.innerHTML = "");
    }
    setLogLevel(level) {
        if (typeof level === "string") {
            level = LogLevelKeyToLogLevel[CONFIG_SERVICE.getConfigValue("log_level")];
        }
        if (level != null) {
            GLOBAL_LOG_LEVEL = level;
        }
    }
    logParts(level, message, ...args) {
        return this.logger.logParts(level, message, ...args);
    }
    newLogSession(name) {
        return this.logger.newSession(name);
    }
    isDebugMode() {
        if (window.location.href.includes("rgthree-debug=false")) {
            return false;
        }
        return GLOBAL_LOG_LEVEL >= LogLevel.DEBUG || window.location.href.includes("rgthree-debug");
    }
    isDevMode() {
        if (window.location.href.includes("rgthree-dev=false")) {
            return false;
        }
        return GLOBAL_LOG_LEVEL >= LogLevel.DEV || window.location.href.includes("rgthree-dev");
    }
    monitorBadLinks() {
        const badLinksFound = WorkflowLinkFixer.create(app.graph).check();
        if (badLinksFound.hasBadLinks && !this.monitorBadLinksAlerted) {
            this.monitorBadLinksAlerted = true;
            alert(`Problematic links just found in live data. Can you save your workflow and file a bug with ` +
                `the last few steps you took to trigger this at ` +
                `https://github.com/rgthree/rgthree-comfy/issues. Thank you!`);
        }
        else if (!badLinksFound.hasBadLinks) {
            this.monitorBadLinksAlerted = false;
        }
        this.monitorLinkTimeout = setTimeout(() => {
            this.monitorBadLinks();
        }, 5000);
    }
}
function getBookmarks() {
    const bookmarks = BOOKMARKS_SERVICE.getCurrentBookmarks();
    const bookmarkItems = bookmarks.map((n) => ({
        content: `[${n.shortcutKey}] ${n.title}`,
        className: "rgthree-contextmenu-item",
        callback: () => {
            n.canvasToBookmark();
        },
    }));
    return !bookmarkItems.length
        ? []
        : [
            {
                content: "🔖 Bookmarks",
                disabled: true,
                className: "rgthree-contextmenu-item rgthree-contextmenu-label",
            },
            ...bookmarkItems,
        ];
}
export const rgthree = new Rgthree();
window.rgthree = rgthree;
app.registerExtension({
    name: "Comfy.RgthreeComfy",
    aboutPageBadges: [
        {
            label: `rgthree-comfy v${rgthree.version}`,
            url: "https://github.com/rgthree/rgthree-comfy",
            icon: "rgthree-comfy-about-badge-logo",
        },
    ],
});
