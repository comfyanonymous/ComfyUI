import { api } from "../../scripts/api.js";
import { wait } from "../../rgthree/common/shared_utils.js";
import { rgthree } from "./rgthree.js";
export class PowerPrompt {
    constructor(node, nodeData) {
        this.combos = {};
        this.combosValues = {};
        this.configuring = false;
        this.node = node;
        this.node.properties = this.node.properties || {};
        this.node.properties["combos_filter"] = "";
        this.nodeData = nodeData;
        this.isSimple = this.nodeData.name.includes("Simple");
        this.promptEl = node.widgets[0].inputEl;
        this.addAndHandleKeyboardLoraEditWeight();
        this.patchNodeRefresh();
        const oldConfigure = this.node.configure;
        this.node.configure = (info) => {
            this.configuring = true;
            oldConfigure === null || oldConfigure === void 0 ? void 0 : oldConfigure.apply(this.node, [info]);
            this.configuring = false;
        };
        const oldOnConnectionsChange = this.node.onConnectionsChange;
        this.node.onConnectionsChange = (type, slotIndex, isConnected, link_info, _ioSlot) => {
            oldOnConnectionsChange === null || oldOnConnectionsChange === void 0 ? void 0 : oldOnConnectionsChange.apply(this.node, [type, slotIndex, isConnected, link_info, _ioSlot]);
            this.onNodeConnectionsChange(type, slotIndex, isConnected, link_info, _ioSlot);
        };
        const oldOnConnectInput = this.node.onConnectInput;
        this.node.onConnectInput = (inputIndex, outputType, outputSlot, outputNode, outputIndex) => {
            let canConnect = true;
            if (oldOnConnectInput) {
                canConnect = oldOnConnectInput.apply(this.node, [
                    inputIndex,
                    outputType,
                    outputSlot,
                    outputNode,
                    outputIndex,
                ]);
            }
            return (this.configuring ||
                rgthree.loadingApiJson ||
                (canConnect && !this.node.inputs[inputIndex].disabled));
        };
        const oldOnConnectOutput = this.node.onConnectOutput;
        this.node.onConnectOutput = (outputIndex, inputType, inputSlot, inputNode, inputIndex) => {
            let canConnect = true;
            if (oldOnConnectOutput) {
                canConnect = oldOnConnectOutput === null || oldOnConnectOutput === void 0 ? void 0 : oldOnConnectOutput.apply(this.node, [
                    outputIndex,
                    inputType,
                    inputSlot,
                    inputNode,
                    inputIndex,
                ]);
            }
            return (this.configuring ||
                rgthree.loadingApiJson ||
                (canConnect && !this.node.outputs[outputIndex].disabled));
        };
        const onPropertyChanged = this.node.onPropertyChanged;
        this.node.onPropertyChanged = (property, value, prevValue) => {
            const v = onPropertyChanged && onPropertyChanged.call(this.node, property, value, prevValue);
            if (property === "combos_filter") {
                this.refreshCombos(this.nodeData);
            }
            return v !== null && v !== void 0 ? v : true;
        };
        for (let i = this.node.widgets.length - 1; i >= 0; i--) {
            if (this.shouldRemoveServerWidget(this.node.widgets[i])) {
                this.node.widgets.splice(i, 1);
            }
        }
        this.refreshCombos(nodeData);
        setTimeout(() => {
            this.stabilizeInputsOutputs();
        }, 32);
    }
    onNodeConnectionsChange(_type, _slotIndex, _isConnected, _linkInfo, _ioSlot) {
        this.stabilizeInputsOutputs();
    }
    stabilizeInputsOutputs() {
        if (this.configuring || rgthree.loadingApiJson) {
            return;
        }
        const clipLinked = this.node.inputs.some((i) => i.name.includes("clip") && !!i.link);
        const modelLinked = this.node.inputs.some((i) => i.name.includes("model") && !!i.link);
        for (const output of this.node.outputs) {
            const type = output.type.toLowerCase();
            if (type.includes("model")) {
                output.disabled = !modelLinked;
            }
            else if (type.includes("conditioning")) {
                output.disabled = !clipLinked;
            }
            else if (type.includes("clip")) {
                output.disabled = !clipLinked;
            }
            else if (type.includes("string")) {
                output.color_off = "#7F7";
                output.color_on = "#7F7";
            }
            if (output.disabled) {
            }
        }
    }
    onFreshNodeDefs(event) {
        this.refreshCombos(event.detail[this.nodeData.name]);
    }
    shouldRemoveServerWidget(widget) {
        var _a, _b, _c, _d;
        return (((_a = widget.name) === null || _a === void 0 ? void 0 : _a.startsWith("insert_")) ||
            ((_b = widget.name) === null || _b === void 0 ? void 0 : _b.startsWith("target_")) ||
            ((_c = widget.name) === null || _c === void 0 ? void 0 : _c.startsWith("crop_")) ||
            ((_d = widget.name) === null || _d === void 0 ? void 0 : _d.startsWith("values_")));
    }
    refreshCombos(nodeData) {
        var _a, _b, _c;
        this.nodeData = nodeData;
        let filter = null;
        if ((_a = this.node.properties["combos_filter"]) === null || _a === void 0 ? void 0 : _a.trim()) {
            try {
                filter = new RegExp(this.node.properties["combos_filter"].trim(), "i");
            }
            catch (e) {
                console.error(`Could not parse "${filter}" for Regular Expression`, e);
                filter = null;
            }
        }
        let data = Object.assign({}, ((_b = this.nodeData.input) === null || _b === void 0 ? void 0 : _b.optional) || {}, ((_c = this.nodeData.input) === null || _c === void 0 ? void 0 : _c.hidden) || {});
        for (const [key, value] of Object.entries(data)) {
            if (Array.isArray(value[0])) {
                let values = value[0];
                if (key.startsWith("insert")) {
                    values = filter
                        ? values.filter((v, i) => i < 1 || (i == 1 && v.match(/^disable\s[a-z]/i)) || (filter === null || filter === void 0 ? void 0 : filter.test(v)))
                        : values;
                    const shouldShow = values.length > 2 || (values.length > 1 && !values[1].match(/^disable\s[a-z]/i));
                    if (shouldShow) {
                        if (!this.combos[key]) {
                            this.combos[key] = this.node.addWidget("combo", key, values[0], (selected) => {
                                if (selected !== values[0] && !selected.match(/^disable\s[a-z]/i)) {
                                    wait().then(() => {
                                        if (key.includes("embedding")) {
                                            this.insertSelectionText(`embedding:${selected}`);
                                        }
                                        else if (key.includes("saved")) {
                                            this.insertSelectionText(this.combosValues[`values_${key}`][values.indexOf(selected)]);
                                        }
                                        else if (key.includes("lora")) {
                                            this.insertSelectionText(`<lora:${selected}:1.0>`);
                                        }
                                        this.combos[key].value = values[0];
                                    });
                                }
                            }, {
                                values,
                                serialize: true,
                            });
                            this.combos[key].oldComputeSize = this.combos[key].computeSize;
                            let node = this.node;
                            this.combos[key].computeSize = function (width) {
                                var _a, _b;
                                const size = ((_b = (_a = this).oldComputeSize) === null || _b === void 0 ? void 0 : _b.call(_a, width)) || [
                                    width,
                                    LiteGraph.NODE_WIDGET_HEIGHT,
                                ];
                                if (this === node.widgets[node.widgets.length - 1]) {
                                    size[1] += 10;
                                }
                                return size;
                            };
                        }
                        this.combos[key].options.values = values;
                        this.combos[key].value = values[0];
                    }
                    else if (!shouldShow && this.combos[key]) {
                        this.node.widgets.splice(this.node.widgets.indexOf(this.combos[key]), 1);
                        delete this.combos[key];
                    }
                }
                else if (key.startsWith("values")) {
                    this.combosValues[key] = values;
                }
            }
        }
    }
    insertSelectionText(text) {
        if (!this.promptEl) {
            console.error("Asked to insert text, but no textbox found.");
            return;
        }
        let prompt = this.promptEl.value;
        let first = prompt.substring(0, this.promptEl.selectionEnd).replace(/ +$/, "");
        first = first + (["\n"].includes(first[first.length - 1]) ? "" : first.length ? " " : "");
        let second = prompt.substring(this.promptEl.selectionEnd).replace(/^ +/, "");
        second = (["\n"].includes(second[0]) ? "" : second.length ? " " : "") + second;
        this.promptEl.value = first + text + second;
        this.promptEl.focus();
        this.promptEl.selectionStart = first.length;
        this.promptEl.selectionEnd = first.length + text.length;
    }
    addAndHandleKeyboardLoraEditWeight() {
        this.promptEl.addEventListener("keydown", (event) => {
            var _a, _b;
            if (!(event.key === "ArrowUp" || event.key === "ArrowDown"))
                return;
            if (!event.ctrlKey && !event.metaKey)
                return;
            const delta = event.shiftKey ? 0.01 : 0.1;
            let start = this.promptEl.selectionStart;
            let end = this.promptEl.selectionEnd;
            let fullText = this.promptEl.value;
            let selectedText = fullText.substring(start, end);
            if (!selectedText) {
                const stopOn = "<>()\r\n\t";
                if (fullText[start] == ">") {
                    start -= 2;
                    end -= 2;
                }
                if (fullText[end - 1] == "<") {
                    start += 2;
                    end += 2;
                }
                while (!stopOn.includes(fullText[start]) && start > 0) {
                    start--;
                }
                while (!stopOn.includes(fullText[end - 1]) && end < fullText.length) {
                    end++;
                }
                selectedText = fullText.substring(start, end);
            }
            if (!selectedText.startsWith("<lora:") || !selectedText.endsWith(">")) {
                return;
            }
            let weight = (_b = Number((_a = selectedText.match(/:(-?\d*(\.\d*)?)>$/)) === null || _a === void 0 ? void 0 : _a[1])) !== null && _b !== void 0 ? _b : 1;
            weight += event.key === "ArrowUp" ? delta : -delta;
            const updatedText = selectedText.replace(/(:-?\d*(\.\d*)?)?>$/, `:${weight.toFixed(2)}>`);
            this.promptEl.setRangeText(updatedText, start, end, "select");
            event.preventDefault();
            event.stopPropagation();
        });
    }
    patchNodeRefresh() {
        this.boundOnFreshNodeDefs = this.onFreshNodeDefs.bind(this);
        api.addEventListener("fresh-node-defs", this.boundOnFreshNodeDefs);
        const oldNodeRemoved = this.node.onRemoved;
        this.node.onRemoved = () => {
            oldNodeRemoved === null || oldNodeRemoved === void 0 ? void 0 : oldNodeRemoved.call(this.node);
            api.removeEventListener("fresh-node-defs", this.boundOnFreshNodeDefs);
        };
    }
}
