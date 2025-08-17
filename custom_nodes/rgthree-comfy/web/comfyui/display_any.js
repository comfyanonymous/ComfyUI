import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";
import { addConnectionLayoutSupport } from "./utils.js";
import { rgthree } from "./rgthree.js";
let hasShownAlertForUpdatingInt = false;
app.registerExtension({
    name: "rgthree.DisplayAny",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "Display Any (rgthree)" || nodeData.name === "Display Int (rgthree)") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated ? onNodeCreated.apply(this, []) : undefined;
                this.showValueWidget = ComfyWidgets["STRING"](this, "output", ["STRING", { multiline: true }], app).widget;
                this.showValueWidget.inputEl.readOnly = true;
                this.showValueWidget.serializeValue = async (node, index) => {
                    const n = rgthree.getNodeFromInitialGraphToPromptSerializedWorkflowBecauseComfyUIBrokeStuff(node);
                    if (n) {
                        n.widgets_values[index] = "";
                    }
                    else {
                        console.warn("No serialized node found in workflow. May be attributed to " +
                            "https://github.com/comfyanonymous/ComfyUI/issues/2193");
                    }
                    return "";
                };
            };
            addConnectionLayoutSupport(nodeType, app, [["Left"], ["Right"]]);
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted === null || onExecuted === void 0 ? void 0 : onExecuted.apply(this, [message]);
                this.showValueWidget.value = message.text[0];
            };
        }
    },
});
