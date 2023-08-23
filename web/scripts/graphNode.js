import { LGraphNode, LGraphCanvas, BuiltInSlotType, BuiltInSlotShape } from "../lib/litegraph.core.js"
import { ComfyWidgets } from "./widgets.js";
import { iterateNodeDefOutputs, iterateNodeDefInputs } from "./nodeDef.js";

export class ComfyGraphNode extends LGraphNode {

}

// comfy class -> input name -> input config
const defaultInputConfigs = {}

export class ComfyBackendNode extends ComfyGraphNode {
    constructor(title, comfyClass, nodeDef) {
        super(title)
        this.type = comfyClass; // XXX: workaround dependency in LGraphNode.addInput()
        this.displayName = nodeDef.display_name;
        this.comfyNodeDef = nodeDef;
        this.comfyClass = comfyClass;
        this.isBackendNode = true;

        const color = LGraphCanvas.node_colors["yellow"];
        if (this.color == null)
            this.color = color.color
        if (this.bgColor == null)
            this.bgColor = color.bgColor

        this.#setup(nodeDef)

        // if (nodeDef.output_node) {
        //     this.addOutput("OUTPUT", BuiltInSlotType.EVENT, { color_off: "rebeccapurple", color_on: "rebeccapurple" });
        // }
    }

    get isOutputNode() {
        return this.comfyNodeDef.output_node;
    }

    #setup(nodeDef) {
        defaultInputConfigs[this.type] = {}

		const widgets = Object.assign({}, ComfyWidgets, ComfyWidgets.customWidgets);

        for (const [inputName, inputData] of iterateNodeDefInputs(nodeDef)) {
            const config = {};

            const [type, opts] = inputData;

            if (opts?.forceInput) {
                if (Array.isArray(type)) {
                    throw new Error(`Can't have forceInput set to true for an enum type! ${type}`)
                }
                this.addInput(inputName, type);
            } else {
                if (Array.isArray(type)) {
                    // Enums
                    Object.assign(config, widgets.COMBO(this, inputName, inputData) || {});
                } else if (`${type}:${inputName}` in widgets) {
                    // Support custom ComfyWidgets by Type:Name
                    Object.assign(config, widgets[`${type}:${inputName}`](this, inputName, inputData) || {});
                } else if (type in widgets) {
                    // Standard type ComfyWidgets
                    Object.assign(config, widgets[type](this, inputName, inputData) || {});
                } else {
                    // Node connection inputs (backend)
                    this.addInput(inputName, type);
                }
            }

            if ("widgetNodeType" in config)
                ComfyBackendNode.defaultInputConfigs[this.type][inputName] = config.config
        }

        for (const output of iterateNodeDefOutputs(nodeDef)) {
            const outputShape = output.is_list ? BuiltInSlotShape.GRID_SHAPE : BuiltInSlotShape.CIRCLE_SHAPE;
            this.addOutput(output.name, output.type, { shape: outputShape });
        }

        this.serialize_widgets = false;
        // app.#invokeExtensionsAsync("nodeCreated", this);
    }

    // onExecuted(outputData) {
    //     console.warn("onExecuted outputs", outputData)
    //     this.triggerSlot(0, outputData)
    // }
}
