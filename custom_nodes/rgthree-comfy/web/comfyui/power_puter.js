import { app } from "../../scripts/app.js";
import { RgthreeBaseServerNode } from "./base_node.js";
import { NodeTypesString } from "./constants.js";
import { removeUnusedInputsFromEnd } from "./utils_inputs_outputs.js";
import { debounce } from "../../rgthree/common/shared_utils.js";
import { ComfyWidgets } from "../../scripts/widgets.js";
import { RgthreeBaseWidget } from "./utils_widgets.js";
import { drawPlusIcon, drawRoundedRectangle, drawWidgetButton, isLowQuality, measureText, } from "./utils_canvas.js";
import { rgthree } from "./rgthree.js";
const ALPHABET = "abcdefghijklmnopqrstuv".split("");
const OUTPUT_TYPES = ["STRING", "INT", "FLOAT", "BOOLEAN", "*"];
class RgthreePowerPuter extends RgthreeBaseServerNode {
    constructor(title = NODE_CLASS.title) {
        super(title);
        this.stabilizeBound = this.stabilize.bind(this);
        this.addAnyInput(2);
        this.addInitialWidgets();
    }
    static setUp(comfyClass, nodeData) {
        RgthreeBaseServerNode.registerForOverride(comfyClass, nodeData, NODE_CLASS);
    }
    onConnectionsChange(...args) {
        var _a;
        (_a = super.onConnectionsChange) === null || _a === void 0 ? void 0 : _a.apply(this, [...arguments]);
        this.scheduleStabilize();
    }
    scheduleStabilize(ms = 64) {
        return debounce(this.stabilizeBound, ms);
    }
    stabilize() {
        removeUnusedInputsFromEnd(this, 1);
        this.addAnyInput();
        this.setOutputs();
    }
    addInitialWidgets() {
        if (!this.outputTypeWidget) {
            this.outputTypeWidget = this.addCustomWidget(new OutputsWidget("outputs", this));
            this.expressionWidget = ComfyWidgets["STRING"](this, "code", ["STRING", { multiline: true }], app).widget;
        }
    }
    addAnyInput(num = 1) {
        for (let i = 0; i < num; i++) {
            this.addInput(ALPHABET[this.inputs.length], "*");
        }
    }
    setOutputs() {
        const desiredOutputs = this.outputTypeWidget.value.outputs;
        for (let i = 0; i < Math.max(this.outputs.length, desiredOutputs.length); i++) {
            const desired = desiredOutputs[i];
            let output = this.outputs[i];
            if (!desired && output) {
                this.disconnectOutput(i);
                this.removeOutput(i);
                continue;
            }
            output = output || this.addOutput("", "");
            const outputLabel = output.label === "*" || output.label === output.type ? null : output.label;
            output.type = String(desired);
            output.label = outputLabel || output.type;
        }
    }
    getHelp() {
        return `
      <p>
        The ${this.type.replace("(rgthree)", "")} is a powerful and versatile node that opens the
        door for a wide range of utility by offering mult-line code parsing for output. This node
        can be used for simple string concatenation, or math operations; to an image dimension or a
        node's widgets with advanced list comprehension.
        If you want to output something in your workflow, this is the node to do it.
      </p>

      <ul>
        <li><p>
          Evaluate almost any kind of input and more, and choose your output from INT, FLOAT,
          STRING, or BOOLEAN.
        </p></li>
        <li><p>
          Connect some nodes and do simply math operations like <code>a + b</code> or
          <code>ceil(1 / 2)</code>.
        </p></li>
        <li><p>
          Or do more advanced things, like input an image, and get the width like
          <code>a.shape[2]</code>.
        </p></li>
        <li><p>
          Even more powerful, you can target nodes in the prompt that's sent to the backend. For
          instance; if you have a Power Lora Loader node at id #5, and want to get a comma-delimited
          list of the enabled loras, you could enter
          <code>', '.join([v.lora for v in node(5).inputs.values() if 'lora' in v and v.on])</code>.
        </p></li>
        <li><p>
          See more at the <a target="_blank"
          href="https://github.com/rgthree/rgthree-comfy/wiki/Node:-Power-Puter">rgthree-comfy
          wiki</a>.
        </p></li>
      </ul>`;
    }
}
RgthreePowerPuter.title = NodeTypesString.POWER_PUTER;
RgthreePowerPuter.type = NodeTypesString.POWER_PUTER;
RgthreePowerPuter.comfyClass = NodeTypesString.POWER_PUTER;
const NODE_CLASS = RgthreePowerPuter;
const OUTPUTS_WIDGET_CHIP_HEIGHT = LiteGraph.NODE_WIDGET_HEIGHT - 4;
const OUTPUTS_WIDGET_CHIP_SPACE = 4;
const OUTPUTS_WIDGET_CHIP_ARROW_WIDTH = 5.5;
const OUTPUTS_WIDGET_CHIP_ARROW_HEIGHT = 4;
class OutputsWidget extends RgthreeBaseWidget {
    constructor(name, node) {
        super(name);
        this.type = "custom";
        this._value = { outputs: ["STRING"] };
        this.rows = 1;
        this.neededHeight = LiteGraph.NODE_WIDGET_HEIGHT + 8;
        this.hitAreas = {
            add: { bounds: [0, 0], onClick: this.onAddChipDown },
            output0: { bounds: [0, 0], onClick: this.onOutputChipDown, data: { index: 0 } },
            output1: { bounds: [0, 0], onClick: this.onOutputChipDown, data: { index: 1 } },
            output2: { bounds: [0, 0], onClick: this.onOutputChipDown, data: { index: 2 } },
            output3: { bounds: [0, 0], onClick: this.onOutputChipDown, data: { index: 3 } },
            output4: { bounds: [0, 0], onClick: this.onOutputChipDown, data: { index: 4 } },
            output5: { bounds: [0, 0], onClick: this.onOutputChipDown, data: { index: 5 } },
            output6: { bounds: [0, 0], onClick: this.onOutputChipDown, data: { index: 6 } },
            output7: { bounds: [0, 0], onClick: this.onOutputChipDown, data: { index: 7 } },
            output8: { bounds: [0, 0], onClick: this.onOutputChipDown, data: { index: 8 } },
            output9: { bounds: [0, 0], onClick: this.onOutputChipDown, data: { index: 9 } },
        };
        this.node = node;
    }
    set value(v) {
        let outputs = typeof v === "string" ? [v] : [...v.outputs];
        outputs = outputs.map((o) => (o === "BOOL" ? "BOOLEAN" : o));
        this._value.outputs = outputs;
    }
    get value() {
        return this._value;
    }
    onAddChipDown(event, pos, node, bounds) {
        new LiteGraph.ContextMenu(OUTPUT_TYPES, {
            event: event,
            title: "Add an output",
            className: "rgthree-dark",
            callback: (value) => {
                if (isLowQuality())
                    return;
                if (typeof value === "string" && OUTPUT_TYPES.includes(value)) {
                    this._value.outputs.push(value);
                    this.node.scheduleStabilize();
                }
            },
        });
        this.cancelMouseDown();
        return true;
    }
    onOutputChipDown(event, pos, node, bounds) {
        const options = [...OUTPUT_TYPES];
        if (this.value.outputs.length > 1) {
            options.push(null, "🗑️ Delete");
        }
        new LiteGraph.ContextMenu(options, {
            event: event,
            title: `Edit output #${bounds.data.index + 1}`,
            className: "rgthree-dark",
            callback: (value) => {
                var _a, _b;
                const index = bounds.data.index;
                if (typeof value !== "string" || value === this._value.outputs[index] || isLowQuality()) {
                    return;
                }
                const output = this.node.outputs[index];
                if (value.toLocaleLowerCase().includes("delete")) {
                    if ((_a = output.links) === null || _a === void 0 ? void 0 : _a.length) {
                        rgthree.showMessage({
                            id: "puter-remove-linked-output",
                            type: "warn",
                            message: "[Power Puter] Removed and disconnected output from that was connected!",
                            timeout: 3000,
                        });
                        this.node.disconnectOutput(index);
                    }
                    this.node.removeOutput(index);
                    this._value.outputs.splice(index, 1);
                    this.node.scheduleStabilize();
                    return;
                }
                if (((_b = output.links) === null || _b === void 0 ? void 0 : _b.length) && value !== "*") {
                    rgthree.showMessage({
                        id: "puter-remove-linked-output",
                        type: "warn",
                        message: "[Power Puter] Changing output type of linked output! You should check for" +
                            " compatibility.",
                        timeout: 3000,
                    });
                }
                this._value.outputs[index] = value;
                this.node.scheduleStabilize();
            },
        });
        this.cancelMouseDown();
        return true;
    }
    computeLayoutSize(node) {
        this.neededHeight =
            OUTPUTS_WIDGET_CHIP_SPACE +
                (OUTPUTS_WIDGET_CHIP_HEIGHT + OUTPUTS_WIDGET_CHIP_SPACE) * this.rows;
        return {
            minHeight: this.neededHeight,
            maxHeight: this.neededHeight,
            minWidth: 0,
        };
    }
    draw(ctx, node, w, posY, height) {
        var _a, _b;
        ctx.save();
        height = this.neededHeight;
        const margin = 10;
        const innerMargin = margin * 0.33;
        const width = node.size[0] - margin * 2;
        let borderRadius = LiteGraph.NODE_WIDGET_HEIGHT * 0.5;
        let midY = posY + height * 0.5;
        let posX = margin;
        let rposX = node.size[0] - margin;
        drawRoundedRectangle(ctx, { pos: [posX, posY], size: [width, height], borderRadius });
        posX += innerMargin * 2;
        rposX -= innerMargin * 2;
        if (isLowQuality()) {
            ctx.restore();
            return;
        }
        ctx.fillStyle = LiteGraph.WIDGET_SECONDARY_TEXT_COLOR;
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText("outputs", posX, midY);
        posX += measureText(ctx, "outputs") + innerMargin * 2;
        ctx.stroke(new Path2D(`M ${posX} ${posY} v ${height}`));
        posX += 1 + innerMargin * 2;
        const inititalPosX = posX;
        posY += OUTPUTS_WIDGET_CHIP_SPACE;
        height = OUTPUTS_WIDGET_CHIP_HEIGHT;
        borderRadius = height * 0.5;
        midY = posY + height / 2;
        ctx.textAlign = "center";
        ctx.lineJoin = ctx.lineCap = "round";
        ctx.fillStyle = ctx.strokeStyle = LiteGraph.WIDGET_TEXT_COLOR;
        let rows = 1;
        const values = (_b = (_a = this.value) === null || _a === void 0 ? void 0 : _a.outputs) !== null && _b !== void 0 ? _b : [];
        const fontSize = ctx.font.match(/(\d+)px/);
        if (fontSize === null || fontSize === void 0 ? void 0 : fontSize[1]) {
            ctx.font = ctx.font.replace(fontSize[1], `${Number(fontSize[1]) - 2}`);
        }
        let i = 0;
        for (i; i < values.length; i++) {
            const hitArea = this.hitAreas[`output${i}`];
            const isClicking = !!hitArea.wasMouseClickedAndIsOver;
            hitArea.data.index = i;
            const text = values[i];
            const textWidth = measureText(ctx, text) + innerMargin * 2;
            const width = textWidth + OUTPUTS_WIDGET_CHIP_ARROW_WIDTH + innerMargin * 5;
            if (posX + width >= rposX) {
                posX = inititalPosX;
                posY = posY + height + 4;
                midY = posY + height / 2;
                rows++;
            }
            drawWidgetButton(ctx, { pos: [posX, posY], size: [width, height], borderRadius }, null, isClicking);
            const startX = posX;
            posX += innerMargin * 2;
            const newMidY = midY + (isClicking ? 1 : 0);
            ctx.fillText(text, posX + textWidth / 2, newMidY);
            posX += textWidth + innerMargin;
            const arrow = new Path2D(`M${posX} ${newMidY - OUTPUTS_WIDGET_CHIP_ARROW_HEIGHT / 2}
         h${OUTPUTS_WIDGET_CHIP_ARROW_WIDTH}
         l-${OUTPUTS_WIDGET_CHIP_ARROW_WIDTH / 2} ${OUTPUTS_WIDGET_CHIP_ARROW_HEIGHT} z`);
            ctx.fill(arrow);
            ctx.stroke(arrow);
            posX += OUTPUTS_WIDGET_CHIP_ARROW_WIDTH + innerMargin * 2;
            hitArea.bounds = [startX, posY, width, height];
            posX += OUTPUTS_WIDGET_CHIP_SPACE;
        }
        for (i; i < 9; i++) {
            const hitArea = this.hitAreas[`output${i}`];
            if (hitArea.bounds[0] > 0) {
                hitArea.bounds = [0, 0, 0, 0];
            }
        }
        const addHitArea = this.hitAreas["add"];
        if (this.value.outputs.length < 10) {
            const isClicking = !!addHitArea.wasMouseClickedAndIsOver;
            const plusSize = 10;
            let plusWidth = innerMargin * 2 + plusSize + innerMargin * 2;
            if (posX + plusWidth >= rposX) {
                posX = inititalPosX;
                posY = posY + height + 4;
                midY = posY + height / 2;
                rows++;
            }
            drawWidgetButton(ctx, { size: [plusWidth, height], pos: [posX, posY], borderRadius }, null, isClicking);
            drawPlusIcon(ctx, posX + innerMargin * 2, midY + (isClicking ? 1 : 0), plusSize);
            addHitArea.bounds = [posX, posY, plusWidth, height];
        }
        else {
            addHitArea.bounds = [0, 0, 0, 0];
        }
        this.rows = rows;
        ctx.restore();
    }
}
app.registerExtension({
    name: "rgthree.PowerPuter",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === NODE_CLASS.type) {
            NODE_CLASS.setUp(nodeType, nodeData);
        }
    },
});
