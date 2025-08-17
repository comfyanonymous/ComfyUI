import type {
  LGraphNode,
  IWidget,
  LGraphNodeConstructor,
  Vector2,
  CanvasMouseEvent,
} from "@comfyorg/frontend";
import type {ComfyNodeDef} from "typings/comfy.js";

import {app} from "scripts/app.js";
import {RgthreeBaseServerNode} from "./base_node.js";
import {NodeTypesString} from "./constants.js";
import {removeUnusedInputsFromEnd} from "./utils_inputs_outputs.js";
import {debounce} from "rgthree/common/shared_utils.js";
import {ComfyWidgets} from "scripts/widgets.js";
import {RgthreeBaseHitAreas, RgthreeBaseWidget, RgthreeBaseWidgetBounds} from "./utils_widgets.js";
import {
  drawPlusIcon,
  drawRoundedRectangle,
  drawWidgetButton,
  isLowQuality,
  measureText,
} from "./utils_canvas.js";
import {rgthree} from "./rgthree.js";

type Vector4 = [number, number, number, number];

const ALPHABET = "abcdefghijklmnopqrstuv".split("");

const OUTPUT_TYPES = ["STRING", "INT", "FLOAT", "BOOLEAN", "*"];

class RgthreePowerPuter extends RgthreeBaseServerNode {
  static override title = NodeTypesString.POWER_PUTER;
  static override type = NodeTypesString.POWER_PUTER;
  static comfyClass = NodeTypesString.POWER_PUTER;

  private outputTypeWidget!: OutputsWidget;
  private expressionWidget!: IWidget;
  private stabilizeBound = this.stabilize.bind(this);

  constructor(title = NODE_CLASS.title) {
    super(title);
    // Note, configure will add as many as was in the stored workflow automatically.
    this.addAnyInput(2);
    this.addInitialWidgets();
  }

  // /**
  //  * We need to patch in the configure to fix a bug where Power Puter was using BOOL instead of
  //  * BOOLEAN.
  //  */
  // override configure(info: ISerialisedNode): void {
  //   super.configure(info);
  //   // Update BOOL to BOOLEAN due to a bug using BOOL instead of BOOLEAN.
  //   this.outputTypeWidget
  // }

  static override setUp(comfyClass: typeof LGraphNode, nodeData: ComfyNodeDef) {
    RgthreeBaseServerNode.registerForOverride(comfyClass, nodeData, NODE_CLASS);
  }

  override onConnectionsChange(...args: any[]): void {
    super.onConnectionsChange?.apply(this, [...arguments] as any);
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

  private addInitialWidgets() {
    if (!this.outputTypeWidget) {
      this.outputTypeWidget = this.addCustomWidget(
        new OutputsWidget("outputs", this),
      ) as OutputsWidget;
      this.expressionWidget = ComfyWidgets["STRING"](
        this,
        "code",
        ["STRING", {multiline: true}],
        app,
      ).widget;
    }
  }

  private addAnyInput(num = 1) {
    for (let i = 0; i < num; i++) {
      this.addInput(ALPHABET[this.inputs.length]!, "*" as string);
    }
  }

  private setOutputs() {
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
      const outputLabel =
        output.label === "*" || output.label === output.type ? null : output.label;
      output.type = String(desired);
      output.label = outputLabel || output.type;
    }
  }

  override getHelp() {
    return `
      <p>
        The ${this.type!.replace("(rgthree)", "")} is a powerful and versatile node that opens the
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

/** An uniformed name reference to the node class. */
const NODE_CLASS = RgthreePowerPuter;

type OutputsWidgetValue = {
  outputs: string[];
};

const OUTPUTS_WIDGET_CHIP_HEIGHT = LiteGraph.NODE_WIDGET_HEIGHT - 4;
const OUTPUTS_WIDGET_CHIP_SPACE = 4;

const OUTPUTS_WIDGET_CHIP_ARROW_WIDTH = 5.5;
const OUTPUTS_WIDGET_CHIP_ARROW_HEIGHT = 4;

/**
 * The OutputsWidget is an advanced widget that has a background similar to others, but then a
 * series of "chips" that correspond to the outputs of the node. The chips are dynamic and wrap to
 * additional rows as space is needed. Additionally, there is a "+" chip to add more.
 */
class OutputsWidget extends RgthreeBaseWidget<OutputsWidgetValue> {
  override readonly type = "custom";
  private _value: OutputsWidgetValue = {outputs: ["STRING"]};

  private rows = 1;
  private neededHeight = LiteGraph.NODE_WIDGET_HEIGHT + 8;
  private node!: RgthreePowerPuter;

  protected override hitAreas: RgthreeBaseHitAreas<
    | "add"
    | "output0"
    | "output1"
    | "output2"
    | "output3"
    | "output4"
    | "output5"
    | "output6"
    | "output7"
    | "output8"
    | "output9"
  > = {
    add: {bounds: [0, 0] as Vector2, onClick: this.onAddChipDown},
    output0: {bounds: [0, 0] as Vector2, onClick: this.onOutputChipDown, data: {index: 0}},
    output1: {bounds: [0, 0] as Vector2, onClick: this.onOutputChipDown, data: {index: 1}},
    output2: {bounds: [0, 0] as Vector2, onClick: this.onOutputChipDown, data: {index: 2}},
    output3: {bounds: [0, 0] as Vector2, onClick: this.onOutputChipDown, data: {index: 3}},
    output4: {bounds: [0, 0] as Vector2, onClick: this.onOutputChipDown, data: {index: 4}},
    output5: {bounds: [0, 0] as Vector2, onClick: this.onOutputChipDown, data: {index: 5}},
    output6: {bounds: [0, 0] as Vector2, onClick: this.onOutputChipDown, data: {index: 6}},
    output7: {bounds: [0, 0] as Vector2, onClick: this.onOutputChipDown, data: {index: 7}},
    output8: {bounds: [0, 0] as Vector2, onClick: this.onOutputChipDown, data: {index: 8}},
    output9: {bounds: [0, 0] as Vector2, onClick: this.onOutputChipDown, data: {index: 9}},
  };

  constructor(name: string, node: RgthreePowerPuter) {
    super(name);
    this.node = node;
  }

  set value(v: OutputsWidgetValue) {
    // Handle a string being passed in, as the original Power Puter output widget was a string.
    let outputs = typeof v === "string" ? [v] : [...v.outputs];
    // Handle a case where the initial version used "BOOL" instead of "BOOLEAN" incorrectly.
    outputs = outputs.map((o) => (o === "BOOL" ? "BOOLEAN" : o));
    this._value.outputs = outputs;
  }

  get value(): OutputsWidgetValue {
    return this._value;
  }

  /** Displays the menu to choose a new output type. */
  onAddChipDown(
    event: CanvasMouseEvent,
    pos: Vector2,
    node: LGraphNode,
    bounds: RgthreeBaseWidgetBounds,
  ) {
    new LiteGraph.ContextMenu(OUTPUT_TYPES, {
      event: event,
      title: "Add an output",
      className: "rgthree-dark",
      callback: (value) => {
        if (isLowQuality()) return;
        if (typeof value === "string" && OUTPUT_TYPES.includes(value)) {
          this._value.outputs.push(value);
          this.node.scheduleStabilize();
        }
      },
    });

    this.cancelMouseDown();
    return true;
  }

  /** Displays a context menu tied to an output chip within our widget. */
  onOutputChipDown(
    event: CanvasMouseEvent,
    pos: Vector2,
    node: LGraphNode,
    bounds: RgthreeBaseWidgetBounds,
  ) {
    const options: Array<null | string> = [...OUTPUT_TYPES];
    if (this.value.outputs.length > 1) {
      options.push(null, "🗑️ Delete");
    }

    new LiteGraph.ContextMenu(options, {
      event: event,
      title: `Edit output #${bounds.data.index + 1}`,
      className: "rgthree-dark",
      callback: (value) => {
        const index = bounds.data.index;
        if (typeof value !== "string" || value === this._value.outputs[index] || isLowQuality()) {
          return;
        }
        const output = this.node.outputs[index]!;
        if (value.toLocaleLowerCase().includes("delete")) {
          if (output.links?.length) {
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
        if (output.links?.length && value !== "*") {
          rgthree.showMessage({
            id: "puter-remove-linked-output",
            type: "warn",
            message:
              "[Power Puter] Changing output type of linked output! You should check for" +
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

  /**
   * Computes the layout size to ensure the height is what we need to accomodate all the chips;
   * specifically, SPACE on the top, plus the CHIP_HEIGHT + SPACE underneath multiplied by the
   * number of rows necessary.
   */
  computeLayoutSize(node: LGraphNode) {
    this.neededHeight =
      OUTPUTS_WIDGET_CHIP_SPACE +
      (OUTPUTS_WIDGET_CHIP_HEIGHT + OUTPUTS_WIDGET_CHIP_SPACE) * this.rows;
    return {
      minHeight: this.neededHeight,
      maxHeight: this.neededHeight,
      minWidth: 0, // Need just zero here to be flexible with the width.
    };
  }

  /**
   * Draws our nifty, advanced widget keeping track of the space and wrapping to multiple lines when
   * more chips than can fit are shown.
   */
  draw(ctx: CanvasRenderingContext2D, node: LGraphNode, w: number, posY: number, height: number) {
    ctx.save();
    // Despite what `height` was passed in, which is often not our actual height, we'll use oun
    //  calculated needed height.
    height = this.neededHeight;
    const margin = 10;
    const innerMargin = margin * 0.33;
    const width = node.size[0] - margin * 2;
    let borderRadius = LiteGraph.NODE_WIDGET_HEIGHT * 0.5;
    let midY = posY + height * 0.5;
    let posX = margin;
    let rposX = node.size[0] - margin;

    // Draw the background encompassing everything, and move our current posX's to create space from
    // the border.

    drawRoundedRectangle(ctx, {pos: [posX, posY], size: [width, height], borderRadius});
    posX += innerMargin * 2;
    rposX -= innerMargin * 2;

    // If low quality, then we're done.
    if (isLowQuality()) {
      ctx.restore();
      return;
    }

    // Add put our "outputs" label, and a divider line.
    ctx.fillStyle = LiteGraph.WIDGET_SECONDARY_TEXT_COLOR;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText("outputs", posX, midY);
    posX += measureText(ctx, "outputs") + innerMargin * 2;
    ctx.stroke(new Path2D(`M ${posX} ${posY} v ${height}`));
    posX += 1 + innerMargin * 2;

    // Now, prepare our values for the chips; adjust the posY to be within the space, the height to
    // be that of the chips, and the new midY for the chips.
    const inititalPosX = posX;
    posY += OUTPUTS_WIDGET_CHIP_SPACE;
    height = OUTPUTS_WIDGET_CHIP_HEIGHT;
    borderRadius = height * 0.5;
    midY = posY + height / 2;
    ctx.textAlign = "center";
    ctx.lineJoin = ctx.lineCap = "round";
    ctx.fillStyle = ctx.strokeStyle = LiteGraph.WIDGET_TEXT_COLOR;
    let rows = 1;
    const values = this.value?.outputs ?? [];
    const fontSize = ctx.font.match(/(\d+)px/);
    if (fontSize?.[1]) {
      ctx.font = ctx.font.replace(fontSize[1], `${Number(fontSize[1]) - 2}`);
    }

    // Loop over our values, and add them from left to right, measuring the width before placing to
    // see if we need to wrap the the next line, and updating the hitAreas of the chips.
    let i = 0;
    for (i; i < values.length; i++) {
      const hitArea = this.hitAreas[`output${i}` as "output1"];
      const isClicking = !!hitArea.wasMouseClickedAndIsOver;
      hitArea.data.index = i;

      const text = values[i]!;
      const textWidth = measureText(ctx, text) + innerMargin * 2;
      const width = textWidth + OUTPUTS_WIDGET_CHIP_ARROW_WIDTH + innerMargin * 5;

      // If our width is too long, then wrap the values and increment our rows.
      if (posX + width >= rposX) {
        posX = inititalPosX;
        posY = posY + height + 4;
        midY = posY + height / 2;
        rows++;
      }

      drawWidgetButton(
        ctx,
        {pos: [posX, posY], size: [width, height], borderRadius},
        null,
        isClicking,
      );
      const startX = posX;
      posX += innerMargin * 2;
      const newMidY = midY + (isClicking ? 1 : 0);
      ctx.fillText(text, posX + textWidth / 2, newMidY);
      posX += textWidth + innerMargin;
      const arrow = new Path2D(
        `M${posX} ${newMidY - OUTPUTS_WIDGET_CHIP_ARROW_HEIGHT / 2}
         h${OUTPUTS_WIDGET_CHIP_ARROW_WIDTH}
         l-${OUTPUTS_WIDGET_CHIP_ARROW_WIDTH / 2} ${OUTPUTS_WIDGET_CHIP_ARROW_HEIGHT} z`,
      );
      ctx.fill(arrow);
      ctx.stroke(arrow);
      posX += OUTPUTS_WIDGET_CHIP_ARROW_WIDTH + innerMargin * 2;
      hitArea.bounds = [startX, posY, width, height] as Vector4;
      posX += OUTPUTS_WIDGET_CHIP_SPACE; // Space Between
    }
    // Zero out and following hitAreas.
    for (i; i < 9; i++) {
      const hitArea = this.hitAreas[`output${i}` as "output1"];
      if (hitArea.bounds[0] > 0) {
        hitArea.bounds = [0, 0, 0, 0] as Vector4;
      }
    }

    // Draw the add arrow, if we're not at the max.
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
      drawWidgetButton(
        ctx,
        {size: [plusWidth, height], pos: [posX, posY], borderRadius},
        null,
        isClicking,
      );
      drawPlusIcon(ctx, posX + innerMargin * 2, midY + (isClicking ? 1 : 0), plusSize);
      addHitArea.bounds = [posX, posY, plusWidth, height] as Vector4;
    } else {
      addHitArea.bounds = [0, 0, 0, 0] as Vector4;
    }

    // Set the rows now that we're drawn.
    this.rows = rows;
    ctx.restore();
  }
}

/** Register the node. */
app.registerExtension({
  name: "rgthree.PowerPuter",
  async beforeRegisterNodeDef(nodeType: typeof LGraphNode, nodeData: ComfyNodeDef) {
    if (nodeData.name === NODE_CLASS.type) {
      NODE_CLASS.setUp(nodeType, nodeData);
    }
  },
});
