import type {
  INodeInputSlot,
  INodeOutputSlot,
  LGraphCanvas,
  LGraphEventMode,
  LGraphNode,
  LLink,
  Vector2,
  ISerialisedNode,
} from "@comfyorg/frontend";

import {app} from "scripts/app.js";
import {
  PassThroughFollowing,
  addConnectionLayoutSupport,
  changeModeOfNodes,
  getConnectedInputNodesAndFilterPassThroughs,
  getConnectedOutputNodesAndFilterPassThroughs,
} from "./utils.js";
import {wait} from "rgthree/common/shared_utils.js";
import {BaseCollectorNode} from "./base_node_collector.js";
import {NodeTypesString, stripRgthree} from "./constants.js";
import {fitString} from "./utils_canvas.js";
import {rgthree} from "./rgthree.js";

const MODE_ALWAYS = 0;
const MODE_MUTE = 2;
const MODE_BYPASS = 4;
const MODE_REPEATS = [MODE_MUTE, MODE_BYPASS];
const MODE_NOTHING = -99; // MADE THIS UP.

const MODE_TO_OPTION = new Map([
  [MODE_ALWAYS, "ACTIVE"],
  [MODE_MUTE, "MUTE"],
  [MODE_BYPASS, "BYPASS"],
  [MODE_NOTHING, "NOTHING"],
]);

const OPTION_TO_MODE = new Map([
  ["ACTIVE", MODE_ALWAYS],
  ["MUTE", MODE_MUTE],
  ["BYPASS", MODE_BYPASS],
  ["NOTHING", MODE_NOTHING],
]);

const MODE_TO_PROPERTY = new Map([
  [MODE_MUTE, "on_muted_inputs"],
  [MODE_BYPASS, "on_bypassed_inputs"],
  [MODE_ALWAYS, "on_any_active_inputs"],
]);

const logger = rgthree.newLogSession("[NodeModeRelay]");

/**
 * Like a BaseCollectorNode, this relay node connects to a Repeater node and _relays_ mode changes
 * changes to the repeater (so it can go on to modify its connections).
 */
class NodeModeRelay extends BaseCollectorNode {
  override readonly inputsPassThroughFollowing: PassThroughFollowing = PassThroughFollowing.ALL;

  static override type = NodeTypesString.NODE_MODE_RELAY;
  static override title = NodeTypesString.NODE_MODE_RELAY;
  override comfyClass = NodeTypesString.NODE_MODE_RELAY;

  static "@on_muted_inputs" = {
    type: "combo",
    values: ["MUTE", "ACTIVE", "BYPASS", "NOTHING"],
  };

  static "@on_bypassed_inputs" = {
    type: "combo",
    values: ["BYPASS", "ACTIVE", "MUTE", "NOTHING"],
  };

  static "@on_any_active_inputs" = {
    type: "combo",
    values: ["BYPASS", "ACTIVE", "MUTE", "NOTHING"],
  };

  constructor(title?: string) {
    super(title);
    this.properties["on_muted_inputs"] = "MUTE";
    this.properties["on_bypassed_inputs"] = "BYPASS";
    this.properties["on_any_active_inputs"] = "ACTIVE";

    this.onConstructed();
  }

  override onConstructed() {
    this.addOutput("REPEATER", "_NODE_REPEATER_", {
      color_on: "#Fc0",
      color_off: "#a80",
      shape: LiteGraph.ARROW_SHAPE,
    });

    setTimeout(() => {
      this.stabilize();
    }, 500);
    return super.onConstructed();
  }

  override onModeChange(from: LGraphEventMode | undefined, to: LGraphEventMode) {
    super.onModeChange(from, to);
    // If we aren't connected to anything, then we'll use our mode to relay when it changes.
    if (this.inputs.length <= 1 && !this.isInputConnected(0) && this.isAnyOutputConnected()) {
      const [n, v] = logger.infoParts(`Mode change without any inputs; relaying our mode.`);
      console[n]?.(...v);
      // Pass "to" since there may be other getters in the way to access this.mode directly.
      this.dispatchModeToRepeater(to);
    }
  }

  override onDrawForeground(ctx: CanvasRenderingContext2D, canvas: LGraphCanvas): void {
    if (this.flags?.collapsed) {
      return;
    }
    if (
      this.properties["on_muted_inputs"] !== "MUTE" ||
      this.properties["on_bypassed_inputs"] !== "BYPASS" ||
      this.properties["on_any_active_inputs"] != "ACTIVE"
    ) {
      let margin = 15;
      ctx.textAlign = "left";
      let label = `*(MUTE > ${this.properties["on_muted_inputs"]},  `;
      label += `BYPASS > ${this.properties["on_bypassed_inputs"]},  `;
      label += `ACTIVE > ${this.properties["on_any_active_inputs"]})`;
      ctx.fillStyle = LiteGraph.WIDGET_SECONDARY_TEXT_COLOR;
      const oldFont = ctx.font;
      ctx.font = "italic " + (LiteGraph.NODE_SUBTEXT_SIZE - 2) + "px Arial";
      ctx.fillText(fitString(ctx, label, this.size[0] - 20), 15, this.size[1] - 6);
      ctx.font = oldFont;
    }
  }

  override computeSize(out: Vector2) {
    let size = super.computeSize(out);
    if (
      this.properties["on_muted_inputs"] !== "MUTE" ||
      this.properties["on_bypassed_inputs"] !== "BYPASS" ||
      this.properties["on_any_active_inputs"] != "ACTIVE"
    ) {
      size[1] += 17;
    }
    return size;
  }
  override onConnectOutput(
    outputIndex: number,
    inputType: string | -1,
    inputSlot: INodeInputSlot,
    inputNode: LGraphNode,
    inputIndex: number,
  ): boolean {
    let canConnect = super.onConnectOutput?.(
      outputIndex,
      inputType,
      inputSlot,
      inputNode,
      inputIndex,
    );
    let nextNode = getConnectedOutputNodesAndFilterPassThroughs(this, inputNode)[0] ?? inputNode;
    return canConnect && nextNode.type === NodeTypesString.NODE_MODE_REPEATER;
  }

  override onConnectionsChange(
    type: number,
    slotIndex: number,
    isConnected: boolean,
    link_info: LLink,
    ioSlot: INodeOutputSlot | INodeInputSlot,
  ): void {
    super.onConnectionsChange(type, slotIndex, isConnected, link_info, ioSlot);
    setTimeout(() => {
      this.stabilize();
    }, 500);
  }

  stabilize() {
    // If we aren't connected to a repeater, then theres no sense in checking. And if we are, but
    // have no inputs, then we're also not ready.
    if (!this.graph || !this.isAnyOutputConnected() || !this.isInputConnected(0)) {
      return;
    }
    const inputNodes = getConnectedInputNodesAndFilterPassThroughs(
      this,
      this,
      -1,
      this.inputsPassThroughFollowing,
    );
    let mode: LGraphEventMode | -99 | undefined = undefined;
    for (const inputNode of inputNodes) {
      // If we haven't set our mode to be, then let's set it. Otherwise, mode will stick if it
      // remains constant, otherwise, if we hit an ALWAYS, then we'll unmute all repeaters and
      // if not then we won't do anything.
      if (mode === undefined) {
        mode = inputNode.mode;
      } else if (mode === inputNode.mode && MODE_REPEATS.includes(mode)) {
        continue;
      } else if (inputNode.mode === MODE_ALWAYS || mode === MODE_ALWAYS) {
        mode = MODE_ALWAYS;
      } else {
        mode = undefined;
      }
    }

    this.dispatchModeToRepeater(mode);
    setTimeout(() => {
      this.stabilize();
    }, 500);
  }

  /**
   * Sends the mode to the repeater, checking to see if we're modifying our mode.
   */
  private dispatchModeToRepeater(mode?: LGraphEventMode | -99 | null) {
    if (mode != null) {
      const propertyVal = this.properties?.[MODE_TO_PROPERTY.get(mode) || ""];
      const newMode = OPTION_TO_MODE.get(propertyVal as string);
      mode = (newMode !== null ? newMode : mode) as LGraphEventMode | -99;
      if (mode !== null && mode !== MODE_NOTHING) {
        if (this.outputs?.length) {
          const outputNodes = getConnectedOutputNodesAndFilterPassThroughs(this);
          for (const outputNode of outputNodes) {
            changeModeOfNodes(outputNode, mode);
            wait(16).then(() => {
              outputNode.setDirtyCanvas(true, true);
            });
          }
        }
      }
    }
  }

  override getHelp() {
    return `
      <p>
        This node will relay its input nodes' modes (Mute, Bypass, or Active) to a connected
        ${stripRgthree(NodeTypesString.NODE_MODE_REPEATER)} (which would then repeat that mode
        change to all of its inputs).
      </p>
      <ul>
          <li><p>
            When all connected input nodes are muted, the relay will set a connected repeater to
            mute (by default).
          </p></li>
          <li><p>
            When all connected input nodes are bypassed, the relay will set a connected repeater to
            bypass (by default).
          </p></li>
          <li><p>
            When any connected input nodes are active, the relay will set a connected repeater to
            active (by default).
          </p></li>
          <li><p>
            If no inputs are connected, the relay will set a connected repeater to its mode <i>when
            its own mode is changed</i>. <b>Note</b>, if any inputs are connected, then the above
            will occur and the Relay's mode does not matter.
          </p></li>
      </ul>
      <p>
        Note, you can change which signals get sent on the above in the <code>Properties</code>.
        For instance, you could configure an inverse relay which will send a MUTE when any of its
        inputs are active (instead of sending an ACTIVE signal), and send an ACTIVE signal when all
        of its inputs are muted (instead of sending a MUTE signal), etc.
      </p>
    `;
  }
}

app.registerExtension({
  name: "rgthree.NodeModeRepeaterHelper",
  registerCustomNodes() {
    addConnectionLayoutSupport(NodeModeRelay, app, [
      ["Left", "Right"],
      ["Right", "Left"],
    ]);

    LiteGraph.registerNodeType(NodeModeRelay.type, NodeModeRelay);
    NodeModeRelay.category = NodeModeRelay._category;
  },
});
