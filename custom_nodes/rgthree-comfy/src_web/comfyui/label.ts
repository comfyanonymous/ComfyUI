import type {
  LGraphCanvas as TLGraphCanvas,
  LGraphNode,
  Vector2,
  CanvasMouseEvent,
} from "@comfyorg/frontend";

import {app} from "scripts/app.js";
import {RgthreeBaseVirtualNode} from "./base_node.js";
import {NodeTypesString} from "./constants.js";
import {rgthree} from "./rgthree.js";

/**
 * A label node that allows you to put floating text anywhere on the graph. The text is the `Title`
 * and the font size, family, color, alignment as well as a background color, padding, and
 * background border radius can all be adjusted in the properties. Multiline text can be added from
 * the properties panel (because ComfyUI let's you shift + enter there, only).
 */
export class Label extends RgthreeBaseVirtualNode {
  static override type = NodeTypesString.LABEL;
  static override title = NodeTypesString.LABEL;
  override comfyClass = NodeTypesString.LABEL;

  static readonly title_mode = LiteGraph.NO_TITLE;
  static collapsable = false;

  static "@fontSize" = {type: "number"};
  static "@fontFamily" = {type: "string"};
  static "@fontColor" = {type: "string"};
  static "@textAlign" = {type: "combo", values: ["left", "center", "right"]};
  static "@backgroundColor" = {type: "string"};
  static "@padding" = {type: "number"};
  static "@borderRadius" = {type: "number"};

  override properties!: RgthreeBaseVirtualNode["properties"] & {
    fontSize: number;
    fontFamily: string;
    fontColor: string;
    textAlign: string;
    backgroundColor: string;
    padding: number;
    borderRadius: number;
  };

  override resizable = false;

  constructor(title = Label.title) {
    super(title);
    this.properties["fontSize"] = 12;
    this.properties["fontFamily"] = "Arial";
    this.properties["fontColor"] = "#ffffff";
    this.properties["textAlign"] = "left";
    this.properties["backgroundColor"] = "transparent";
    this.properties["padding"] = 0;
    this.properties["borderRadius"] = 0;
    this.color = "#fff0";
    this.bgcolor = "#fff0";

    this.onConstructed();
  }

  draw(ctx: CanvasRenderingContext2D) {
    this.flags = this.flags || {};
    this.flags.allow_interaction = !this.flags.pinned;
    ctx.save();
    this.color = "#fff0";
    this.bgcolor = "#fff0";
    const fontColor = this.properties["fontColor"] || "#ffffff";
    const backgroundColor = this.properties["backgroundColor"] || "";
    ctx.font = `${Math.max(this.properties["fontSize"] || 0, 1)}px ${
      this.properties["fontFamily"] ?? "Arial"
    }`;
    const padding = Number(this.properties["padding"]) ?? 0;

    const lines = this.title.replace(/\n*$/, "").split("\n");
    const maxWidth = Math.max(...lines.map((s) => ctx.measureText(s).width));
    this.size[0] = maxWidth + padding * 2;
    this.size[1] = this.properties["fontSize"] * lines.length + padding * 2;
    if (backgroundColor) {
      ctx.beginPath();
      const borderRadius = Number(this.properties["borderRadius"]) || 0;
      ctx.roundRect(0, 0, this.size[0], this.size[1], [borderRadius]);
      ctx.fillStyle = backgroundColor;
      ctx.fill();
    }
    ctx.textAlign = "left";
    let textX = padding;
    if (this.properties["textAlign"] === "center") {
      ctx.textAlign = "center";
      textX = this.size[0] / 2;
    } else if (this.properties["textAlign"] === "right") {
      ctx.textAlign = "right";
      textX = this.size[0] - padding;
    }
    ctx.textBaseline = "top";
    ctx.fillStyle = fontColor;
    let currentY = padding;
    for (let i = 0; i < lines.length; i++) {
      ctx.fillText(lines[i] || " ", textX, currentY);
      currentY += this.properties["fontSize"];
    }
    ctx.restore();
  }

  override onDblClick(event: CanvasMouseEvent, pos: Vector2, canvas: TLGraphCanvas) {
    // Since everything we can do here is in the properties, let's pop open the properties panel.
    LGraphCanvas.active_canvas.showShowNodePanel(this);
  }

  override onShowCustomPanelInfo(panel: HTMLElement) {
    panel.querySelector('div.property[data-property="Mode"]')?.remove();
    panel.querySelector('div.property[data-property="Color"]')?.remove();
  }

  override inResizeCorner(x: number, y: number) {
    // A little ridiculous there's both a resizable property and this method separately to draw the
    // resize icon...
    return this.resizable;
  }

  override getHelp() {
    return `
      <p>
        The rgthree-comfy ${this.type!.replace("(rgthree)", "")} node allows you to add a floating
        label to your workflow.
      </p>
      <p>
        The text shown is the "Title" of the node and you can adjust the the font size, font family,
        font color, text alignment as well as a background color, padding, and background border
        radius from the node's properties. You can double-click the node to open the properties
        panel.
      <p>
      <ul>
        <li>
          <p>
            <strong>Pro tip #1:</strong> You can add multiline text from the properties panel
            <i>(because ComfyUI let's you shift + enter there, only)</i>.
          </p>
        </li>
        <li>
          <p>
            <strong>Pro tip #2:</strong> You can use ComfyUI's native "pin" option in the
            right-click menu to make the label stick to the workflow and clicks to "go through".
            You can right-click at any time to unpin.
          </p>
        </li>
        <li>
          <p>
            <strong>Pro tip #3:</strong> Color values are hexidecimal strings, like "#FFFFFF" for
            white, or "#660000" for dark red. You can supply a 7th & 8th value (or 5th if using
            shorthand) to create a transluscent color. For instance, "#FFFFFF88" is semi-transparent
            white.
          </p>
        </li>
      </ul>`;
  }
}

/**
 * We override the drawNode to see if we're drawing our label and, if so, hijack it so we can draw
 * it like we want. We also do call out to oldDrawNode, which takes care of very minimal things,
 * like a select box.
 */
const oldDrawNode = LGraphCanvas.prototype.drawNode;
LGraphCanvas.prototype.drawNode = function (node: LGraphNode, ctx: CanvasRenderingContext2D) {
  if (node.constructor === Label.prototype.constructor) {
    // These get set very aggressively; maybe an extension is doing it. We'll just clear them out
    // each time.
    (node as Label).bgcolor = "transparent";
    (node as Label).color = "transparent";
    const v = oldDrawNode.apply(this, arguments as any);
    (node as Label).draw(ctx);
    return v;
  }

  const v = oldDrawNode.apply(this, arguments as any);
  return v;
};

/**
 * We override LGraph getNodeOnPos to see if we're being called while also processing a mouse down
 * and, if so, filter out any label nodes on labels that are pinned. This makes the click go
 * "through" the label. We still allow right clicking (so you can unpin) and double click for the
 * properties panel, though that takes two double clicks (one to select, one to actually double
 * click).
 */
const oldGetNodeOnPos = LGraph.prototype.getNodeOnPos;
LGraph.prototype.getNodeOnPos = function (x: number, y: number, nodes_list?: LGraphNode[]) {
  if (
    // processMouseDown always passes in the nodes_list
    nodes_list &&
    rgthree.processingMouseDown &&
    rgthree.lastCanvasMouseEvent?.type.includes("down") &&
    rgthree.lastCanvasMouseEvent?.which === 1
  ) {
    // Using the same logic from LGraphCanvas processMouseDown, let's see if we consider this a
    // double click.
    let isDoubleClick = LiteGraph.getTime() - LGraphCanvas.active_canvas.last_mouseclick < 300;
    if (!isDoubleClick) {
      nodes_list = [...nodes_list].filter((n) => !(n instanceof Label) || !n.flags?.pinned);
    }
  }
  return oldGetNodeOnPos.apply(this, [x, y, nodes_list]);
};

// Register the extension.
app.registerExtension({
  name: "rgthree.Label",
  registerCustomNodes() {
    Label.setUp();
  },
});
