import { app } from "../../scripts/app.js";
import { RgthreeBaseVirtualNode } from "./base_node.js";
import { NodeTypesString } from "./constants.js";
import { rgthree } from "./rgthree.js";
export class Label extends RgthreeBaseVirtualNode {
    constructor(title = Label.title) {
        super(title);
        this.comfyClass = NodeTypesString.LABEL;
        this.resizable = false;
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
    draw(ctx) {
        var _a, _b;
        this.flags = this.flags || {};
        this.flags.allow_interaction = !this.flags.pinned;
        ctx.save();
        this.color = "#fff0";
        this.bgcolor = "#fff0";
        const fontColor = this.properties["fontColor"] || "#ffffff";
        const backgroundColor = this.properties["backgroundColor"] || "";
        ctx.font = `${Math.max(this.properties["fontSize"] || 0, 1)}px ${(_a = this.properties["fontFamily"]) !== null && _a !== void 0 ? _a : "Arial"}`;
        const padding = (_b = Number(this.properties["padding"])) !== null && _b !== void 0 ? _b : 0;
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
        }
        else if (this.properties["textAlign"] === "right") {
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
    onDblClick(event, pos, canvas) {
        LGraphCanvas.active_canvas.showShowNodePanel(this);
    }
    onShowCustomPanelInfo(panel) {
        var _a, _b;
        (_a = panel.querySelector('div.property[data-property="Mode"]')) === null || _a === void 0 ? void 0 : _a.remove();
        (_b = panel.querySelector('div.property[data-property="Color"]')) === null || _b === void 0 ? void 0 : _b.remove();
    }
    inResizeCorner(x, y) {
        return this.resizable;
    }
    getHelp() {
        return `
      <p>
        The rgthree-comfy ${this.type.replace("(rgthree)", "")} node allows you to add a floating
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
Label.type = NodeTypesString.LABEL;
Label.title = NodeTypesString.LABEL;
Label.title_mode = LiteGraph.NO_TITLE;
Label.collapsable = false;
Label["@fontSize"] = { type: "number" };
Label["@fontFamily"] = { type: "string" };
Label["@fontColor"] = { type: "string" };
Label["@textAlign"] = { type: "combo", values: ["left", "center", "right"] };
Label["@backgroundColor"] = { type: "string" };
Label["@padding"] = { type: "number" };
Label["@borderRadius"] = { type: "number" };
const oldDrawNode = LGraphCanvas.prototype.drawNode;
LGraphCanvas.prototype.drawNode = function (node, ctx) {
    if (node.constructor === Label.prototype.constructor) {
        node.bgcolor = "transparent";
        node.color = "transparent";
        const v = oldDrawNode.apply(this, arguments);
        node.draw(ctx);
        return v;
    }
    const v = oldDrawNode.apply(this, arguments);
    return v;
};
const oldGetNodeOnPos = LGraph.prototype.getNodeOnPos;
LGraph.prototype.getNodeOnPos = function (x, y, nodes_list) {
    var _a, _b;
    if (nodes_list &&
        rgthree.processingMouseDown &&
        ((_a = rgthree.lastCanvasMouseEvent) === null || _a === void 0 ? void 0 : _a.type.includes("down")) &&
        ((_b = rgthree.lastCanvasMouseEvent) === null || _b === void 0 ? void 0 : _b.which) === 1) {
        let isDoubleClick = LiteGraph.getTime() - LGraphCanvas.active_canvas.last_mouseclick < 300;
        if (!isDoubleClick) {
            nodes_list = [...nodes_list].filter((n) => { var _a; return !(n instanceof Label) || !((_a = n.flags) === null || _a === void 0 ? void 0 : _a.pinned); });
        }
    }
    return oldGetNodeOnPos.apply(this, [x, y, nodes_list]);
};
app.registerExtension({
    name: "rgthree.Label",
    registerCustomNodes() {
        Label.setUp();
    },
});
