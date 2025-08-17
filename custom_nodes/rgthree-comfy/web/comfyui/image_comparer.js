import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { RgthreeBaseServerNode } from "./base_node.js";
import { NodeTypesString } from "./constants.js";
import { addConnectionLayoutSupport } from "./utils.js";
import { RgthreeBaseWidget } from "./utils_widgets.js";
import { measureText } from "./utils_canvas.js";
function imageDataToUrl(data) {
    return api.apiURL(`/view?filename=${encodeURIComponent(data.filename)}&type=${data.type}&subfolder=${data.subfolder}${app.getPreviewFormatParam()}${app.getRandParam()}`);
}
export class RgthreeImageComparer extends RgthreeBaseServerNode {
    constructor(title = RgthreeImageComparer.title) {
        super(title);
        this.imageIndex = 0;
        this.imgs = [];
        this.serialize_widgets = true;
        this.isPointerDown = false;
        this.isPointerOver = false;
        this.pointerOverPos = [0, 0];
        this.canvasWidget = null;
        this.properties["comparer_mode"] = "Slide";
    }
    onExecuted(output) {
        var _a;
        (_a = super.onExecuted) === null || _a === void 0 ? void 0 : _a.call(this, output);
        if ("images" in output) {
            this.canvasWidget.value = {
                images: (output.images || []).map((d, i) => {
                    return {
                        name: i === 0 ? "A" : "B",
                        selected: true,
                        url: imageDataToUrl(d),
                    };
                }),
            };
        }
        else {
            output.a_images = output.a_images || [];
            output.b_images = output.b_images || [];
            const imagesToChoose = [];
            const multiple = output.a_images.length + output.b_images.length > 2;
            for (const [i, d] of output.a_images.entries()) {
                imagesToChoose.push({
                    name: output.a_images.length > 1 || multiple ? `A${i + 1}` : "A",
                    selected: i === 0,
                    url: imageDataToUrl(d),
                });
            }
            for (const [i, d] of output.b_images.entries()) {
                imagesToChoose.push({
                    name: output.b_images.length > 1 || multiple ? `B${i + 1}` : "B",
                    selected: i === 0,
                    url: imageDataToUrl(d),
                });
            }
            this.canvasWidget.value = { images: imagesToChoose };
        }
    }
    onSerialize(serialised) {
        var _a;
        super.onSerialize && super.onSerialize(serialised);
        for (let [index, widget_value] of (serialised.widgets_values || []).entries()) {
            if (((_a = this.widgets[index]) === null || _a === void 0 ? void 0 : _a.name) === "rgthree_comparer") {
                serialised.widgets_values[index] = this.widgets[index].value.images.map((d) => {
                    d = { ...d };
                    delete d.img;
                    return d;
                });
            }
        }
    }
    onNodeCreated() {
        this.canvasWidget = this.addCustomWidget(new RgthreeImageComparerWidget("rgthree_comparer", this));
        this.setSize(this.computeSize());
        this.setDirtyCanvas(true, true);
    }
    setIsPointerDown(down = this.isPointerDown) {
        const newIsDown = down && !!app.canvas.pointer_is_down;
        if (this.isPointerDown !== newIsDown) {
            this.isPointerDown = newIsDown;
            this.setDirtyCanvas(true, false);
        }
        this.imageIndex = this.isPointerDown ? 1 : 0;
        if (this.isPointerDown) {
            requestAnimationFrame(() => {
                this.setIsPointerDown();
            });
        }
    }
    onMouseDown(event, pos, canvas) {
        var _a;
        (_a = super.onMouseDown) === null || _a === void 0 ? void 0 : _a.call(this, event, pos, canvas);
        this.setIsPointerDown(true);
        return false;
    }
    onMouseEnter(event) {
        var _a;
        (_a = super.onMouseEnter) === null || _a === void 0 ? void 0 : _a.call(this, event);
        this.setIsPointerDown(!!app.canvas.pointer_is_down);
        this.isPointerOver = true;
    }
    onMouseLeave(event) {
        var _a;
        (_a = super.onMouseLeave) === null || _a === void 0 ? void 0 : _a.call(this, event);
        this.setIsPointerDown(false);
        this.isPointerOver = false;
    }
    onMouseMove(event, pos, canvas) {
        var _a;
        (_a = super.onMouseMove) === null || _a === void 0 ? void 0 : _a.call(this, event, pos, canvas);
        this.pointerOverPos = [...pos];
        this.imageIndex = this.pointerOverPos[0] > this.size[0] / 2 ? 1 : 0;
    }
    getHelp() {
        return `
      <p>
        The ${this.type.replace("(rgthree)", "")} node compares two images on top of each other.
      </p>
      <ul>
        <li>
          <p>
            <strong>Notes</strong>
          </p>
          <ul>
            <li><p>
              The right-click menu may show image options (Open Image, Save Image, etc.) which will
              correspond to the first image (image_a) if clicked on the left-half of the node, or
              the second image if on the right half of the node.
            </p></li>
          </ul>
        </li>
        <li>
          <p>
            <strong>Inputs</strong>
          </p>
          <ul>
            <li><p>
              <code>image_a</code> <i>Optional.</i> The first image to use to compare.
              image_a.
            </p></li>
            <li><p>
              <code>image_b</code> <i>Optional.</i> The second image to use to compare.
            </p></li>
            <li><p>
              <b>Note</b> <code>image_a</code> and <code>image_b</code> work best when a single
              image is provided. However, if each/either are a batch, you can choose which item
              from each batch are chosen to be compared. If either <code>image_a</code> or
              <code>image_b</code> are not provided, the node will choose the first two from the
              provided input if it's a batch, otherwise only show the single image (just as
              Preview Image would).
            </p></li>
          </ul>
        </li>
        <li>
          <p>
            <strong>Properties.</strong> You can change the following properties (by right-clicking
            on the node, and select "Properties" or "Properties Panel" from the menu):
          </p>
          <ul>
            <li><p>
              <code>comparer_mode</code> - Choose between "Slide" and "Click". Defaults to "Slide".
            </p></li>
          </ul>
        </li>
      </ul>`;
    }
    static setUp(comfyClass, nodeData) {
        RgthreeBaseServerNode.registerForOverride(comfyClass, nodeData, RgthreeImageComparer);
    }
    static onRegisteredForOverride(comfyClass) {
        addConnectionLayoutSupport(RgthreeImageComparer, app, [
            ["Left", "Right"],
            ["Right", "Left"],
        ]);
        setTimeout(() => {
            RgthreeImageComparer.category = comfyClass.category;
        });
    }
}
RgthreeImageComparer.title = NodeTypesString.IMAGE_COMPARER;
RgthreeImageComparer.type = NodeTypesString.IMAGE_COMPARER;
RgthreeImageComparer.comfyClass = NodeTypesString.IMAGE_COMPARER;
RgthreeImageComparer["@comparer_mode"] = {
    type: "combo",
    values: ["Slide", "Click"],
};
class RgthreeImageComparerWidget extends RgthreeBaseWidget {
    constructor(name, node) {
        super(name);
        this.type = "custom";
        this.hitAreas = {};
        this.selected = [];
        this._value = { images: [] };
        this.node = node;
    }
    set value(v) {
        let cleanedVal;
        if (Array.isArray(v)) {
            cleanedVal = v.map((d, i) => {
                if (!d || typeof d === "string") {
                    d = { url: d, name: i == 0 ? "A" : "B", selected: true };
                }
                return d;
            });
        }
        else {
            cleanedVal = v.images || [];
        }
        if (cleanedVal.length > 2) {
            const hasAAndB = cleanedVal.some((i) => i.name.startsWith("A")) &&
                cleanedVal.some((i) => i.name.startsWith("B"));
            if (!hasAAndB) {
                cleanedVal = [cleanedVal[0], cleanedVal[1]];
            }
        }
        let selected = cleanedVal.filter((d) => d.selected);
        if (!selected.length && cleanedVal.length) {
            cleanedVal[0].selected = true;
        }
        selected = cleanedVal.filter((d) => d.selected);
        if (selected.length === 1 && cleanedVal.length > 1) {
            cleanedVal.find((d) => !d.selected).selected = true;
        }
        this._value.images = cleanedVal;
        selected = cleanedVal.filter((d) => d.selected);
        this.setSelected(selected);
    }
    get value() {
        return this._value;
    }
    setSelected(selected) {
        this._value.images.forEach((d) => (d.selected = false));
        this.node.imgs.length = 0;
        for (const sel of selected) {
            if (!sel.img) {
                sel.img = new Image();
                sel.img.src = sel.url;
                this.node.imgs.push(sel.img);
            }
            sel.selected = true;
        }
        this.selected = selected;
    }
    draw(ctx, node, width, y) {
        var _a;
        this.hitAreas = {};
        if (this.value.images.length > 2) {
            ctx.textAlign = "left";
            ctx.textBaseline = "top";
            ctx.font = `14px Arial`;
            const drawData = [];
            const spacing = 5;
            let x = 0;
            for (const img of this.value.images) {
                const width = measureText(ctx, img.name);
                drawData.push({
                    img,
                    text: img.name,
                    x,
                    width: measureText(ctx, img.name),
                });
                x += width + spacing;
            }
            x = (node.size[0] - (x - spacing)) / 2;
            for (const d of drawData) {
                ctx.fillStyle = d.img.selected ? "rgba(180, 180, 180, 1)" : "rgba(180, 180, 180, 0.5)";
                ctx.fillText(d.text, x, y);
                this.hitAreas[d.text] = {
                    bounds: [x, y, d.width, 14],
                    data: d.img,
                    onDown: this.onSelectionDown,
                };
                x += d.width + spacing;
            }
            y += 20;
        }
        if (((_a = node.properties) === null || _a === void 0 ? void 0 : _a["comparer_mode"]) === "Click") {
            this.drawImage(ctx, this.selected[this.node.isPointerDown ? 1 : 0], y);
        }
        else {
            this.drawImage(ctx, this.selected[0], y);
            if (node.isPointerOver) {
                this.drawImage(ctx, this.selected[1], y, this.node.pointerOverPos[0]);
            }
        }
    }
    onSelectionDown(event, pos, node, bounds) {
        const selected = [...this.selected];
        if (bounds === null || bounds === void 0 ? void 0 : bounds.data.name.startsWith("A")) {
            selected[0] = bounds.data;
        }
        else if (bounds === null || bounds === void 0 ? void 0 : bounds.data.name.startsWith("B")) {
            selected[1] = bounds.data;
        }
        this.setSelected(selected);
    }
    drawImage(ctx, image, y, cropX) {
        var _a, _b;
        if (!((_a = image === null || image === void 0 ? void 0 : image.img) === null || _a === void 0 ? void 0 : _a.naturalWidth) || !((_b = image === null || image === void 0 ? void 0 : image.img) === null || _b === void 0 ? void 0 : _b.naturalHeight)) {
            return;
        }
        let [nodeWidth, nodeHeight] = this.node.size;
        const imageAspect = (image === null || image === void 0 ? void 0 : image.img.naturalWidth) / (image === null || image === void 0 ? void 0 : image.img.naturalHeight);
        let height = nodeHeight - y;
        const widgetAspect = nodeWidth / height;
        let targetWidth, targetHeight;
        let offsetX = 0;
        if (imageAspect > widgetAspect) {
            targetWidth = nodeWidth;
            targetHeight = nodeWidth / imageAspect;
        }
        else {
            targetHeight = height;
            targetWidth = height * imageAspect;
            offsetX = (nodeWidth - targetWidth) / 2;
        }
        const widthMultiplier = (image === null || image === void 0 ? void 0 : image.img.naturalWidth) / targetWidth;
        const sourceX = 0;
        const sourceY = 0;
        const sourceWidth = cropX != null ? (cropX - offsetX) * widthMultiplier : image === null || image === void 0 ? void 0 : image.img.naturalWidth;
        const sourceHeight = image === null || image === void 0 ? void 0 : image.img.naturalHeight;
        const destX = (nodeWidth - targetWidth) / 2;
        const destY = y + (height - targetHeight) / 2;
        const destWidth = cropX != null ? cropX - offsetX : targetWidth;
        const destHeight = targetHeight;
        ctx.save();
        ctx.beginPath();
        let globalCompositeOperation = ctx.globalCompositeOperation;
        if (cropX) {
            ctx.rect(destX, destY, destWidth, destHeight);
            ctx.clip();
        }
        ctx.drawImage(image === null || image === void 0 ? void 0 : image.img, sourceX, sourceY, sourceWidth, sourceHeight, destX, destY, destWidth, destHeight);
        if (cropX != null && cropX >= (nodeWidth - targetWidth) / 2 && cropX <= targetWidth + offsetX) {
            ctx.beginPath();
            ctx.moveTo(cropX, destY);
            ctx.lineTo(cropX, destY + destHeight);
            ctx.globalCompositeOperation = "difference";
            ctx.strokeStyle = "rgba(255,255,255, 1)";
            ctx.stroke();
        }
        ctx.globalCompositeOperation = globalCompositeOperation;
        ctx.restore();
    }
    computeSize(width) {
        return [width, 20];
    }
    serializeValue(node, index) {
        const v = [];
        for (const data of this._value.images) {
            const d = { ...data };
            delete d.img;
            v.push(d);
        }
        return { images: v };
    }
}
app.registerExtension({
    name: "rgthree.ImageComparer",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === RgthreeImageComparer.type) {
            RgthreeImageComparer.setUp(nodeType, nodeData);
        }
    },
});
