import {
  LGraphCanvas,
  LGraphNode,
  Vector2,
  LGraphNodeConstructor,
  CanvasMouseEvent,
  ISerialisedNode,
  Point,
  CanvasPointerEvent,
} from "@comfyorg/frontend";
import type {ComfyNodeDef} from "typings/comfy.js";

import {app} from "scripts/app.js";
import {api} from "scripts/api.js";
import {RgthreeBaseServerNode} from "./base_node.js";
import {NodeTypesString} from "./constants.js";
import {addConnectionLayoutSupport} from "./utils.js";
import {RgthreeBaseHitAreas, RgthreeBaseWidget, RgthreeBaseWidgetBounds} from "./utils_widgets.js";
import {measureText} from "./utils_canvas.js";

type ComfyImageServerData = {filename: string; type: string; subfolder: string};
type ComfyImageData = {name: string; selected: boolean; url: string; img?: HTMLImageElement};
type OldExecutedPayload = {
  images: ComfyImageServerData[];
};
type ExecutedPayload = {
  a_images?: ComfyImageServerData[];
  b_images?: ComfyImageServerData[];
};

function imageDataToUrl(data: ComfyImageServerData) {
  return api.apiURL(
    `/view?filename=${encodeURIComponent(data.filename)}&type=${data.type}&subfolder=${
      data.subfolder
    }${app.getPreviewFormatParam()}${app.getRandParam()}`,
  );
}

/**
 * Compares two images in one canvas node.
 */
export class RgthreeImageComparer extends RgthreeBaseServerNode {
  static override title = NodeTypesString.IMAGE_COMPARER;
  static override type = NodeTypesString.IMAGE_COMPARER;
  static comfyClass = NodeTypesString.IMAGE_COMPARER;

  // These is what the core preview image node uses to show the context menu. May not be that helpful
  // since it likely will always be "0" when a context menu is invoked without manually changing
  // something.
  override imageIndex: number = 0;
  override imgs: InstanceType<typeof Image>[] = [];

  override serialize_widgets = true;

  isPointerDown = false;
  isPointerOver = false;
  pointerOverPos: Vector2 = [0, 0];

  private canvasWidget: RgthreeImageComparerWidget | null = null;

  static "@comparer_mode" = {
    type: "combo",
    values: ["Slide", "Click"],
  };

  constructor(title = RgthreeImageComparer.title) {
    super(title);
    this.properties["comparer_mode"] = "Slide";
  }

  override onExecuted(output: ExecutedPayload | OldExecutedPayload) {
    super.onExecuted?.(output);
    if ("images" in output) {
      this.canvasWidget!.value = {
        images: (output.images || []).map((d, i) => {
          return {
            name: i === 0 ? "A" : "B",
            selected: true,
            url: imageDataToUrl(d),
          };
        }),
      };
    } else {
      output.a_images = output.a_images || [];
      output.b_images = output.b_images || [];
      const imagesToChoose: ComfyImageData[] = [];
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
      this.canvasWidget!.value = {images: imagesToChoose};
    }
  }

  override onSerialize(serialised: ISerialisedNode) {
    super.onSerialize && super.onSerialize(serialised);
    for (let [index, widget_value] of (serialised.widgets_values || []).entries()) {
      if (this.widgets[index]?.name === "rgthree_comparer") {
        serialised.widgets_values![index] = (
          this.widgets[index] as unknown as RgthreeImageComparerWidget
        ).value.images.map((d) => {
          d = {...d};
          delete d.img;
          return d;
        });
      }
    }
  }

  override onNodeCreated() {
    this.canvasWidget = this.addCustomWidget(
      new RgthreeImageComparerWidget("rgthree_comparer", this),
    ) as RgthreeImageComparerWidget;
    this.setSize(this.computeSize());
    this.setDirtyCanvas(true, true);
  }

  /**
   * Sets mouse as down or up based on param. If it's down, we also loop to check pointer is still
   * down. This is because LiteGraph doesn't fire `onMouseUp` every time there's a mouse up, so we
   * need to manually monitor `pointer_is_down` and, when it's no longer true, set mouse as up here.
   */
  private setIsPointerDown(down: boolean = this.isPointerDown) {
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

  override onMouseDown(event: CanvasPointerEvent, pos: Point, canvas: LGraphCanvas): boolean {
    super.onMouseDown?.(event, pos, canvas);
    this.setIsPointerDown(true);
    return false;
  }

  override onMouseEnter(event: CanvasPointerEvent): void {
    super.onMouseEnter?.(event);
    this.setIsPointerDown(!!app.canvas.pointer_is_down);
    this.isPointerOver = true;
  }

  override onMouseLeave(event: CanvasPointerEvent): void {
    super.onMouseLeave?.(event);
    this.setIsPointerDown(false);
    this.isPointerOver = false;
  }

  override onMouseMove(event: CanvasPointerEvent, pos: Point, canvas: LGraphCanvas): void {
    super.onMouseMove?.(event, pos, canvas);
    this.pointerOverPos = [...pos] as Point;
    this.imageIndex = this.pointerOverPos[0] > this.size[0] / 2 ? 1 : 0;
  }

  override getHelp(): string {
    return `
      <p>
        The ${this.type!.replace("(rgthree)", "")} node compares two images on top of each other.
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

  static override setUp(comfyClass: typeof LGraphNode, nodeData: ComfyNodeDef) {
    RgthreeBaseServerNode.registerForOverride(comfyClass, nodeData, RgthreeImageComparer);
  }

  static override onRegisteredForOverride(comfyClass: any) {
    addConnectionLayoutSupport(RgthreeImageComparer, app, [
      ["Left", "Right"],
      ["Right", "Left"],
    ]);
    setTimeout(() => {
      RgthreeImageComparer.category = comfyClass.category;
    });
  }
}

type RgthreeImageComparerWidgetValue = {
  images: ComfyImageData[];
};

class RgthreeImageComparerWidget extends RgthreeBaseWidget<RgthreeImageComparerWidgetValue> {
  override readonly type = "custom";

  private node: RgthreeImageComparer;

  protected override hitAreas: RgthreeBaseHitAreas<any> = {
    // We dynamically set this when/if we draw the labels.
  };

  private selected: [ComfyImageData?, ComfyImageData?] = [];

  constructor(name: string, node: RgthreeImageComparer) {
    super(name);
    this.node = node;
  }

  private _value: RgthreeImageComparerWidgetValue = {images: []};

  set value(v: RgthreeImageComparerWidgetValue) {
    // Despite `v` typed as RgthreeImageComparerWidgetValue, we may have gotten an array of strings
    // from previous versions. We can handle that gracefully.
    let cleanedVal;
    if (Array.isArray(v)) {
      cleanedVal = v.map((d, i) => {
        if (!d || typeof d === "string") {
          // We usually only have two here, so they're selected.
          d = {url: d, name: i == 0 ? "A" : "B", selected: true};
        }
        return d;
      });
    } else {
      cleanedVal = v.images || [];
    }

    // If we have multiple items in our sent value but we don't have both an "A" and a "B" then
    // just simplify it down to the first two in the list.
    if (cleanedVal.length > 2) {
      const hasAAndB =
        cleanedVal.some((i) => i.name.startsWith("A")) &&
        cleanedVal.some((i) => i.name.startsWith("B"));
      if (!hasAAndB) {
        cleanedVal = [cleanedVal[0], cleanedVal[1]];
      }
    }

    let selected = cleanedVal.filter((d) => d.selected);
    // None are selected.
    if (!selected.length && cleanedVal.length) {
      cleanedVal[0]!.selected = true;
    }

    selected = cleanedVal.filter((d) => d.selected);
    if (selected.length === 1 && cleanedVal.length > 1) {
      cleanedVal.find((d) => !d.selected)!.selected = true;
    }

    this._value.images = cleanedVal;

    selected = cleanedVal.filter((d) => d.selected);
    this.setSelected(selected as [ComfyImageData, ComfyImageData]);
  }

  get value() {
    return this._value;
  }

  setSelected(selected: [ComfyImageData, ComfyImageData]) {
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

  draw(ctx: CanvasRenderingContext2D, node: RgthreeImageComparer, width: number, y: number) {
    this.hitAreas = {};
    if (this.value.images.length > 2) {
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      ctx.font = `14px Arial`;
      // Let's calculate the widths of all the labels.
      const drawData: any = [];
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

    if (node.properties?.["comparer_mode"] === "Click") {
      this.drawImage(ctx, this.selected[this.node.isPointerDown ? 1 : 0], y);
    } else {
      this.drawImage(ctx, this.selected[0], y);
      if (node.isPointerOver) {
        this.drawImage(ctx, this.selected[1], y, this.node.pointerOverPos[0]);
      }
    }
  }

  private onSelectionDown(
    event: CanvasMouseEvent,
    pos: Vector2,
    node: LGraphNode,
    bounds?: RgthreeBaseWidgetBounds,
  ) {
    const selected = [...this.selected];
    if (bounds?.data.name.startsWith("A")) {
      selected[0] = bounds.data;
    } else if (bounds?.data.name.startsWith("B")) {
      selected[1] = bounds.data;
    }
    this.setSelected(selected as [ComfyImageData, ComfyImageData]);
  }

  private drawImage(
    ctx: CanvasRenderingContext2D,
    image: ComfyImageData | undefined,
    y: number,
    cropX?: number,
  ) {
    if (!image?.img?.naturalWidth || !image?.img?.naturalHeight) {
      return;
    }
    let [nodeWidth, nodeHeight] = this.node.size as [number, number];
    const imageAspect = image?.img.naturalWidth / image?.img.naturalHeight;
    let height = nodeHeight - y;
    const widgetAspect = nodeWidth / height;
    let targetWidth, targetHeight;
    let offsetX = 0;
    if (imageAspect > widgetAspect) {
      targetWidth = nodeWidth;
      targetHeight = nodeWidth / imageAspect;
    } else {
      targetHeight = height;
      targetWidth = height * imageAspect;
      offsetX = (nodeWidth - targetWidth) / 2;
    }
    const widthMultiplier = image?.img.naturalWidth / targetWidth;

    const sourceX = 0;
    const sourceY = 0;
    const sourceWidth =
      cropX != null ? (cropX - offsetX) * widthMultiplier : image?.img.naturalWidth;
    const sourceHeight = image?.img.naturalHeight;
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
    ctx.drawImage(
      image?.img,
      sourceX,
      sourceY,
      sourceWidth,
      sourceHeight,
      destX,
      destY,
      destWidth,
      destHeight,
    );
    // Shows a label overlayed on the image. Not perfect, keeping commented out.
    // ctx.globalCompositeOperation = "difference";
    // ctx.fillStyle = "rgba(180, 180, 180, 1)";
    // ctx.textAlign = "center";
    // ctx.font = `32px Arial`;
    // ctx.fillText(image.name, nodeWidth / 2, y + 32);
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

  computeSize(width: number): Vector2 {
    return [width, 20];
  }

  override serializeValue(
    node: LGraphNode,
    index: number,
  ): RgthreeImageComparerWidgetValue | Promise<RgthreeImageComparerWidgetValue> {
    const v = [];
    for (const data of this._value.images) {
      // Remove the img since it can't serialize.
      const d = {...data};
      delete d.img;
      v.push(d);
    }
    return {images: v};
  }
}

app.registerExtension({
  name: "rgthree.ImageComparer",
  async beforeRegisterNodeDef(nodeType: typeof LGraphNode, nodeData: ComfyNodeDef) {
    if (nodeData.name === RgthreeImageComparer.type) {
      RgthreeImageComparer.setUp(nodeType, nodeData);
    }
  },
});
