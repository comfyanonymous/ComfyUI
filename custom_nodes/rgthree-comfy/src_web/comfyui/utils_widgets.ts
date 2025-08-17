import type {
  LGraphNode,
  LGraphCanvas as TLGraphCanvas,
  Vector2,
  ICustomWidget,
  IWidgetOptions,
  CanvasPointerEvent,
} from "@comfyorg/frontend";

import {app} from "scripts/app.js";
import {drawNodeWidget, drawWidgetButton, fitString, isLowQuality} from "./utils_canvas.js";

type Vector4 = [number, number, number, number];

/**
 * Draws a label on teft, and a value on the right, ellipsizing when out of space.
 */
export function drawLabelAndValue(
  ctx: CanvasRenderingContext2D,
  label: string,
  value: string,
  width: number,
  posY: number,
  height: number,
  options?: {offsetLeft: number},
) {
  const outerMargin = 15;
  const innerMargin = 10;
  const midY = posY + height / 2;
  ctx.save();
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.fillStyle = LiteGraph.WIDGET_SECONDARY_TEXT_COLOR;
  const labelX = outerMargin + innerMargin + (options?.offsetLeft ?? 0);
  ctx.fillText(label, labelX, midY);

  const valueXLeft = labelX + ctx.measureText(label).width + 7;
  const valueXRight = width - (outerMargin + innerMargin);

  ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
  ctx.textAlign = "right";
  ctx.fillText(fitString(ctx, value, valueXRight - valueXLeft), valueXRight, midY);
  ctx.restore();
}

export type RgthreeBaseWidgetBounds = {
  /** The bounds, either [x, width] assuming the full height, or [x, y, width, height] if height. */
  bounds: Vector2 | Vector4;
  onDown?(
    event: CanvasPointerEvent,
    pos: Vector2,
    node: LGraphNode,
    bounds: RgthreeBaseWidgetBounds,
  ): boolean | void;
  onUp?(
    event: CanvasPointerEvent,
    pos: Vector2,
    node: LGraphNode,
    bounds: RgthreeBaseWidgetBounds,
  ): boolean | void;
  onMove?(
    event: CanvasPointerEvent,
    pos: Vector2,
    node: LGraphNode,
    bounds: RgthreeBaseWidgetBounds,
  ): boolean | void;
  onClick?(
    event: CanvasPointerEvent,
    pos: Vector2,
    node: LGraphNode,
    bounds: RgthreeBaseWidgetBounds,
  ): boolean | void;
  data?: any;
  wasMouseClickedAndIsOver?: boolean;
};

export type RgthreeBaseHitAreas<Keys extends string> = {
  [K in Keys]: RgthreeBaseWidgetBounds;
};

type NotArray<T> = T extends Array<any> ? never : T;

/**
 * A base widget that handles mouse events more properly.
 */
export abstract class RgthreeBaseWidget<V extends ICustomWidget["value"]> implements ICustomWidget {
  // We don't want our value to be an array as a widget will be serialized as an "input" for the API
  // which uses an array value to represent a link. To keep things simpler, we'll avoid using an
  // array at all.
  abstract value: NotArray<V>;

  type: ICustomWidget["type"] = "custom";
  name: string;
  options: IWidgetOptions = {};
  y: number = 0;
  last_y: number = 0;

  protected mouseDowned: Vector2 | null = null;
  protected isMouseDownedAndOver: boolean = false;

  // protected hitAreas: {[key: string]: RgthreeBaseWidgetBounds} = {};
  protected readonly hitAreas: RgthreeBaseHitAreas<any> = {};
  private downedHitAreasForMove: RgthreeBaseWidgetBounds[] = [];
  private downedHitAreasForClick: RgthreeBaseWidgetBounds[] = [];

  constructor(name: string) {
    this.name = name;
  }

  serializeValue(node: LGraphNode, index: number): Promise<V> | V {
    return this.value;
  }

  private clickWasWithinBounds(pos: Vector2, bounds: Vector2 | Vector4) {
    let xStart = bounds[0];
    let xEnd = xStart + (bounds.length > 2 ? bounds[2]! : bounds[1]!);
    const clickedX = pos[0] >= xStart && pos[0] <= xEnd;
    if (bounds.length === 2) {
      return clickedX;
    }
    return clickedX && pos[1] >= bounds[1] && pos[1] <= bounds[1] + bounds[3]!;
  }

  mouse(event: CanvasPointerEvent, pos: Vector2, node: LGraphNode) {
    const canvas = app.canvas as TLGraphCanvas;

    if (event.type == "pointerdown") {
      this.mouseDowned = [...pos] as Vector2;
      this.isMouseDownedAndOver = true;
      this.downedHitAreasForMove.length = 0;
      this.downedHitAreasForClick.length = 0;
      // Loop over out bounds data and call any specifics.
      let anyHandled = false;
      for (const part of Object.values(this.hitAreas)) {
        if (this.clickWasWithinBounds(pos, part.bounds)) {
          if (part.onMove) {
            this.downedHitAreasForMove.push(part);
          }
          if (part.onClick) {
            this.downedHitAreasForClick.push(part);
          }
          if (part.onDown) {
            const thisHandled = part.onDown.apply(this, [event, pos, node, part]);
            anyHandled = anyHandled || thisHandled == true;
          }
          part.wasMouseClickedAndIsOver = true;
        }
      }
      return this.onMouseDown(event, pos, node) ?? anyHandled;
    }

    // This only fires when LiteGraph has a node_widget (meaning it's pressed), but we may not be
    // the original widget pressed, so we still need `mouseDowned`.
    if (event.type == "pointerup") {
      if (!this.mouseDowned) return true;
      this.downedHitAreasForMove.length = 0;
      const wasMouseDownedAndOver = this.isMouseDownedAndOver;
      this.cancelMouseDown();
      let anyHandled = false;
      for (const part of Object.values(this.hitAreas)) {
        if (part.onUp && this.clickWasWithinBounds(pos, part.bounds)) {
          const thisHandled = part.onUp.apply(this, [event, pos, node, part]);
          anyHandled = anyHandled || thisHandled == true;
        }
        part.wasMouseClickedAndIsOver = false;
      }
      for (const part of this.downedHitAreasForClick) {
        if (this.clickWasWithinBounds(pos, part.bounds)) {
          const thisHandled = part.onClick!.apply(this, [event, pos, node, part]);
          anyHandled = anyHandled || thisHandled == true;
        }
      }
      this.downedHitAreasForClick.length = 0;
      if (wasMouseDownedAndOver) {
        const thisHandled = this.onMouseClick(event, pos, node);
        anyHandled = anyHandled || thisHandled == true;
      }
      return this.onMouseUp(event, pos, node) ?? anyHandled;
    }

    // This only fires when LiteGraph has a node_widget (meaning it's pressed).
    if (event.type == "pointermove") {
      this.isMouseDownedAndOver = !!this.mouseDowned;
      // If we've moved off the button while pressing, then consider us no longer pressing.
      if (
        this.mouseDowned &&
        (pos[0] < 15 ||
          pos[0] > node.size[0] - 15 ||
          pos[1] < this.last_y ||
          pos[1] > this.last_y + LiteGraph.NODE_WIDGET_HEIGHT)
      ) {
        this.isMouseDownedAndOver = false;
      }
      for (const part of Object.values(this.hitAreas)) {
        if (this.downedHitAreasForMove.includes(part)) {
          part.onMove!.apply(this, [event, pos, node, part]);
        }
        if (this.downedHitAreasForClick.includes(part)) {
          part.wasMouseClickedAndIsOver = this.clickWasWithinBounds(pos, part.bounds);
        }
      }
      return this.onMouseMove(event, pos, node) ?? true;
    }
    return false;
  }

  /** Sometimes we want to cancel a mouse down, so that an up/move aren't fired. */
  cancelMouseDown() {
    this.mouseDowned = null;
    this.isMouseDownedAndOver = false;
    this.downedHitAreasForMove.length = 0;
  }

  /** An event that fires when the pointer is pressed down (once). */
  onMouseDown(event: CanvasPointerEvent, pos: Vector2, node: LGraphNode): boolean | void {
    return;
  }

  /**
   * An event that fires when the pointer is let go. Only fires if this was the widget that was
   * originally pressed down.
   */
  onMouseUp(event: CanvasPointerEvent, pos: Vector2, node: LGraphNode): boolean | void {
    return;
  }

  /**
   * An event that fires when the pointer is let go _over the widget_ and when the widget that was
   * originally pressed down.
   */
  onMouseClick(event: CanvasPointerEvent, pos: Vector2, node: LGraphNode): boolean | void {
    return;
  }

  /**
   * An event that fires when the pointer is moving after pressing down. Will fire both on and off
   * of the widget. Check `isMouseDownedAndOver` to determine if the mouse is currently over the
   * widget or not.
   */
  onMouseMove(event: CanvasPointerEvent, pos: Vector2, node: LGraphNode): boolean | void {
    return;
  }
}

/**
 * A better implementation of the LiteGraph button widget.
 */
export class RgthreeBetterButtonWidget extends RgthreeBaseWidget<string> {
  override readonly type = "custom";

  value: string = "";
  label: string = "";
  mouseClickCallback: (event: CanvasPointerEvent, pos: Vector2, node: LGraphNode) => boolean | void;

  constructor(
    name: string,
    mouseClickCallback: (event: CanvasPointerEvent, pos: Vector2, node: LGraphNode) => boolean | void,
    label?: string,
  ) {
    super(name);
    this.mouseClickCallback = mouseClickCallback;
    this.label = label || name;
  }

  draw(ctx: CanvasRenderingContext2D, node: LGraphNode, width: number, y: number, height: number) {
    drawWidgetButton(
      ctx,
      {size: [width - 30, height], pos: [15, y]},
      this.label,
      this.isMouseDownedAndOver,
    );
  }

  override onMouseClick(event: CanvasPointerEvent, pos: Vector2, node: LGraphNode) {
    return this.mouseClickCallback(event, pos, node);
  }
}

/**
 * A better implementation of the LiteGraph text widget, including auto ellipsis.
 */
export class RgthreeBetterTextWidget extends RgthreeBaseWidget<string> {
  value: string;

  constructor(name: string, value: string) {
    super(name);
    this.name = name;
    this.value = value;
  }

  draw(ctx: CanvasRenderingContext2D, node: LGraphNode, width: number, y: number, height: number) {
    const widgetData = drawNodeWidget(ctx, {size: [width, height], pos: [15, y]});

    if (!widgetData.lowQuality) {
      drawLabelAndValue(ctx, this.name, this.value, width, y, height);
    }
  }

  override mouse(event: CanvasPointerEvent, pos: Vector2, node: LGraphNode): boolean {
    const canvas = app.canvas as TLGraphCanvas;
    if (event.type == "pointerdown") {
      canvas.prompt("Label", this.value, (v: string) => (this.value = v), event);
      return true;
    }
    return false;
  }
}

/**
 * Options for the Divider Widget.
 */
type RgthreeDividerWidgetOptions = {
  marginTop: number;
  marginBottom: number;
  marginLeft: number;
  marginRight: number;
  color: string;
  thickness: number;
};

/**
 * A divider widget; can also be used as a spacer if fed a 0 thickness.
 */
export class RgthreeDividerWidget extends RgthreeBaseWidget<{}> {
  override value = {};
  override options = {serialize: false};
  override readonly type = "custom";

  private readonly widgetOptions: RgthreeDividerWidgetOptions = {
    marginTop: 7,
    marginBottom: 7,
    marginLeft: 15,
    marginRight: 15,
    color: LiteGraph.WIDGET_OUTLINE_COLOR,
    thickness: 1,
  };

  constructor(widgetOptions?: Partial<RgthreeDividerWidgetOptions>) {
    super("divider");
    Object.assign(this.widgetOptions, widgetOptions || {});
  }

  draw(ctx: CanvasRenderingContext2D, node: LGraphNode, width: number, posY: number, h: number) {
    if (this.widgetOptions.thickness) {
      ctx.strokeStyle = this.widgetOptions.color;
      const x = this.widgetOptions.marginLeft;
      const y = posY + this.widgetOptions.marginTop;
      const w = width - this.widgetOptions.marginLeft - this.widgetOptions.marginRight;
      ctx.stroke(new Path2D(`M ${x} ${y} h ${w}`));
    }
  }

  computeSize(width: number): [number, number] {
    return [
      width,
      this.widgetOptions.marginTop + this.widgetOptions.marginBottom + this.widgetOptions.thickness,
    ];
  }
}

/**
 * Options for the Label Widget.
 */
export type RgthreeLabelWidgetOptions = {
  align?: "left" | "center" | "right";
  color?: string;
  italic?: boolean;
  size?: number;
  text?: string | (() => string); // Text, or fall back to the name.

  /** A label to put on the right side. */
  actionLabel?: "__PLUS_ICON__" | string;
  actionCallback?: (event: PointerEvent | CanvasPointerEvent) => void;
};

/**
 * A simple label widget, drawn with no background.
 */
export class RgthreeLabelWidget extends RgthreeBaseWidget<string> {
  override readonly type = "custom";
  override options = {serialize: false};
  value = "";

  private readonly widgetOptions: RgthreeLabelWidgetOptions = {};
  private posY: number = 0;

  constructor(name: string, widgetOptions?: RgthreeLabelWidgetOptions) {
    super(name);
    Object.assign(this.widgetOptions, widgetOptions);
  }

  update(widgetOptions: RgthreeLabelWidgetOptions) {
    Object.assign(this.widgetOptions, widgetOptions);
  }

  draw(
    ctx: CanvasRenderingContext2D,
    node: LGraphNode,
    width: number,
    posY: number,
    height: number,
  ) {
    this.posY = posY;
    ctx.save();

    let text = this.widgetOptions.text ?? this.name;
    if (typeof text === "function") {
      text = text();
    }

    ctx.textAlign = this.widgetOptions.align || "left";
    ctx.fillStyle = this.widgetOptions.color || LiteGraph.WIDGET_TEXT_COLOR;
    const oldFont = ctx.font;
    if (this.widgetOptions.italic) {
      ctx.font = "italic " + ctx.font;
    }
    if (this.widgetOptions.size) {
      ctx.font = ctx.font.replace(/\d+px/, `${this.widgetOptions.size}px`);
    }

    const midY = posY + height / 2;
    ctx.textBaseline = "middle";

    if (this.widgetOptions.align === "center") {
      ctx.fillText(text, node.size[0] / 2, midY);
    } else {
      ctx.fillText(text, 15, midY);
    } // TODO(right);

    ctx.font = oldFont;

    if (this.widgetOptions.actionLabel === "__PLUS_ICON__") {
      const plus = new Path2D(
        `M${node.size[0] - 15 - 2} ${posY + 7} v4 h-4 v4 h-4 v-4 h-4 v-4 h4 v-4 h4 v4 h4 z`,
      );
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.fillStyle = "#3a3";
      ctx.strokeStyle = "#383";
      ctx.fill(plus);
      ctx.stroke(plus);
    }
    ctx.restore();
  }

  override mouse(event: CanvasPointerEvent, nodePos: Vector2, node: LGraphNode): boolean {
    if (
      event.type !== "pointerdown" ||
      isLowQuality() ||
      !this.widgetOptions.actionLabel ||
      !this.widgetOptions.actionCallback
    ) {
      return false;
    }

    const pos: Vector2 = [nodePos[0], nodePos[1] - this.posY];
    const rightX = node.size[0] - 15;
    if (pos[0] > rightX || pos[0] < rightX - 16) {
      return false;
    }
    this.widgetOptions.actionCallback(event);
    return true;
  }
}

/** An invisible widget. */
export class RgthreeInvisibleWidget<T extends ICustomWidget["value"]> extends RgthreeBaseWidget<T> {
  override readonly type = "custom";

  value: NotArray<T>;
  private serializeValueFn?: (node: LGraphNode, index: number) => Promise<T> | T;

  constructor(
    name: string,
    type: string,
    value: NotArray<T>,
    serializeValueFn?: (node: LGraphNode, index: number) => Promise<T> | T,
  ) {
    super(name);
    // this.type = type;
    this.value = value;
    this.serializeValueFn = serializeValueFn;
  }

  draw() {
    return;
  }
  computeSize(width: number): Vector2 {
    return [0, 0];
  }

  override serializeValue(node: LGraphNode, index: number): T | Promise<T> {
    return this.serializeValueFn != null
      ? this.serializeValueFn(node, index)
      : super.serializeValue(node, index);
  }
}
