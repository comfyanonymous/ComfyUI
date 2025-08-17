import type {
  LGraphNode as TLGraphNode,
  LGraphCanvas,
  Vector2,
  IContextMenuValue,
  IFoundSlot,
  CanvasMouseEvent,
  ISerialisedNode,
  ICustomWidget,
  CanvasPointerEvent,
} from "@comfyorg/frontend";
import type {ComfyNodeDef} from "typings/comfy.js";
import type {RgthreeModelInfo} from "typings/rgthree.js";

import {app} from "scripts/app.js";
import {RgthreeBaseServerNode} from "./base_node.js";
import {rgthree} from "./rgthree.js";
import {addConnectionLayoutSupport} from "./utils.js";
import {NodeTypesString} from "./constants.js";
import {
  drawInfoIcon,
  drawNumberWidgetPart,
  drawRoundedRectangle,
  drawTogglePart,
  fitString,
  isLowQuality,
} from "./utils_canvas.js";
import {
  RgthreeBaseHitAreas,
  RgthreeBaseWidget,
  RgthreeBetterButtonWidget,
  RgthreeDividerWidget,
} from "./utils_widgets.js";
import {rgthreeApi} from "rgthree/common/rgthree_api.js";
import {showLoraChooser} from "./utils_menu.js";
import {moveArrayItem, removeArrayItem} from "rgthree/common/shared_utils.js";
import {RgthreeLoraInfoDialog} from "./dialog_info.js";
import {LORA_INFO_SERVICE} from "rgthree/common/model_info_service.js";
// import { RgthreePowerLoraChooserDialog } from "./dialog_power_lora_chooser.js";

const PROP_LABEL_SHOW_STRENGTHS = "Show Strengths";
const PROP_LABEL_SHOW_STRENGTHS_STATIC = `@${PROP_LABEL_SHOW_STRENGTHS}`;
const PROP_VALUE_SHOW_STRENGTHS_SINGLE = "Single Strength";
const PROP_VALUE_SHOW_STRENGTHS_SEPARATE = "Separate Model & Clip";

/**
 * The Power Lora Loader is a super-simply Lora Loader node that can load multiple Loras at once
 * in an ultra-condensed node allowing fast toggling, and advanced strength setting.
 */
class RgthreePowerLoraLoader extends RgthreeBaseServerNode {
  static override title = NodeTypesString.POWER_LORA_LOADER;
  static override type = NodeTypesString.POWER_LORA_LOADER;
  static comfyClass = NodeTypesString.POWER_LORA_LOADER;

  override serialize_widgets = true;

  private logger = rgthree.newLogSession(`[Power Lora Stack]`);

  static [PROP_LABEL_SHOW_STRENGTHS_STATIC] = {
    type: "combo",
    values: [PROP_VALUE_SHOW_STRENGTHS_SINGLE, PROP_VALUE_SHOW_STRENGTHS_SEPARATE],
  };

  /** Counts the number of lora widgets. This is used to give unique names.  */
  private loraWidgetsCounter = 0;

  /** Keep track of the spacer, new lora widgets will go before it when it exists. */
  private widgetButtonSpacer: ICustomWidget | null = null;

  constructor(title = NODE_CLASS.title) {
    super(title);

    this.properties[PROP_LABEL_SHOW_STRENGTHS] = PROP_VALUE_SHOW_STRENGTHS_SINGLE;

    // Prefetch loras list.
    rgthreeApi.getLoras();
  }

  /**
   * Handles configuration from a saved workflow by first removing our default widgets that were
   * added in `onNodeCreated`, letting `super.configure` and do nothing, then create our lora
   * widgets and, finally, add back in our default widgets.
   */
  override configure(info: ISerialisedNode): void {
    while (this.widgets?.length) this.removeWidget(0);
    this.widgetButtonSpacer = null;
    super.configure(info);

    (this as any)._tempWidth = this.size[0];
    (this as any)._tempHeight = this.size[1];
    for (const widgetValue of info.widgets_values || []) {
      if ((widgetValue as PowerLoraLoaderWidgetValue)?.lora !== undefined) {
        const widget = this.addNewLoraWidget();
        widget.value = {...(widgetValue as PowerLoraLoaderWidgetValue)};
      }
    }
    this.addNonLoraWidgets();
    this.size[0] = (this as any)._tempWidth;
    this.size[1] = Math.max((this as any)._tempHeight, this.computeSize()[1]);
  }

  /**
   * Adds the non-lora widgets. If we'll be configured then we remove them and add them back, so
   * this is really only for newly created nodes in the current session.
   */
  override onNodeCreated() {
    super.onNodeCreated?.();
    this.addNonLoraWidgets();
    const computed = this.computeSize();
    this.size = this.size || [0, 0];
    this.size[0] = Math.max(this.size[0], computed[0]);
    this.size[1] = Math.max(this.size[1], computed[1]);
    this.setDirtyCanvas(true, true);
  }

  /** Adds a new lora widget in the proper slot. */
  private addNewLoraWidget(lora?: string) {
    this.loraWidgetsCounter++;
    const widget = this.addCustomWidget(
      new PowerLoraLoaderWidget("lora_" + this.loraWidgetsCounter),
    ) as PowerLoraLoaderWidget;
    if (lora) widget.setLora(lora);
    if (this.widgetButtonSpacer) {
      moveArrayItem(this.widgets, widget, this.widgets.indexOf(this.widgetButtonSpacer));
    }

    return widget;
  }

  /** Adds the non-lora widgets around any lora ones that may be there from configuration. */
  private addNonLoraWidgets() {
    moveArrayItem(
      this.widgets,
      this.addCustomWidget(new RgthreeDividerWidget({marginTop: 4, marginBottom: 0, thickness: 0})),
      0,
    );
    moveArrayItem(this.widgets, this.addCustomWidget(new PowerLoraLoaderHeaderWidget()), 1);

    this.widgetButtonSpacer = this.addCustomWidget(
      new RgthreeDividerWidget({marginTop: 4, marginBottom: 0, thickness: 0}),
    ) as RgthreeDividerWidget;

    this.addCustomWidget(
      new RgthreeBetterButtonWidget(
        "➕ Add Lora",
        (event: CanvasMouseEvent, pos: Vector2, node: TLGraphNode) => {
          rgthreeApi.getLoras().then((lorasDetails) => {
            const loras = lorasDetails.map((l) => l.file);
            showLoraChooser(
              event as MouseEvent,
              (value: IContextMenuValue | string) => {
                if (typeof value === "string") {
                  if (value.includes("Power Lora Chooser")) {
                    // new RgthreePowerLoraChooserDialog().show();
                  } else if (value !== "NONE") {
                    this.addNewLoraWidget(value);
                    const computed = this.computeSize();
                    const tempHeight = (this as any)._tempHeight ?? 15;
                    this.size[1] = Math.max(tempHeight, computed[1]);
                    this.setDirtyCanvas(true, true);
                  }
                }
                // }, null, ["⚡️ Power Lora Chooser", ...loras]);
              },
              null,
              [...loras],
            );
          });
          return true;
        },
      ),
    );
  }

  /**
   * Hacks the `getSlotInPosition` call made from LiteGraph so we can show a custom context menu
   * for widgets.
   *
   * Normally this method, called from LiteGraph's processContextMenu, will only get Inputs or
   * Outputs. But that's not good enough because we we also want to provide a custom menu when
   * clicking a widget for this node... so we are left to HACK once again!
   *
   * To achieve this:
   *  - Here, in LiteGraph's processContextMenu it asks the clicked node to tell it which input or
   *    output the user clicked on in `getSlotInPosition`
   *  - We check, and if we didn't, then we see if we clicked a widget and, if so, pass back some
   *    data that looks like we clicked an output to fool LiteGraph like a silly child.
   *  - As LiteGraph continues in its `processContextMenu`, it will then immediately call
   *    the clicked node's `getSlotMenuOptions` when `getSlotInPosition` returns data.
   *  - So, just below, we can then give LiteGraph the ContextMenu options we have.
   *
   * The only issue is that LiteGraph also checks `input/output.type` to set the ContextMenu title,
   * so we need to supply that property (and set it to what we want our title). Otherwise, this
   * should be pretty clean.
   */
  override getSlotInPosition(canvasX: number, canvasY: number): any {
    const slot = super.getSlotInPosition(canvasX, canvasY);
    // No slot, let's see if it's a widget.
    if (!slot) {
      let lastWidget = null;
      for (const widget of this.widgets) {
        // If last_y isn't set, something is wrong. Bail.
        if (!widget.last_y) return;
        if (canvasY > this.pos[1] + widget.last_y) {
          lastWidget = widget;
          continue;
        }
        break;
      }
      // Only care about lora widget clicks.
      if (lastWidget?.name?.startsWith("lora_")) {
        return {widget: lastWidget, output: {type: "LORA WIDGET"}};
      }
    }
    return slot;
  }

  /**
   * Working with the overridden `getSlotInPosition` above, this method checks if the passed in
   * option is actually a widget from it and then hijacks the context menu all together.
   */
  override getSlotMenuOptions(slot: IFoundSlot) {
    // Oddly, LiteGraph doesn't call back into our node with a custom menu (even though it let's us
    // define a custom menu to begin with... wtf?). So, we'll return null so the default is not
    // triggered and then we'll just show one ourselves because.. yea.
    if (slot?.widget?.name?.startsWith("lora_")) {
      const widget = slot.widget as PowerLoraLoaderWidget;
      const index = this.widgets.indexOf(widget);
      const canMoveUp = !!this.widgets[index - 1]?.name?.startsWith("lora_");
      const canMoveDown = !!this.widgets[index + 1]?.name?.startsWith("lora_");
      const menuItems: (IContextMenuValue | null)[] = [
        {
          content: `ℹ️ Show Info`,
          callback: () => {
            widget.showLoraInfoDialog();
          },
        },
        null, // Divider
        {
          content: `${widget.value.on ? "⚫" : "🟢"} Toggle ${widget.value.on ? "Off" : "On"}`,
          callback: () => {
            widget.value.on = !widget.value.on;
          },
        },
        {
          content: `⬆️ Move Up`,
          disabled: !canMoveUp,
          callback: () => {
            moveArrayItem(this.widgets, widget, index - 1);
          },
        },
        {
          content: `⬇️ Move Down`,
          disabled: !canMoveDown,
          callback: () => {
            moveArrayItem(this.widgets, widget, index + 1);
          },
        },
        {
          content: `🗑️ Remove`,
          callback: () => {
            removeArrayItem(this.widgets, widget);
          },
        },
      ];
      new LiteGraph.ContextMenu(menuItems, {
        title: "LORA WIDGET",
        event: rgthree.lastCanvasMouseEvent!,
      });

      // [🤮] ComfyUI doesn't have a possible return type as falsy, even though the impl skips the
      // menu when the return is falsy. Casting as any.
      return undefined as any;
    }
    return this.defaultGetSlotMenuOptions(slot);
  }

  /**
   * When `refreshComboInNode` is called from ComfyUI, then we'll kick off a fresh loras fetch.
   */
  override refreshComboInNode(defs: any) {
    rgthreeApi.getLoras(true);
  }

  /**
   * Returns true if there are any Lora Widgets. Useful for widgets to ask as they render.
   */
  hasLoraWidgets() {
    return !!this.widgets?.find((w) => w.name?.startsWith("lora_"));
  }

  /**
   * This will return true when all lora widgets are on, false when all are off, or null if it's
   * mixed.
   */
  allLorasState() {
    let allOn = true;
    let allOff = true;
    for (const widget of this.widgets) {
      if (widget.name?.startsWith("lora_")) {
        const on = (widget.value as any)?.on;
        allOn = allOn && on === true;
        allOff = allOff && on === false;
        if (!allOn && !allOff) {
          return null;
        }
      }
    }
    return allOn && this.widgets?.length ? true : false;
  }

  /**
   * Toggles all the loras on or off.
   */
  toggleAllLoras() {
    const allOn = this.allLorasState();
    const toggledTo = !allOn ? true : false;
    for (const widget of this.widgets) {
      if (widget.name?.startsWith("lora_") && (widget.value as any)?.on != null) {
        (widget.value as any).on = toggledTo;
      }
    }
  }

  static override setUp(comfyClass: typeof LGraphNode, nodeData: ComfyNodeDef) {
    RgthreeBaseServerNode.registerForOverride(comfyClass, nodeData, NODE_CLASS);
  }

  static override onRegisteredForOverride(comfyClass: any, ctxClass: any) {
    addConnectionLayoutSupport(NODE_CLASS, app, [
      ["Left", "Right"],
      ["Right", "Left"],
    ]);
    setTimeout(() => {
      NODE_CLASS.category = comfyClass.category;
    });
  }

  override getHelp() {
    return `
      <p>
        The ${this.type!.replace("(rgthree)", "")} is a powerful node that condenses 100s of pixels
        of functionality in a single, dynamic node that allows you to add loras, change strengths,
        and quickly toggle on/off all without taking up half your screen.
      </p>
      <ul>
        <li><p>
          Add as many Lora's as you would like by clicking the "+ Add Lora" button.
          There's no real limit!
        </p></li>
        <li><p>
          Right-click on a Lora widget for special options to move the lora up or down
          (no image affect, only presentational), toggle it on/off, or delete the row all together.
        </p></li>
        <li>
          <p>
            <strong>Properties.</strong> You can change the following properties (by right-clicking
            on the node, and select "Properties" or "Properties Panel" from the menu):
          </p>
          <ul>
            <li><p>
              <code>${PROP_LABEL_SHOW_STRENGTHS}</code> - Change between showing a single, simple
              strength (which will be used for both model and clip), or a more advanced view with
              both model and clip strengths being modifiable.
            </p></li>
          </ul>
        </li>
      </ul>`;
  }
}

/**
 * The PowerLoraLoaderHeaderWidget that renders a toggle all switch, as well as some title info
 * (more necessary for the double model & clip strengths to label them).
 */
class PowerLoraLoaderHeaderWidget extends RgthreeBaseWidget<{type: string}> {
  override value = {type: "PowerLoraLoaderHeaderWidget"};
  override readonly type = "custom";

  protected override hitAreas: RgthreeBaseHitAreas<"toggle"> = {
    toggle: {bounds: [0, 0] as Vector2, onDown: this.onToggleDown},
  };

  private showModelAndClip: boolean | null = null;

  constructor(name: string = "PowerLoraLoaderHeaderWidget") {
    super(name);
  }

  draw(
    ctx: CanvasRenderingContext2D,
    node: RgthreePowerLoraLoader,
    w: number,
    posY: number,
    height: number,
  ) {
    if (!node.hasLoraWidgets()) {
      return;
    }
    // Since draw is the loop that runs, this is where we'll check the property state (rather than
    // expect the node to tell us it's state etc).
    this.showModelAndClip =
      node.properties[PROP_LABEL_SHOW_STRENGTHS] === PROP_VALUE_SHOW_STRENGTHS_SEPARATE;
    const margin = 10;
    const innerMargin = margin * 0.33;
    const lowQuality = isLowQuality();
    const allLoraState = node.allLorasState();

    // Move slightly down. We don't have a border and this feels a bit nicer.
    posY += 2;
    const midY = posY + height * 0.5;
    let posX = 10;
    ctx.save();
    this.hitAreas.toggle.bounds = drawTogglePart(ctx, {posX, posY, height, value: allLoraState});

    if (!lowQuality) {
      posX += this.hitAreas.toggle.bounds[1] + innerMargin;

      ctx.globalAlpha = app.canvas.editor_alpha * 0.55;
      ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText("Toggle All", posX, midY);

      let rposX = node.size[0] - margin - innerMargin - innerMargin;
      ctx.textAlign = "center";
      ctx.fillText(
        this.showModelAndClip ? "Clip" : "Strength",
        rposX - drawNumberWidgetPart.WIDTH_TOTAL / 2,
        midY,
      );
      if (this.showModelAndClip) {
        rposX = rposX - drawNumberWidgetPart.WIDTH_TOTAL - innerMargin * 2;
        ctx.fillText("Model", rposX - drawNumberWidgetPart.WIDTH_TOTAL / 2, midY);
      }
    }
    ctx.restore();
  }

  /**
   * Handles a pointer down on the toggle's defined hit area.
   */
  onToggleDown(event: CanvasMouseEvent, pos: Vector2, node: TLGraphNode) {
    (node as RgthreePowerLoraLoader).toggleAllLoras();
    this.cancelMouseDown();
    return true;
  }
}

const DEFAULT_LORA_WIDGET_DATA: PowerLoraLoaderWidgetValue = {
  on: true,
  lora: null as string | null,
  strength: 1,
  strengthTwo: null as number | null,
};

type PowerLoraLoaderWidgetValue = {
  on: boolean;
  lora: string | null;
  strength: number;
  strengthTwo: number | null;
};

/**
 * The PowerLoaderWidget that combines several custom drawing and functionality in a single row.
 */
class PowerLoraLoaderWidget extends RgthreeBaseWidget<PowerLoraLoaderWidgetValue> {
  override readonly type = "custom";

  /** Whether the strength has changed with mouse move (to cancel mouse up). */
  private haveMouseMovedStrength = false;
  private loraInfoPromise: Promise<RgthreeModelInfo | null> | null = null;
  private loraInfo: RgthreeModelInfo | null = null;

  private showModelAndClip: boolean | null = null;

  protected override hitAreas: RgthreeBaseHitAreas<
    | "toggle"
    | "lora"
    // | "info"
    | "strengthDec"
    | "strengthVal"
    | "strengthInc"
    | "strengthAny"
    | "strengthTwoDec"
    | "strengthTwoVal"
    | "strengthTwoInc"
    | "strengthTwoAny"
  > = {
    toggle: {bounds: [0, 0] as Vector2, onDown: this.onToggleDown},
    lora: {bounds: [0, 0] as Vector2, onClick: this.onLoraClick},
    // info: { bounds: [0, 0] as Vector2, onDown: this.onInfoDown },

    strengthDec: {bounds: [0, 0] as Vector2, onClick: this.onStrengthDecDown},
    strengthVal: {bounds: [0, 0] as Vector2, onClick: this.onStrengthValUp},
    strengthInc: {bounds: [0, 0] as Vector2, onClick: this.onStrengthIncDown},
    strengthAny: {bounds: [0, 0] as Vector2, onMove: this.onStrengthAnyMove},

    strengthTwoDec: {bounds: [0, 0] as Vector2, onClick: this.onStrengthTwoDecDown},
    strengthTwoVal: {bounds: [0, 0] as Vector2, onClick: this.onStrengthTwoValUp},
    strengthTwoInc: {bounds: [0, 0] as Vector2, onClick: this.onStrengthTwoIncDown},
    strengthTwoAny: {bounds: [0, 0] as Vector2, onMove: this.onStrengthTwoAnyMove},
  };

  constructor(name: string) {
    super(name);
  }

  private _value = {
    on: true,
    lora: null as string | null,
    strength: 1,
    strengthTwo: null as number | null,
  };

  set value(v) {
    this._value = v;
    // In case widgets are messed up, we can correct course here.
    if (typeof this._value !== "object") {
      this._value = {...DEFAULT_LORA_WIDGET_DATA};
      if (this.showModelAndClip) {
        this._value.strengthTwo = this._value.strength;
      }
    }
    this.getLoraInfo();
  }

  get value() {
    return this._value;
  }

  setLora(lora: string) {
    this._value.lora = lora;
    this.getLoraInfo();
  }

  /** Draws our widget with a toggle, lora selector, and number selector all in a single row. */
  draw(ctx: CanvasRenderingContext2D, node: TLGraphNode, w: number, posY: number, height: number) {
    // Since draw is the loop that runs, this is where we'll check the property state (rather than
    // expect the node to tell us it's state etc).
    let currentShowModelAndClip =
      node.properties[PROP_LABEL_SHOW_STRENGTHS] === PROP_VALUE_SHOW_STRENGTHS_SEPARATE;
    if (this.showModelAndClip !== currentShowModelAndClip) {
      let oldShowModelAndClip = this.showModelAndClip;
      this.showModelAndClip = currentShowModelAndClip;
      if (this.showModelAndClip) {
        // If we're setting show both AND we're not null, then re-set to the current strength.
        if (oldShowModelAndClip != null) {
          this.value.strengthTwo = this.value.strength ?? 1;
        }
      } else {
        this.value.strengthTwo = null;
        this.hitAreas.strengthTwoDec.bounds = [0, -1];
        this.hitAreas.strengthTwoVal.bounds = [0, -1];
        this.hitAreas.strengthTwoInc.bounds = [0, -1];
        this.hitAreas.strengthTwoAny.bounds = [0, -1];
      }
    }

    ctx.save();
    const margin = 10;
    const innerMargin = margin * 0.33;
    const lowQuality = isLowQuality();
    const midY = posY + height * 0.5;

    // We'll move posX along as we draw things.
    let posX = margin;

    // Draw the background.
    drawRoundedRectangle(ctx, {pos: [posX, posY], size: [node.size[0] - margin * 2, height]});

    // Draw the toggle
    this.hitAreas.toggle.bounds = drawTogglePart(ctx, {posX, posY, height, value: this.value.on});
    posX += this.hitAreas.toggle.bounds[1] + innerMargin;

    // If low quality, then we're done rendering.
    if (lowQuality) {
      ctx.restore();
      return;
    }

    // If we're not toggled on, then make everything after faded.
    if (!this.value.on) {
      ctx.globalAlpha = app.canvas.editor_alpha * 0.4;
    }

    ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;

    // Now, we draw the strength number part on the right, so we know the width of it to draw the
    // lora label as flexible.
    let rposX = node.size[0] - margin - innerMargin - innerMargin;

    const strengthValue = this.showModelAndClip
      ? (this.value.strengthTwo ?? 1)
      : (this.value.strength ?? 1);

    let textColor: string | undefined = undefined;
    if (this.loraInfo?.strengthMax != null && strengthValue > this.loraInfo?.strengthMax) {
      textColor = "#c66";
    } else if (this.loraInfo?.strengthMin != null && strengthValue < this.loraInfo?.strengthMin) {
      textColor = "#c66";
    }

    const [leftArrow, text, rightArrow] = drawNumberWidgetPart(ctx, {
      posX: node.size[0] - margin - innerMargin - innerMargin,
      posY,
      height,
      value: strengthValue,
      direction: -1,
      textColor,
    });

    this.hitAreas.strengthDec.bounds = leftArrow;
    this.hitAreas.strengthVal.bounds = text;
    this.hitAreas.strengthInc.bounds = rightArrow;
    this.hitAreas.strengthAny.bounds = [leftArrow[0], rightArrow[0] + rightArrow[1] - leftArrow[0]];

    rposX = leftArrow[0] - innerMargin;

    if (this.showModelAndClip) {
      rposX -= innerMargin;
      // If we're showing both, then the rightmost we just drew is our "strengthTwo", so reset and
      // then draw our model ("strength" one) to the left.
      this.hitAreas.strengthTwoDec.bounds = this.hitAreas.strengthDec.bounds;
      this.hitAreas.strengthTwoVal.bounds = this.hitAreas.strengthVal.bounds;
      this.hitAreas.strengthTwoInc.bounds = this.hitAreas.strengthInc.bounds;
      this.hitAreas.strengthTwoAny.bounds = this.hitAreas.strengthAny.bounds;

      let textColor: string | undefined = undefined;
      if (this.loraInfo?.strengthMax != null && this.value.strength > this.loraInfo?.strengthMax) {
        textColor = "#c66";
      } else if (
        this.loraInfo?.strengthMin != null &&
        this.value.strength < this.loraInfo?.strengthMin
      ) {
        textColor = "#c66";
      }
      const [leftArrow, text, rightArrow] = drawNumberWidgetPart(ctx, {
        posX: rposX,
        posY,
        height,
        value: this.value.strength ?? 1,
        direction: -1,
        textColor,
      });
      this.hitAreas.strengthDec.bounds = leftArrow;
      this.hitAreas.strengthVal.bounds = text;
      this.hitAreas.strengthInc.bounds = rightArrow;
      this.hitAreas.strengthAny.bounds = [
        leftArrow[0],
        rightArrow[0] + rightArrow[1] - leftArrow[0],
      ];
      rposX = leftArrow[0] - innerMargin;
    }

    const infoIconSize = height * 0.66;
    const infoWidth = infoIconSize + innerMargin + innerMargin;
    // Draw an info emoji; if checks if it's enabled (to quickly turn it on or off)
    if ((this.hitAreas as any)["info"]) {
      rposX -= innerMargin;
      drawInfoIcon(ctx, rposX - infoIconSize, posY + (height - infoIconSize) / 2, infoIconSize);
      // ctx.fillText('ℹ', posX, midY);
      (this.hitAreas as any).info.bounds = [rposX - infoIconSize, infoWidth];
      rposX = rposX - infoIconSize - innerMargin;
    }

    // Draw lora label
    const loraWidth = rposX - posX;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    const loraLabel = String(this.value?.lora || "None");
    ctx.fillText(fitString(ctx, loraLabel, loraWidth), posX, midY);

    this.hitAreas.lora.bounds = [posX, loraWidth];
    posX += loraWidth + innerMargin;

    ctx.globalAlpha = app.canvas.editor_alpha;
    ctx.restore();
  }

  override serializeValue(
    node: TLGraphNode,
    index: number,
  ): PowerLoraLoaderWidgetValue | Promise<PowerLoraLoaderWidgetValue> {
    const v = {...this.value};
    // Never send the second value to the backend if we're not showing it, otherwise, let's just
    // make sure it's not null.
    if (!this.showModelAndClip) {
      delete (v as any).strengthTwo;
    } else {
      this.value.strengthTwo = this.value.strengthTwo ?? 1;
      v.strengthTwo = this.value.strengthTwo;
    }
    return v;
  }

  onToggleDown(event: CanvasMouseEvent, pos: Vector2, node: TLGraphNode) {
    this.value.on = !this.value.on;
    this.cancelMouseDown(); // Clear the down since we handle it.
    return true;
  }

  onInfoDown(event: CanvasMouseEvent, pos: Vector2, node: TLGraphNode) {
    this.showLoraInfoDialog();
  }

  onLoraClick(event: CanvasMouseEvent, pos: Vector2, node: TLGraphNode) {
    showLoraChooser(event, (value: IContextMenuValue) => {
      if (typeof value === "string") {
        this.value.lora = value;
        this.loraInfo = null;
        this.getLoraInfo();
      }
      node.setDirtyCanvas(true, true);
    });
    this.cancelMouseDown();
  }

  onStrengthDecDown(event: CanvasMouseEvent, pos: Vector2, node: TLGraphNode) {
    this.stepStrength(-1, false);
  }
  onStrengthIncDown(event: CanvasMouseEvent, pos: Vector2, node: TLGraphNode) {
    this.stepStrength(1, false);
  }
  onStrengthTwoDecDown(event: CanvasMouseEvent, pos: Vector2, node: TLGraphNode) {
    this.stepStrength(-1, true);
  }
  onStrengthTwoIncDown(event: CanvasMouseEvent, pos: Vector2, node: TLGraphNode) {
    this.stepStrength(1, true);
  }

  onStrengthAnyMove(event: CanvasMouseEvent, pos: Vector2, node: TLGraphNode) {
    this.doOnStrengthAnyMove(event, false);
  }

  onStrengthTwoAnyMove(event: CanvasMouseEvent, pos: Vector2, node: TLGraphNode) {
    this.doOnStrengthAnyMove(event, true);
  }

  private doOnStrengthAnyMove(event: CanvasMouseEvent, isTwo = false) {
    if (event.deltaX) {
      let prop: "strengthTwo" | "strength" = isTwo ? "strengthTwo" : "strength";
      this.haveMouseMovedStrength = true;
      this.value[prop] = (this.value[prop] ?? 1) + event.deltaX * 0.05;
    }
  }

  onStrengthValUp(event: CanvasPointerEvent, pos: Vector2, node: TLGraphNode) {
    this.doOnStrengthValUp(event, false);
  }

  onStrengthTwoValUp(event: CanvasPointerEvent, pos: Vector2, node: TLGraphNode) {
    this.doOnStrengthValUp(event, true);
  }

  private doOnStrengthValUp(event: CanvasPointerEvent, isTwo = false) {
    if (this.haveMouseMovedStrength) return;
    let prop: "strengthTwo" | "strength" = isTwo ? "strengthTwo" : "strength";
    const canvas = app.canvas as LGraphCanvas;
    canvas.prompt("Value", this.value[prop], (v: string) => (this.value[prop] = Number(v)), event);
  }

  override onMouseUp(event: CanvasPointerEvent, pos: Vector2, node: TLGraphNode): boolean | void {
    super.onMouseUp(event, pos, node);
    this.haveMouseMovedStrength = false;
  }

  showLoraInfoDialog() {
    if (!this.value.lora || this.value.lora === "None") {
      return;
    }
    const infoDialog = new RgthreeLoraInfoDialog(this.value.lora).show();
    infoDialog.addEventListener("close", ((e: CustomEvent<{dirty: boolean}>) => {
      if (e.detail.dirty) {
        this.getLoraInfo(true);
      }
    }) as EventListener);
  }

  private stepStrength(direction: -1 | 1, isTwo = false) {
    let step = 0.05;
    let prop: "strengthTwo" | "strength" = isTwo ? "strengthTwo" : "strength";
    let strength = (this.value[prop] ?? 1) + step * direction;
    this.value[prop] = Math.round(strength * 100) / 100;
  }

  private getLoraInfo(force = false) {
    if (!this.loraInfoPromise || force == true) {
      let promise;
      if (this.value.lora && this.value.lora != "None") {
        promise = LORA_INFO_SERVICE.getInfo(this.value.lora, force, true);
      } else {
        promise = Promise.resolve(null);
      }
      this.loraInfoPromise = promise.then((v) => (this.loraInfo = v));
    }
    return this.loraInfoPromise;
  }
}

/** An uniformed name reference to the node class. */
const NODE_CLASS = RgthreePowerLoraLoader;

/** Register the node. */
app.registerExtension({
  name: "rgthree.PowerLoraLoader",
  async beforeRegisterNodeDef(nodeType: typeof LGraphNode, nodeData: ComfyNodeDef) {
    if (nodeData.name === NODE_CLASS.type) {
      NODE_CLASS.setUp(nodeType, nodeData);
    }
  },
});
