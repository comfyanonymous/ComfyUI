import type {
  IWidget,
  INodeInputSlot,
  LGraphCanvas as TLGraphCanvas,
  LGraphNodeConstructor,
  IContextMenuValue,
  INodeOutputSlot,
  ISlotType,
  ISerialisedNode,
  LLink,
  IBaseWidget,
} from "@comfyorg/frontend";
import type {ComfyNodeDef} from "typings/comfy.js";

import {app} from "scripts/app.js";
import {RgthreeBaseServerNode} from "./base_node.js";
import {NodeTypesString} from "./constants.js";
import {ComfyWidgets} from "scripts/widgets.js";
import {moveArrayItem} from "rgthree/common/shared_utils.js";

const PROPERTY_HIDE_TYPE_SELECTOR = "hideTypeSelector";
const PRIMITIVES = {
  STRING: "STRING",
  // "STRING (multiline)": "STRING",
  INT: "INT",
  FLOAT: "FLOAT",
  BOOLEAN: "BOOLEAN",
};

class RgthreePowerPrimitive extends RgthreeBaseServerNode {
  static override title = NodeTypesString.POWER_PRIMITIVE;
  static override type = NodeTypesString.POWER_PRIMITIVE;
  static comfyClass = NodeTypesString.POWER_PRIMITIVE;

  private outputTypeWidget!: IWidget;
  private valueWidget!: IWidget;
  private typeState: string = '';

  static "@hideTypeSelector" = {type: "boolean"};

  override properties!: RgthreeBaseServerNode["properties"] & {
    [PROPERTY_HIDE_TYPE_SELECTOR]: boolean;
  };

  constructor(title = NODE_CLASS.title) {
    super(title);
    this.properties[PROPERTY_HIDE_TYPE_SELECTOR] = false;
  }

  static override setUp(comfyClass: typeof LGraphNode, nodeData: ComfyNodeDef) {
    RgthreeBaseServerNode.registerForOverride(comfyClass, nodeData, NODE_CLASS);
  }

  /**
   * Adds the non-lora widgets. If we'll be configured then we remove them and add them back, so
   * this is really only for newly created nodes in the current session.
   */
  override onNodeCreated() {
    super.onNodeCreated?.();
    this.addInitialWidgets();
  }

  /**
   * Ensures to set the type widget whenever we configure.
   */
  override configure(info: ISerialisedNode): void {
    super.configure(info);
    // Update BOOL to BOOLEAN due to a bug using BOOL instead of BOOLEAN.
    if (this.outputTypeWidget.value === 'BOOL') {
      this.outputTypeWidget.value = 'BOOLEAN';
    }
    setTimeout(() => {
      this.setTypedData();
    });
  }

  /**
   * Adds menu options for the node: quick toggle to show/hide the first widget, and a menu-option
   * to change the type (for easier changing when hiding the first widget).
   */
  override getExtraMenuOptions(
    canvas: TLGraphCanvas,
    options: (IContextMenuValue<unknown> | null)[],
  ) {
    const that = this;
    super.getExtraMenuOptions(canvas, options);
    const isHidden = !!this.properties[PROPERTY_HIDE_TYPE_SELECTOR];

    const menuItems = [
      {
        content: `${isHidden ? "Show" : "Hide"} Type Selector Widget`,
        callback: (...args: any[]) => {
          this.setProperty(
            PROPERTY_HIDE_TYPE_SELECTOR,
            !this.properties[PROPERTY_HIDE_TYPE_SELECTOR],
          );
        },
      },
      {
        content: `Set type`,
        submenu: {
          options: Object.keys(PRIMITIVES),
          callback(value: any, ...args: any[]) {
            that.outputTypeWidget.value = value;
            that.setTypedData();
          },
        },
      },
    ];

    options.splice(0, 0, ...menuItems, null);
    return [];
  }

  private addInitialWidgets() {
    if (!this.outputTypeWidget) {
      this.outputTypeWidget = this.addWidget(
        "combo",
        "type",
        "STRING",
        (...args) => {
          this.setTypedData();
        },
        {
          values: Object.keys(PRIMITIVES),
        },
      ) as IWidget;
      this.outputTypeWidget.hidden = this.properties[PROPERTY_HIDE_TYPE_SELECTOR];
    }
    this.setTypedData();
  }

  /**
   * Sets the correct inputs, outputs, and widgets for the designated type (with the
   * `outputTypeWidget`) being the source of truth.
   */
  private setTypedData() {
    const name = "value";
    const type = this.outputTypeWidget.value as string;
    const linked = !!this.inputs?.[0]?.link;
    const newTypeState = `${type}|${linked}`;
    if (this.typeState == newTypeState) return;
    this.typeState = newTypeState;

    let value = this.valueWidget?.value ?? null;
    let newWidget: IWidget | null= null;
    // If we're linked, then set the UI to an empty string widget input, since the ComfyUI is rather
    // confusing by showing a value that is not the actual value used (from the input).
    if (linked) {
      newWidget = ComfyWidgets["STRING"](this, name, ["STRING"], app).widget;
      newWidget.value = "";
    } else if (type == "STRING") {
      newWidget = ComfyWidgets["STRING"](this, name, ["STRING", {multiline: true}], app).widget;
      newWidget.value = value ? "" : String(value);
    } else if (type === "INT" || type === "FLOAT") {
      const isFloat = type === "FLOAT";
      newWidget = this.addWidget("number", name, value ?? 1 as any, undefined, {
        precision: isFloat ? 1 : 0,
        step2: isFloat ? 0.1 : 0,
      }) as IWidget;
      value = Number(value);
      value = value == null || isNaN(value) ? 0 : value;
      newWidget.value = value;
    } else if (type === "BOOLEAN") {
      newWidget = this.addWidget("toggle", name, !!(value ?? true), undefined, {
        on: "true",
        off: "false",
      }) as IWidget;
      if (typeof value === "string") {
        value = !["false", "null", "None", "", "0"].includes(value.toLowerCase());
      }
      newWidget.value = !!value;
    }
    if (newWidget == null) {
      throw new Error(`Unsupported type "${type}".`);
    }

    if (this.valueWidget) {
      this.replaceWidget(this.valueWidget, newWidget);
    } else {
      if (!this.widgets.includes(newWidget)) {
        this.addCustomWidget(newWidget);
      }
      moveArrayItem(this.widgets, newWidget, 1);
    }
    this.valueWidget = newWidget;

    // Set the input data.
    if (!this.inputs?.length) {
      this.addInput("value", "*", {widget: this.valueWidget as any});
    } else {
      this.inputs[0]!.widget = this.valueWidget as any;
    }

    // Set the output data.
    const output = this.outputs[0]!;
    const outputLabel = output.label === "*" || output.label === output.type ? null : output.label;
    output.type = type;
    output.label = outputLabel || output.type;
  }

  /**
   * Sets the correct typed data when we change any connections (really care about
   * onnecting/disconnecting the value input.)
   */
  override onConnectionsChange(
    type: ISlotType,
    index: number,
    isConnected: boolean,
    link_info: LLink | null | undefined,
    inputOrOutput: INodeInputSlot | INodeOutputSlot,
  ): void {
    super.onConnectionsChange?.apply(this, [...arguments] as any);
    if (this.inputs.includes(inputOrOutput as INodeInputSlot)) {
      this.setTypedData();
    }
  }

  /**
   * Sets the correct output type widget state when our `PROPERTY_HIDE_TYPE_SELECTOR` changes.
   */
  override onPropertyChanged(name: string, value: unknown, prev_value?: unknown): boolean {
    if (name === PROPERTY_HIDE_TYPE_SELECTOR) {
      if (!this.outputTypeWidget) {
        return true;
      }
      this.outputTypeWidget.hidden = this.properties[PROPERTY_HIDE_TYPE_SELECTOR];
      if (this.outputTypeWidget.hidden) {
        this.outputTypeWidget.computeLayoutSize = () => ({
          minHeight: 0,
          minWidth: 0,
          maxHeight: 0,
          maxWidth: 0,
        });
      } else {
        this.outputTypeWidget.computeLayoutSize = undefined;
      }
    }
    return true;
  }
}

/** An uniformed name reference to the node class. */
const NODE_CLASS = RgthreePowerPrimitive;

/** Register the node. */
app.registerExtension({
  name: "rgthree.PowerPrimitive",
  async beforeRegisterNodeDef(nodeType: typeof LGraphNode, nodeData: ComfyNodeDef) {
    if (nodeData.name === NODE_CLASS.type) {
      NODE_CLASS.setUp(nodeType, nodeData);
    }
  },
});
