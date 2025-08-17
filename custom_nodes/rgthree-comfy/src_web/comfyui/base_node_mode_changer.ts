import type {LGraphNode, IWidget} from "@comfyorg/frontend";

import {BaseAnyInputConnectedNode} from "./base_any_input_connected_node.js";
import {changeModeOfNodes, PassThroughFollowing} from "./utils.js";
import {wait} from "rgthree/common/shared_utils.js";

export class BaseNodeModeChanger extends BaseAnyInputConnectedNode {
  override readonly inputsPassThroughFollowing: PassThroughFollowing = PassThroughFollowing.ALL;

  static collapsible = false;
  override isVirtualNode = true;

  // These Must be overriden
  readonly modeOn: number = -1;
  readonly modeOff: number = -1;

  static "@toggleRestriction" = {
    type: "combo",
    values: ["default", "max one", "always one"],
  };

  constructor(title?: string) {
    super(title);
    this.properties["toggleRestriction"] = "default";
  }

  override onConstructed(): boolean {
    wait(10).then(() => {
      if (this.modeOn < 0 || this.modeOff < 0) {
        throw new Error("modeOn and modeOff must be overridden.");
      }
    });
    this.addOutput("OPT_CONNECTION", "*");
    return super.onConstructed();
  }

  override handleLinkedNodesStabilization(linkedNodes: LGraphNode[]) {
    let changed = false;
    for (const [index, node] of linkedNodes.entries()) {
      let widget: IWidget | undefined = this.widgets && this.widgets[index];
      if (!widget) {
        // When we add a widget, litegraph is going to mess up the size, so we
        // store it so we can retrieve it in computeSize. Hacky..
        (this as any)._tempWidth = this.size[0];
        widget = this.addWidget("toggle", "", false, "", {on: "yes", off: "no"}) as IWidget;
        changed = true;
      }
      if (node) {
        changed = this.setWidget(widget, node) || changed;
      }
    }
    if (this.widgets && this.widgets.length > linkedNodes.length) {
      this.widgets.length = linkedNodes.length;
      changed = true;
    }
    return changed;
  }

  private setWidget(widget: IWidget, linkedNode: LGraphNode, forceValue?: boolean) {
    let changed = false;
    const value = forceValue == null ? linkedNode.mode === this.modeOn : forceValue;
    let name = `Enable ${linkedNode.title}`;
    // Need to set initally
    if (widget.name !== name) {
      widget.name = `Enable ${linkedNode.title}`;
      widget.options = {on: "yes", off: "no"};
      widget.value = value;
      (widget as any).doModeChange = (forceValue?: boolean, skipOtherNodeCheck?: boolean) => {
        let newValue = forceValue == null ? linkedNode.mode === this.modeOff : forceValue;
        if (skipOtherNodeCheck !== true) {
          if (newValue && (this.properties?.["toggleRestriction"] as string)?.includes(" one")) {
            for (const widget of this.widgets) {
              (widget as any).doModeChange(false, true);
            }
          } else if (!newValue && this.properties?.["toggleRestriction"] === "always one") {
            newValue = this.widgets.every((w) => !w.value || w === widget);
          }
        }
        changeModeOfNodes(linkedNode, (newValue ? this.modeOn : this.modeOff))
        widget.value = newValue;
      };
      widget.callback = () => {
        (widget as any).doModeChange();
      };
      changed = true;
    }
    if (forceValue != null) {
      const newMode = (forceValue ? this.modeOn : this.modeOff) as 1 | 2 | 3 | 4;
      if (linkedNode.mode !== newMode) {
        changeModeOfNodes(linkedNode, newMode);
        changed = true;
      }
    }
    return changed;
  }

  forceWidgetOff(widget: IWidget, skipOtherNodeCheck?: boolean) {
    (widget as any).doModeChange(false, skipOtherNodeCheck);
  }
  forceWidgetOn(widget: IWidget, skipOtherNodeCheck?: boolean) {
    (widget as any).doModeChange(true, skipOtherNodeCheck);
  }
  forceWidgetToggle(widget: IWidget, skipOtherNodeCheck?: boolean) {
    (widget as any).doModeChange(!widget.value, skipOtherNodeCheck);
  }
}
