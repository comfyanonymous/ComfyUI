import type {Parser, Node, Tree} from "web-tree-sitter";
import type {IStringWidget, IWidget} from "@comfyorg/frontend";

import {app} from "scripts/app.js";
import {Exposed, execute, PyTuple} from "rgthree/common/py_parser.js";
import {RgthreeBaseVirtualNode} from "./base_node.js";
import {RgthreeBetterButtonWidget} from "./utils_widgets.js";
import {NodeTypesString} from "./constants.js";
import {ComfyWidgets} from "scripts/widgets.js";
import {SERVICE as CONFIG_SERVICE} from "./services/config_service.js";
import { changeModeOfNodes, getNodeById } from "./utils.js";

const BUILT_INS = {
  node: {
    fn: (query: string | number) => {
      if (typeof query === "number" || /^\d+(\.\d+)?/.exec(query)) {
        return new ComfyNodeWrapper(Number(query));
      }
      return null;
    },
  },
};

class RgthreePowerConductor extends RgthreeBaseVirtualNode {
  static override title = NodeTypesString.POWER_CONDUCTOR;
  static override type = NodeTypesString.POWER_CONDUCTOR;
  override comfyClass = NodeTypesString.POWER_CONDUCTOR;

  override serialize_widgets = true;

  private codeWidget: IStringWidget;
  private buttonWidget: RgthreeBetterButtonWidget;

  constructor(title = RgthreePowerConductor.title) {
    super(title);

    this.codeWidget = ComfyWidgets.STRING(this, "", ["STRING", {multiline: true}], app).widget;
    this.addCustomWidget(this.codeWidget);

    (this.buttonWidget = new RgthreeBetterButtonWidget("Run", (...args: any[]) => {
      this.execute();
    })),
      this.addCustomWidget(this.buttonWidget);

    this.onConstructed();
  }

  private execute() {
    execute(this.codeWidget.value, {}, BUILT_INS);
  }
}

const NODE_CLASS = RgthreePowerConductor;

/**
 * A wrapper around nodes to add helpers and control the exposure of properties and methods.
 */
class ComfyNodeWrapper {
  #id: number;

  constructor(id: number) {
    this.#id = id;
  }

  private getNode() {
    return getNodeById(this.#id)!;
  }

  @Exposed get id() {
    return this.getNode().id;
  }

  @Exposed get title() {
    return this.getNode().title;
  }
  set title(value: string) {
    this.getNode().title = value;
  }

  @Exposed get widgets() {
    return new PyTuple(this.getNode().widgets?.map((w) => new ComfyWidgetWrapper(w as IWidget)));
  }

  @Exposed get mode() {
    return this.getNode().mode;
  }

  @Exposed mute() {
    changeModeOfNodes(this.getNode(), 2);
  }

  @Exposed bypass() {
    changeModeOfNodes(this.getNode(), 4);
  }

  @Exposed enable() {
    changeModeOfNodes(this.getNode(), 0);
  }
}

/**
 * A wrapper around widgets to add helpers and control the exposure of properties and methods.
 */
class ComfyWidgetWrapper {
  #widget: IWidget;

  constructor(widget: IWidget) {
    this.#widget = widget;
  }

  @Exposed get value() {
    return this.#widget.value;
  }

  @Exposed get label() {
    return this.#widget.label;
  }

  @Exposed toggle(value?: boolean) {
    // IF the widget has a "toggle" method, then call it.
    if (typeof (this.#widget as any)["toggle"] === "function") {
      (this.#widget as any)["toggle"](value);
    } else {
      // Error.
    }
  }
}

/** Register the node. */
app.registerExtension({
  name: "rgthree.PowerConductor",
  registerCustomNodes() {
    if (CONFIG_SERVICE.getConfigValue("unreleased.power_conductor.enabled")) {
      NODE_CLASS.setUp();
    }
  },
});
