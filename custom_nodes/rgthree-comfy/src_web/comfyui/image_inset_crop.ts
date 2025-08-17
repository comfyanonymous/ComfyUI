import type {LGraph, LGraphNodeConstructor, ISerialisedNode} from "@comfyorg/frontend";
import type {ComfyNodeDef} from "typings/comfy.js";

import {app} from "scripts/app.js";
import {RgthreeBaseServerNode} from "./base_node.js";
import {NodeTypesString} from "./constants.js";

class ImageInsetCrop extends RgthreeBaseServerNode {
  static override title = NodeTypesString.IMAGE_INSET_CROP;
  static override type = NodeTypesString.IMAGE_INSET_CROP;
  static comfyClass = NodeTypesString.IMAGE_INSET_CROP;

  static override exposedActions = ["Reset Crop"];
  static maxResolution = 8192;

  constructor(title = ImageInsetCrop.title) {
    super(title);
  }

  override onAdded(graph: LGraph): void {
    const measurementWidget = this.widgets[0]!;
    let callback = measurementWidget.callback;
    measurementWidget.callback = (...args) => {
      this.setWidgetStep();
      callback && callback.apply(measurementWidget, [...args]);
    };
    this.setWidgetStep();
  }

  override configure(info: ISerialisedNode): void {
    super.configure(info);
    this.setWidgetStep();
  }

  private setWidgetStep() {
    const measurementWidget = this.widgets[0]!;
    for (let i = 1; i <= 4; i++) {
      if (measurementWidget.value === "Pixels") {
        this.widgets[i]!.options.step = 80;
        this.widgets[i]!.options.max = ImageInsetCrop.maxResolution;
      } else {
        this.widgets[i]!.options.step = 10;
        this.widgets[i]!.options.max = 99;
      }
    }
  }

  override async handleAction(action: string): Promise<void> {
    if (action === "Reset Crop") {
      for (const widget of this.widgets) {
        if (["left", "right", "top", "bottom"].includes(widget.name!)) {
          widget.value = 0;
        }
      }
    }
  }

  static override setUp(comfyClass: typeof LGraphNode, nodeData: ComfyNodeDef) {
    RgthreeBaseServerNode.registerForOverride(comfyClass, nodeData, ImageInsetCrop);
  }
}

app.registerExtension({
  name: "rgthree.ImageInsetCrop",
  async beforeRegisterNodeDef(nodeType: typeof LGraphNode, nodeData: ComfyNodeDef) {
    if (nodeData.name === NodeTypesString.IMAGE_INSET_CROP) {
      ImageInsetCrop.setUp(nodeType, nodeData);
    }
  },
});
