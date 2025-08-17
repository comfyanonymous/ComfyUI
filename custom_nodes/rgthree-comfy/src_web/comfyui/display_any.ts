import type {LGraphNodeConstructor, LGraphNode as TLGraphNode} from "@comfyorg/frontend";
import type {ComfyNodeDef} from "typings/comfy.js";
import type {ComfyApp} from "@comfyorg/frontend";

import {app} from "scripts/app.js";
import {ComfyWidgets} from "scripts/widgets.js";
import {addConnectionLayoutSupport} from "./utils.js";
import {rgthree} from "./rgthree.js";

let hasShownAlertForUpdatingInt = false;

app.registerExtension({
  name: "rgthree.DisplayAny",
  async beforeRegisterNodeDef(nodeType: typeof LGraphNode, nodeData: ComfyNodeDef, app: ComfyApp) {
    if (nodeData.name === "Display Any (rgthree)" || nodeData.name === "Display Int (rgthree)") {
      const onNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        onNodeCreated ? onNodeCreated.apply(this, []) : undefined;

        (this as any).showValueWidget = ComfyWidgets["STRING"](
          this,
          "output",
          ["STRING", {multiline: true}],
          app,
        ).widget;
        (this as any).showValueWidget.inputEl!.readOnly = true;
        (this as any).showValueWidget.serializeValue = async (node: TLGraphNode, index: number) => {
          const n =
            rgthree.getNodeFromInitialGraphToPromptSerializedWorkflowBecauseComfyUIBrokeStuff(node);
          if (n) {
            // Since we need a round trip to get the value, the serizalized value means nothing, and
            // saving it to the metadata would just be confusing. So, we clear it here.
            n.widgets_values![index] = "";
          } else {
            console.warn(
              "No serialized node found in workflow. May be attributed to " +
                "https://github.com/comfyanonymous/ComfyUI/issues/2193",
            );
          }
          return "";
        };
      };

      addConnectionLayoutSupport(nodeType as LGraphNodeConstructor, app, [["Left"], ["Right"]]);

      const onExecuted = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (message: any) {
        onExecuted?.apply(this, [message]);
        (this as any).showValueWidget.value = message.text[0];
      };
    }
  },

  // This ports Display Int to DisplayAny, but ComfyUI still shows an error.
  // If https://github.com/comfyanonymous/ComfyUI/issues/1527 is fixed, this could work.
  // async loadedGraphNode(node: TLGraphNode) {
  //   if (node.type === "Display Int (rgthree)") {
  //     replaceNode(node, "Display Any (rgthree)", new Map([["input", "source"]]));
  //     if (!hasShownAlertForUpdatingInt) {
  //       hasShownAlertForUpdatingInt = true;
  //       setTimeout(() => {
  //         alert(
  //           "Don't worry, your 'Display Int' nodes have been updated to the new " +
  //             "'Display Any' nodes! You can ignore the error message underneath (for that node)." +
  //             "\n\nThanks.\n- rgthree",
  //         );
  //       }, 128);
  //     }
  //   }
  // },
});
