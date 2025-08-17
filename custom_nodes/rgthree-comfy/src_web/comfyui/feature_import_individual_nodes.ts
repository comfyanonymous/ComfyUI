import type {INodeSlot, LGraphNode, LGraphNodeConstructor} from "@comfyorg/frontend";
import type {ComfyNodeDef} from "typings/comfy.js";

import {app} from "scripts/app.js";
import {tryToGetWorkflowDataFromEvent} from "rgthree/common/utils_workflow.js";
import {SERVICE as CONFIG_SERVICE} from "./services/config_service.js";
import {NodeTypesString} from "./constants.js";

/**
 * Registers the GroupHeaderToggles which places a mute and/or bypass icons in groups headers for
 * quick, single-click ability to mute/bypass.
 */
app.registerExtension({
  name: "rgthree.ImportIndividualNodes",
  async beforeRegisterNodeDef(nodeType: typeof LGraphNode, nodeData: ComfyNodeDef) {
    const onDragOver = nodeType.prototype.onDragOver;
    nodeType.prototype.onDragOver = function (e: DragEvent) {
      let handled = onDragOver?.apply?.(this, [...arguments] as any);
      if (handled != null) {
        return handled;
      }
      return importIndividualNodesInnerOnDragOver(this, e);
    };

    const onDragDrop = nodeType.prototype.onDragDrop;
    nodeType.prototype.onDragDrop = async function (e: DragEvent) {
      const alreadyHandled = await onDragDrop?.apply?.(this, [...arguments] as any);
      if (alreadyHandled) {
        return alreadyHandled;
      }
      return importIndividualNodesInnerOnDragDrop(this, e);
    };
  },
});

export function importIndividualNodesInnerOnDragOver(node: LGraphNode, e: DragEvent): boolean {
  return (
    (node.widgets?.length && !!CONFIG_SERVICE.getFeatureValue("import_individual_nodes.enabled")) ||
    false
  );
}

export async function importIndividualNodesInnerOnDragDrop(node: LGraphNode, e: DragEvent) {
  if (!node.widgets?.length || !CONFIG_SERVICE.getFeatureValue("import_individual_nodes.enabled")) {
    return false;
  }

  const dynamicWidgetLengthNodes = [NodeTypesString.POWER_LORA_LOADER];

  let handled = false;
  const {workflow, prompt} = await tryToGetWorkflowDataFromEvent(e);
  const exact = (workflow?.nodes || []).find(
    (n: any) =>
      n.id === node.id &&
      n.type === node.type &&
      (dynamicWidgetLengthNodes.includes(node.type) ||
        n.widgets_values?.length === node.widgets_values?.length),
  );
  if (!exact) {
    // If we tried, but didn't find an exact match, then allow user to stop the default behavior.
    handled = !confirm(
      "[rgthree-comfy] Could not find a matching node (same id & type) in the dropped workflow." +
        " Would you like to continue with the default drop behaviour instead?",
    );
  } else if (!exact.widgets_values?.length) {
    handled = !confirm(
      "[rgthree-comfy] Matching node found (same id & type) but there's no widgets to set." +
        " Would you like to continue with the default drop behaviour instead?",
    );
  } else if (
    confirm(
      "[rgthree-comfy] Found a matching node (same id & type) in the dropped workflow." +
        " Would you like to set the widget values?",
    )
  ) {
    node.configure({
      // Title is overridden if it's not supplied; set it to the current then.
      title: node.title,
      widgets_values: [...(exact?.widgets_values || [])],
      mode: exact.mode,
    } as any);
    handled = true;
  }
  return handled;
}
