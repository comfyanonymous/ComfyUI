export const CONVERTED_TYPE = 'converted-widget';

export function chainCallback(object, property, callback) {
  if (object == undefined) {
    //This should not happen.
    console.error("Tried to add callback to non-existant object")
    return;
  }
  if (property in object) {
    const callback_orig = object[property]
    object[property] = function () {
      const r = callback_orig.apply(this, arguments);
      callback.apply(this, arguments);
      return r
    };
  } else {
    object[property] = callback;
  }
}

export function getComfyUIFrontendVersion() {
  return window['__COMFYUI_FRONTEND_VERSION__'] || "0.0.0";
}

// Dynamically import the appropriate widget based on app version
export async function dynamicImportByVersion(latestModulePath, legacyModulePath) {
  // Parse app version and compare with 1.12.6 (version when tags widget API changed)
  const currentVersion = getComfyUIFrontendVersion();
  const versionParts = currentVersion.split('.').map(part => parseInt(part, 10));
  const requiredVersion = [1, 12, 6];
  
  // Compare version numbers
  for (let i = 0; i < 3; i++) {
    if (versionParts[i] > requiredVersion[i]) {
      console.log(`Using latest widget: ${latestModulePath}`);
      return import(latestModulePath);
    } else if (versionParts[i] < requiredVersion[i]) {
      console.log(`Using legacy widget: ${legacyModulePath}`);
      return import(legacyModulePath);
    }
  }
  
  // If we get here, versions are equal, use the latest module
  console.log(`Using latest widget: ${latestModulePath}`);
  return import(latestModulePath);
}

export function hideWidgetForGood(node, widget, suffix = "") {
  widget.origType = widget.type;
  widget.origComputeSize = widget.computeSize;
  widget.origSerializeValue = widget.serializeValue;
  widget.computeSize = () => [0, -4]; // -4 is due to the gap litegraph adds between widgets automatically
  widget.type = CONVERTED_TYPE + suffix;
  // widget.serializeValue = () => {
  //     // Prevent serializing the widget if we have no input linked
  //     const w = node.inputs?.find((i) => i.widget?.name === widget.name);
  //     if (w?.link == null) {
  //         return undefined;
  //     }
  //     return widget.origSerializeValue ? widget.origSerializeValue() : widget.value;
  // };

  // Hide any linked widgets, e.g. seed+seedControl
  if (widget.linkedWidgets) {
    for (const w of widget.linkedWidgets) {
      hideWidgetForGood(node, w, `:${widget.name}`);
    }
  }
}

// Wrapper class to handle 'two element array bug' in LiteGraph or comfyui
export class DataWrapper {
  constructor(data) {
    this.data = data;
  }

  getData() {
    return this.data;
  }

  setData(data) {
    this.data = data;
  }
}

// Function to get the appropriate loras widget based on ComfyUI version
export async function getLorasWidgetModule() {
  return await dynamicImportByVersion("./loras_widget.js", "./legacy_loras_widget.js");
}

// Update pattern to match both formats: <lora:name:model_strength> or <lora:name:model_strength:clip_strength>
export const LORA_PATTERN = /<lora:([^:]+):([-\d\.]+)(?::([-\d\.]+))?>/g;

// Get connected Lora Stacker nodes that feed into the current node
export function getConnectedInputStackers(node) {
    const connectedStackers = [];
    
    if (node.inputs) {
        for (const input of node.inputs) {
            if (input.name === "lora_stack" && input.link) {
                const link = app.graph.links[input.link];
                if (link) {
                    const sourceNode = app.graph.getNodeById(link.origin_id);
                    if (sourceNode && sourceNode.comfyClass === "Lora Stacker (LoraManager)") {
                        connectedStackers.push(sourceNode);
                    }
                }
            }
        }
    }
    return connectedStackers;
}

// Get connected TriggerWord Toggle nodes that receive output from the current node
export function getConnectedTriggerToggleNodes(node) {
    const connectedNodes = [];
    
    if (node.outputs && node.outputs.length > 0) {
        for (const output of node.outputs) {
            if (output.links && output.links.length > 0) {
                for (const linkId of output.links) {
                    const link = app.graph.links[linkId];
                    if (link) {
                        const targetNode = app.graph.getNodeById(link.target_id);
                        if (targetNode && targetNode.comfyClass === "TriggerWord Toggle (LoraManager)") {
                            connectedNodes.push(targetNode.id);
                        }
                    }
                }
            }
        }
    }
    return connectedNodes;
}

// Extract active lora names from a node's widgets
export function getActiveLorasFromNode(node) {
    const activeLoraNames = new Set();
    
    // For lorasWidget style entries (array of objects)
    if (node.lorasWidget && node.lorasWidget.value) {
        node.lorasWidget.value.forEach(lora => {
            if (lora.active) {
                activeLoraNames.add(lora.name);
            }
        });
    }
    
    return activeLoraNames;
}

// Recursively collect all active loras from a node and its input chain
export function collectActiveLorasFromChain(node, visited = new Set()) {
    // Prevent infinite loops from circular references
    if (visited.has(node.id)) {
        return new Set();
    }
    visited.add(node.id);
    
    // Get active loras from current node
    const allActiveLoraNames = getActiveLorasFromNode(node);
    
    // Get connected input stackers and collect their active loras
    const inputStackers = getConnectedInputStackers(node);
    for (const stacker of inputStackers) {
        const stackerLoras = collectActiveLorasFromChain(stacker, visited);
        stackerLoras.forEach(name => allActiveLoraNames.add(name));
    }
    
    return allActiveLoraNames;
}

// Update trigger words for connected toggle nodes
export function updateConnectedTriggerWords(node, loraNames) {
    const connectedNodeIds = getConnectedTriggerToggleNodes(node);
    if (connectedNodeIds.length > 0) {
        fetch("/api/loras/get_trigger_words", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                lora_names: Array.from(loraNames),
                node_ids: connectedNodeIds
            })
        }).catch(err => console.error("Error fetching trigger words:", err));
    }
}

export function mergeLoras(lorasText, lorasArr) {
  // Parse lorasText into a map: name -> {strength, clipStrength}
  const parsedLoras = {};
  let match;
  LORA_PATTERN.lastIndex = 0;
  while ((match = LORA_PATTERN.exec(lorasText)) !== null) {
    const name = match[1];
    const modelStrength = Number(match[2]);
    const clipStrength = match[3] ? Number(match[3]) : modelStrength;
    parsedLoras[name] = { strength: modelStrength, clipStrength };
  }

  // Build result array in the order of lorasArr
  const result = [];
  const usedNames = new Set();

  for (const lora of lorasArr) {
    if (parsedLoras[lora.name]) {
      result.push({
        name: lora.name,
        strength: lora.strength !== undefined ? lora.strength : parsedLoras[lora.name].strength,
        active: lora.active !== undefined ? lora.active : true,
        clipStrength: lora.clipStrength !== undefined ? lora.clipStrength : parsedLoras[lora.name].clipStrength,
      });
      usedNames.add(lora.name);
    }
  }

  // Add any new loras from lorasText that are not in lorasArr, in their text order
  for (const name in parsedLoras) {
    if (!usedNames.has(name)) {
      result.push({
        name,
        strength: parsedLoras[name].strength,
        active: true,
        clipStrength: parsedLoras[name].clipStrength,
      });
    }
  }

  return result;
}