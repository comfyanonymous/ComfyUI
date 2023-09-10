import { ComfyApp, app } from "/scripts/app.js";

const VALID_TYPES = ["STRING", "combo", "number", "BOOLEAN"];

function isConvertableWidget(widget, config) {
	return (VALID_TYPES.includes(widget.type) || VALID_TYPES.includes(config[0])) && !widget.options?.forceInput;
}


const CONVERTED_TYPE = "converted-widget";

function hideWidget(node, widget, suffix = "") {
	widget.origType = widget.type;
	widget.origComputeSize = widget.computeSize;
	widget.origSerializeValue = widget.serializeValue;
	widget.computeSize = () => [0, -4]; // -4 is due to the gap litegraph adds between widgets automatically
	widget.type = CONVERTED_TYPE + suffix;
	widget.serializeValue = () => {
		// Prevent serializing the widget if we have no input linked
		if (!node.inputs) {
			return undefined;
		}
		let node_input = node.inputs.find((i) => i.widget?.name === widget.name);

		if (!node_input || !node_input.link) {
			return undefined;
		}
		return widget.origSerializeValue ? widget.origSerializeValue() : widget.value;
	};

	// Hide any linked widgets, e.g. seed+seedControl
	if (widget.linkedWidgets) {
		for (const w of widget.linkedWidgets) {
			hideWidget(node, w, ":" + widget.name);
		}
	}
}

function showWidget(widget) {
	widget.type = widget.origType;
	widget.computeSize = widget.origComputeSize;
	widget.serializeValue = widget.origSerializeValue;

	delete widget.origType;
	delete widget.origComputeSize;
	delete widget.origSerializeValue;

	// Hide any linked widgets, e.g. seed+seedControl
	if (widget.linkedWidgets) {
		for (const w of widget.linkedWidgets) {
			showWidget(w);
		}
	}
}

function convertToInput(node, widget, config) {
	hideWidget(node, widget);

	const { linkType } = getWidgetType(config);

	// Add input and store widget config for creating on primitive node
	const sz = node.size;
	node.addInput(widget.name, linkType, {
		widget: { name: widget.name, config },
	});

	for (const widget of node.widgets) {
		widget.last_y += LiteGraph.NODE_SLOT_HEIGHT;
	}

	// Restore original size but grow if needed
	node.setSize([Math.max(sz[0], node.size[0]), Math.max(sz[1], node.size[1])]);
}

function convertToWidget(node, widget) {
	showWidget(widget);
	const sz = node.size;
	node.removeInput(node.inputs.findIndex((i) => i.widget?.name === widget.name));

	for (const widget of node.widgets) {
		widget.last_y -= LiteGraph.NODE_SLOT_HEIGHT;
	}

	// Restore original size but grow if needed
	node.setSize([Math.max(sz[0], node.size[0]), Math.max(sz[1], node.size[1])]);
}

export function getWidgetType(config) {
	// Special handling for COMBO so we restrict links based on the entries
	let type = config[0];
	let linkType = type;
	if (type instanceof Array) {
		type = "COMBO";
		linkType = linkType.join(",");
	}
	return { type, linkType };
}


/** Forward values from the `node`'s outputs to all linked input widgets.
 * 
 * @param {LGraphNode} node The source node where we want to forward the output values to the
 *        linked input widgets.
 * @param {Function} valueForOutput Function to determine the value for the given `node`'s
 *        output entry
 */
export function forwardOutputValues(node, valueForOutput) {
    function getValueReceivers(node, output) {
      var receivers = [];
      for (const link of output.links || []) {
        const link_info = app.graph.links[link];
        const receiver = node.graph.getNodeById(link_info.target_id);
        if (receiver.type == "Reroute") {
          receivers = receivers.concat(getValueReceivers(receiver));
        } else {
          receivers.push({ receiver: receiver, input: receiver.inputs[link_info.target_slot] });
        }
      }
      return receivers;
    }
  
    for (const output of node.outputs) {
      const receivers = getValueReceivers(node, output);
      for (const receiver of receivers) {
        const widget_name = receiver.input.widget.name;
        const widget = widget_name ? receiver.receiver.widgets.find((w) => w.name == widget_name) : null;
        if (widget) {
          widget.value = valueForOutput(output);
          if (widget.callback) {
            widget.callback(widget.value, app.canvas, receiver.receiver, app.canvas.graph_mouse, {});
          }
        }
      }
    }
  }
  
  /** Add context menu entries for input widgets of a node which a user can convert to inputs and back to widgets.
   * 
   * @param {LGraphNode} nodeType The node class that we want to extend with the menu entries.
   * @param {ComfyObjectInfo} nodeData Construction data for the node
   * @param {ComfyApp} app The application object
   */
  export async function applyInputWidgetConversionMenu(nodeType, nodeData, app) {
    const origGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
    nodeType.prototype.getExtraMenuOptions = function (_, options) {
      const r = origGetExtraMenuOptions ? origGetExtraMenuOptions.apply(this, arguments) : undefined;
  
      if (this.widgets) {
        let toInput = [];
        let toWidget = [];
        for (const w of this.widgets) {
          if (w.options?.forceInput) {
            continue;
          }
          if (w.type === CONVERTED_TYPE) {
            toWidget.push({
              content: `Convert ${w.name} to widget`,
              callback: () => convertToWidget(this, w),
            });
          } else {
            const config = nodeData?.input?.required[w.name] ||
              nodeData?.input?.optional?.[w.name] || [w.type, w.options || {}];
            if (isConvertableWidget(w, config)) {
              toInput.push({
                content: `Convert ${w.name} to input`,
                callback: () => convertToInput(this, w, config),
              });
            }
          }
        }
        if (toInput.length) {
          options.push(...toInput, null);
        }
  
        if (toWidget.length) {
          options.push(...toWidget, null);
        }
      }
  
      return r;
    };
  
    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = origOnNodeCreated ? origOnNodeCreated.apply(this) : undefined;
      if (this.widgets) {
        for (const w of this.widgets) {
					if (w?.options?.forceInput || w?.options?.defaultInput) {
            const config = nodeData?.input?.required[w.name] ||
              nodeData?.input?.optional?.[w.name] || [w.type, w.options || {}];
            convertToInput(this, w, config);
          }
        }
      }
      return r;
    };
  
    // On initial configure of nodes hide all converted widgets
    const origOnConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = origOnConfigure ? origOnConfigure.apply(this, arguments) : undefined;
  
      if (this.inputs) {
        for (const input of this.inputs) {
          if (input.widget && !input.widget.config[1]?.forceInput) {
            const w = this.widgets.find((w) => w.name === input.widget.name);
            if (w) {
              hideWidget(this, w);
            } else {
              convertToWidget(this, input);
            }
          }
        }
      }
  
      return r;
    };
  
    function isNodeAtPos(pos) {
      for (const n of app.graph._nodes) {
        if (n.pos[0] === pos[0] && n.pos[1] === pos[1]) {
          return true;
        }
      }
      return false;
    }
  
    // Double click a widget input to automatically attach a primitive
    const origOnInputDblClick = nodeType.prototype.onInputDblClick;
    const ignoreDblClick = Symbol();
    nodeType.prototype.onInputDblClick = function (slot) {
      const r = origOnInputDblClick ? origOnInputDblClick.apply(this, arguments) : undefined;
  
      const input = this.inputs[slot];
      if (!input.widget || !input[ignoreDblClick]) {
        // Not a widget input or already handled input
        if (!(input.type in ComfyWidgets) && !(input.widget.config?.[0] instanceof Array)) {
          return r; //also Not a ComfyWidgets input or combo (do nothing)
        }
      }
  
      // Create a primitive node
      const node = LiteGraph.createNode("PrimitiveNode");
      app.graph.add(node);
  
      // Calculate a position that wont directly overlap another node
      const pos = [this.pos[0] - node.size[0] - 30, this.pos[1]];
      while (isNodeAtPos(pos)) {
        pos[1] += LiteGraph.NODE_TITLE_HEIGHT;
      }
  
      node.pos = pos;
      node.connect(0, this, slot);
      node.title = input.name;
  
      // Prevent adding duplicates due to triple clicking
      input[ignoreDblClick] = true;
      setTimeout(() => {
        delete input[ignoreDblClick];
      }, 300);
  
      return r;
    };
  }
  