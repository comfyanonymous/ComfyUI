import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { 
    LORA_PATTERN,
    collectActiveLorasFromChain,
    updateConnectedTriggerWords,
    chainCallback,
    mergeLoras
} from "./utils.js";
import { addLorasWidget } from "./loras_widget.js";

app.registerExtension({
    name: "LoraManager.LoraLoader",
    
    setup() {
        // Add message handler to listen for messages from Python
        api.addEventListener("lora_code_update", (event) => {
            const { id, lora_code, mode } = event.detail;
            this.handleLoraCodeUpdate(id, lora_code, mode);
        });
    },
    
    // Handle lora code updates from Python
    handleLoraCodeUpdate(id, loraCode, mode) {
        // Handle broadcast mode (for Desktop/non-browser support)
        if (id === -1) {
            // Find all Lora Loader nodes in the current graph
            const loraLoaderNodes = [];
            for (const nodeId in app.graph._nodes_by_id) {
                const node = app.graph._nodes_by_id[nodeId];
                if (node.comfyClass === "Lora Loader (LoraManager)") {
                    loraLoaderNodes.push(node);
                }
            }
            
            // Update each Lora Loader node found
            if (loraLoaderNodes.length > 0) {
                loraLoaderNodes.forEach(node => {
                    this.updateNodeLoraCode(node, loraCode, mode);
                });
                console.log(`Updated ${loraLoaderNodes.length} Lora Loader nodes in broadcast mode`);
            } else {
                console.warn("No Lora Loader nodes found in the workflow for broadcast update");
            }
            
            return;
        }
        
        // Standard mode - update a specific node
        const node = app.graph.getNodeById(+id);
        if (!node || (node.comfyClass !== "Lora Loader (LoraManager)" && 
                node.comfyClass !== "Lora Stacker (LoraManager)" && 
                node.comfyClass !== "WanVideo Lora Select (LoraManager)")) {
            console.warn("Node not found or not a LoraLoader:", id);
            return;
        }
        
        this.updateNodeLoraCode(node, loraCode, mode);
    },

    // Helper method to update a single node's lora code
    updateNodeLoraCode(node, loraCode, mode) {
        // Update the input widget with new lora code
        const inputWidget = node.inputWidget;
        if (!inputWidget) return;
        
        // Get the current lora code
        const currentValue = inputWidget.value || '';
        
        // Update based on mode (replace or append)
        if (mode === 'replace') {
            inputWidget.value = loraCode;
        } else {
            // Append mode - add a space if the current value isn't empty
            inputWidget.value = currentValue.trim() 
                ? `${currentValue.trim()} ${loraCode}` 
                : loraCode;
        }
        
        // Trigger the callback to update the loras widget
        if (typeof inputWidget.callback === 'function') {
            inputWidget.callback(inputWidget.value);
        }
    },

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeType.comfyClass == "Lora Loader (LoraManager)") {
          chainCallback(nodeType.prototype, "onNodeCreated", function () {
            // Enable widget serialization
            this.serialize_widgets = true;

            this.addInput("clip", "CLIP", {
              shape: 7,
            });

            this.addInput("lora_stack", "LORA_STACK", {
              shape: 7, // 7 is the shape of the optional input
            });

            // Restore saved value if exists
            let existingLoras = [];
            if (this.widgets_values && this.widgets_values.length > 0) {
              // 0 for input widget, 1 for loras widget
              const savedValue = this.widgets_values[1];
              existingLoras = savedValue || [];
            }
            // Merge the loras data
            const mergedLoras = mergeLoras(
              this.widgets[0].value,
              existingLoras
            );

            // Add flag to prevent callback loops
            let isUpdating = false;

            // Get the widget object directly from the returned object
            this.lorasWidget = addLorasWidget(
              this,
              "loras",
              {
                defaultVal: mergedLoras, // Pass object directly
              },
              (value) => {
                // Collect all active loras from this node and its input chain
                const allActiveLoraNames = collectActiveLorasFromChain(this);

                // Update trigger words for connected toggle nodes with the aggregated lora names
                updateConnectedTriggerWords(this, allActiveLoraNames);

                // Prevent recursive calls
                if (isUpdating) return;
                isUpdating = true;

                try {
                  // Remove loras that are not in the value array
                  const inputWidget = this.widgets[0];
                  const currentLoras = value.map((l) => l.name);

                  // Use the constant pattern here as well
                  let newText = inputWidget.value.replace(
                    LORA_PATTERN,
                    (match, name, strength, clipStrength) => {
                      return currentLoras.includes(name) ? match : "";
                    }
                  );

                  // Clean up multiple spaces and trim
                  newText = newText.replace(/\s+/g, " ").trim();

                  inputWidget.value = newText;
                } finally {
                  isUpdating = false;
                }
              }
            ).widget;

            // Update input widget callback
            const inputWidget = this.widgets[0];
            inputWidget.options.getMaxHeight = () => 100;
            this.inputWidget = inputWidget;
            inputWidget.callback = (value) => {
              if (isUpdating) return;
              isUpdating = true;

              try {
                const currentLoras = this.lorasWidget.value || [];
                const mergedLoras = mergeLoras(value, currentLoras);

                this.lorasWidget.value = mergedLoras;
              } finally {
                isUpdating = false;
              }
            };

            // Register this node with the backend
            this.registerNode = async () => {
              try {
                await fetch('/api/register-node', {
                  method: 'POST',
                  headers: {
                    'Content-Type': 'application/json',
                  },
                  body: JSON.stringify({
                    node_id: this.id,
                    bgcolor: this.bgcolor,
                    title: this.title,
                    graph_id: this.graph.id
                  })
                });
              } catch (error) {
                console.warn('Failed to register node:', error);
              }
            };

            // Ensure the node is registered after creation
            // Call registration
            // setTimeout(() => {
            //   this.registerNode();
            // }, 0);
          });
        }
    },
});