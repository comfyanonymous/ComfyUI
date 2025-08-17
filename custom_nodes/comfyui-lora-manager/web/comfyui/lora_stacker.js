import { app } from "../../scripts/app.js";
import { 
    LORA_PATTERN, 
    getActiveLorasFromNode,
    collectActiveLorasFromChain,
    updateConnectedTriggerWords,
    chainCallback,
    mergeLoras
} from "./utils.js";
import { addLorasWidget } from "./loras_widget.js";

app.registerExtension({
    name: "LoraManager.LoraStacker",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeType.comfyClass === "Lora Stacker (LoraManager)") {
            chainCallback(nodeType.prototype, "onNodeCreated", async function() {
                // Enable widget serialization
                this.serialize_widgets = true;

                this.addInput("lora_stack", 'LORA_STACK', {
                    "shape": 7  // 7 is the shape of the optional input
                });

                // Restore saved value if exists
                let existingLoras = [];
                if (this.widgets_values && this.widgets_values.length > 0) {
                    // 0 for input widget, 1 for loras widget
                    const savedValue = this.widgets_values[1];
                    existingLoras = savedValue || [];
                }
                // Merge the loras data
                const mergedLoras = mergeLoras(this.widgets[0].value, existingLoras);
                
                // Add flag to prevent callback loops
                let isUpdating = false;
                 
                const result = addLorasWidget(this, "loras", {
                    defaultVal: mergedLoras  // Pass object directly
                }, (value) => {
                    // Prevent recursive calls
                    if (isUpdating) return;
                    isUpdating = true;
                    
                    try {
                        // Remove loras that are not in the value array
                        const inputWidget = this.widgets[0];
                        const currentLoras = value.map(l => l.name);
                        
                        // Use the constant pattern here as well
                        let newText = inputWidget.value.replace(LORA_PATTERN, (match, name, strength) => {
                            return currentLoras.includes(name) ? match : '';
                        });
                        
                        // Clean up multiple spaces and trim
                        newText = newText.replace(/\s+/g, ' ').trim();
                        
                        inputWidget.value = newText;
                        
                        // Update this stacker's direct trigger toggles with its own active loras
                        const activeLoraNames = new Set();
                        value.forEach(lora => {
                            if (lora.active) {
                                activeLoraNames.add(lora.name);
                            }
                        });
                        updateConnectedTriggerWords(this, activeLoraNames);
                        
                        // Find all Lora Loader nodes in the chain that might need updates
                        updateDownstreamLoaders(this);
                    } finally {
                        isUpdating = false;
                    }
                });
                
                this.lorasWidget = result.widget;

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
                        
                        // Update this stacker's direct trigger toggles with its own active loras
                        const activeLoraNames = getActiveLorasFromNode(this);
                        updateConnectedTriggerWords(this, activeLoraNames);
                        
                        // Find all Lora Loader nodes in the chain that might need updates
                        updateDownstreamLoaders(this);
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

                // Call registration
                // setTimeout(() => {
                //     this.registerNode();
                // }, 0);
            });
        }
    },
});

// Helper function to find and update downstream Lora Loader nodes
function updateDownstreamLoaders(startNode, visited = new Set()) {
    if (visited.has(startNode.id)) return;
    visited.add(startNode.id);
    
    // Check each output link
    if (startNode.outputs) {
        for (const output of startNode.outputs) {
            if (output.links) {
                for (const linkId of output.links) {
                    const link = app.graph.links[linkId];
                    if (link) {
                        const targetNode = app.graph.getNodeById(link.target_id);
                        
                        // If target is a Lora Loader, collect all active loras in the chain and update
                        if (targetNode && targetNode.comfyClass === "Lora Loader (LoraManager)") {
                            const allActiveLoraNames = collectActiveLorasFromChain(targetNode);
                            updateConnectedTriggerWords(targetNode, allActiveLoraNames);
                        }
                        // If target is another Lora Stacker, recursively check its outputs
                        else if (targetNode && targetNode.comfyClass === "Lora Stacker (LoraManager)") {
                            updateDownstreamLoaders(targetNode, visited);
                        }
                    }
                }
            }
        }
    }
}