import { app } from "../../scripts/app.js";
import { 
    LORA_PATTERN, 
    getActiveLorasFromNode,
    updateConnectedTriggerWords,
    chainCallback,
    mergeLoras
} from "./utils.js";
import { addLorasWidget } from "./loras_widget.js";

app.registerExtension({
    name: "LoraManager.WanVideoLoraSelect",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeType.comfyClass === "WanVideo Lora Select (LoraManager)") {
            chainCallback(nodeType.prototype, "onNodeCreated", async function() {
                // Enable widget serialization
                this.serialize_widgets = true;

                // Add optional inputs
                this.addInput("prev_lora", 'WANVIDLORA', {
                    "shape": 7  // 7 is the shape of the optional input
                });
                
                this.addInput("blocks", 'SELECTEDBLOCKS', {
                    "shape": 7  // 7 is the shape of the optional input
                });

                // Restore saved value if exists
                let existingLoras = [];
                if (this.widgets_values && this.widgets_values.length > 0) {
                    // 0 for low_mem_load, 1 for text widget, 2 for loras widget
                    const savedValue = this.widgets_values[2];
                    existingLoras = savedValue || [];
                }
                // Merge the loras data
                const mergedLoras = mergeLoras(this.widgets[1].value, existingLoras);
                
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
                        const inputWidget = this.widgets[1];
                        const currentLoras = value.map(l => l.name);
                        
                        // Use the constant pattern here as well
                        let newText = inputWidget.value.replace(LORA_PATTERN, (match, name, strength) => {
                            return currentLoras.includes(name) ? match : '';
                        });
                        
                        // Clean up multiple spaces and trim
                        newText = newText.replace(/\s+/g, ' ').trim();
                        
                        inputWidget.value = newText;
                        
                        // Update this node's direct trigger toggles with its own active loras
                        const activeLoraNames = new Set();
                        value.forEach(lora => {
                            if (lora.active) {
                                activeLoraNames.add(lora.name);
                            }
                        });
                        updateConnectedTriggerWords(this, activeLoraNames);
                    } finally {
                        isUpdating = false;
                    }
                });
                
                this.lorasWidget = result.widget;

                // Update input widget callback
                const inputWidget = this.widgets[1];
                inputWidget.options.getMaxHeight = () => 100;
                this.inputWidget = inputWidget;
                inputWidget.callback = (value) => {
                    if (isUpdating) return;
                    isUpdating = true;
                    
                    try {
                        const currentLoras = this.lorasWidget.value || [];
                        const mergedLoras = mergeLoras(value, currentLoras);
                        
                        this.lorasWidget.value = mergedLoras;
                        
                        // Update this node's direct trigger toggles with its own active loras
                        const activeLoraNames = getActiveLorasFromNode(this);
                        updateConnectedTriggerWords(this, activeLoraNames);
                    } finally {
                        isUpdating = false;
                    }
                };
            });
        }
    },
});
