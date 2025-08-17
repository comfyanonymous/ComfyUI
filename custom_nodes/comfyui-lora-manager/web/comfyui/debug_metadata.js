import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { addJsonDisplayWidget } from "./json_display_widget.js";

app.registerExtension({
    name: "LoraManager.DebugMetadata",
    
    setup() {
        // Add message handler to listen for metadata updates from Python
        api.addEventListener("metadata_update", (event) => {
            const { id, metadata } = event.detail;
            this.handleMetadataUpdate(id, metadata);
        });
    },

    async nodeCreated(node) {
        if (node.comfyClass === "Debug Metadata (LoraManager)") {
            // Enable widget serialization
            node.serialize_widgets = true;

            // Add a widget to display metadata
            const jsonWidget = addJsonDisplayWidget(node, "metadata", {
                defaultVal: "",
            }).widget;
            
            // Store reference to the widget
            node.jsonWidget = jsonWidget;

            // Restore saved value if exists
            if (node.widgets_values && node.widgets_values.length > 0) {
                const savedValue = node.widgets_values[0];
                if (savedValue) {
                    jsonWidget.value = savedValue;
                }
            }
        }
    },
    
    // Handle metadata updates from Python
    handleMetadataUpdate(id, metadata) {
        const node = app.graph.getNodeById(+id);
        if (!node || node.comfyClass !== "Debug Metadata (LoraManager)") {
            console.warn("Node not found or not a DebugMetadata node:", id);
            return;
        }
        
        if (node.jsonWidget) {
            // Update the widget with the received metadata
            node.jsonWidget.value = metadata;
        }
    }
});