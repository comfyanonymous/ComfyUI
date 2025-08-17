// ComfyUI extension to track model usage statistics
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Register the extension
app.registerExtension({
    name: "LoraManager.UsageStats",
    
    setup() {
        // Listen for successful executions
        api.addEventListener("execution_success", ({ detail }) => {
            if (detail && detail.prompt_id) {
                this.updateUsageStats(detail.prompt_id);
            }
        });

        // Listen for registry refresh requests
        api.addEventListener("lora_registry_refresh", () => {
            this.refreshRegistry();
        });
    },
    
    async updateUsageStats(promptId) {
        try {
            // Call backend endpoint with the prompt_id
            const response = await fetch(`/api/update-usage-stats`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ prompt_id: promptId }),
            });
            
            if (!response.ok) {
                console.warn("Failed to update usage statistics:", response.statusText);
            }
        } catch (error) {
            console.error("Error updating usage statistics:", error);
        }
    },

    async refreshRegistry() {
        try {
            // Get current workflow nodes
            const prompt = await app.graphToPrompt();
            const workflow = prompt.workflow;
            if (!workflow || !workflow.nodes) {
                console.warn("No workflow nodes found for registry refresh");
                return;
            }

            // Find all Lora nodes
            const loraNodes = [];
            for (const node of workflow.nodes.values()) {
                if (node.type === "Lora Loader (LoraManager)" || 
                    node.type === "Lora Stacker (LoraManager)" || 
                    node.type === "WanVideo Lora Select (LoraManager)") {
                    loraNodes.push({
                        node_id: node.id,
                        bgcolor: node.bgcolor || null,
                        title: node.title || node.type,
                        type: node.type
                    });
                }
            }

            const response = await fetch('/api/register-nodes', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ nodes: loraNodes }),
            });

            if (!response.ok) {
                console.warn("Failed to register Lora nodes:", response.statusText);
            } else {
                console.log(`Successfully registered ${loraNodes.length} Lora nodes`);
            }
        } catch (error) {
            console.error("Error refreshing registry:", error);
        }
    }
});
