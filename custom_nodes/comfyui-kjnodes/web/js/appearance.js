import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "KJNodes.appearance",
        nodeCreated(node) {
            switch (node.comfyClass) {
                case "INTConstant":
                    node.setSize([200, 58]);
                    node.color = "#1b4669";
                    node.bgcolor = "#29699c";
                    break;
                case "FloatConstant":
                    node.setSize([200, 58]);
                    node.color = LGraphCanvas.node_colors.green.color;
                    node.bgcolor = LGraphCanvas.node_colors.green.bgcolor;
                    break;
                case "ConditioningMultiCombine":
                    node.color = LGraphCanvas.node_colors.brown.color;
                    node.bgcolor = LGraphCanvas.node_colors.brown.bgcolor;
                    break;
            }
        }
});
