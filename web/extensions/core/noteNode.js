import {app} from "../../scripts/app.js";
import {ComfyWidgets} from "../../scripts/widgets.js";
// Node that add notes to your project

app.registerExtension({
    name: "Comfy.NoteNode",
    registerCustomNodes() {
        class NoteNode {
            color=LGraphCanvas.node_colors.yellow.color;
            bgcolor=LGraphCanvas.node_colors.yellow.bgcolor;
            groupcolor = LGraphCanvas.node_colors.yellow.groupcolor;
            constructor() {
                if (!this.properties) {
                    this.properties = {};
                }

                ComfyWidgets.STRING(this, "", ["", {multiline: true}], app)
                // This node is purely frontend and does not impact the resulting prompt so should not be serialized
                this.isVirtualNode = true;
            }


        }

        // Load default visibility

        LiteGraph.registerNodeType(
            "Note",
            Object.assign(NoteNode, {
                title_mode: LiteGraph.NORMAL_TITLE,
                title: "Note",
                collapsable: true,
            })
        );

        NoteNode.category = "utils";
    },
});
