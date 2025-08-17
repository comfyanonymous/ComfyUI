import { app } from "../../scripts/app.js";
import { RgthreeBaseServerNode } from "./base_node.js";
import { NodeTypesString } from "./constants.js";
class RgthreeImageOrLatentSize extends RgthreeBaseServerNode {
    static setUp(comfyClass, nodeData) {
        RgthreeBaseServerNode.registerForOverride(comfyClass, nodeData, NODE_CLASS);
    }
    constructor(title = NODE_CLASS.title) {
        super(title);
    }
    onNodeCreated() {
        var _a;
        (_a = super.onNodeCreated) === null || _a === void 0 ? void 0 : _a.call(this);
        this.addInput("input", ["IMAGE", "LATENT", "MASK"]);
    }
    configure(info) {
        var _a;
        super.configure(info);
        if ((_a = this.inputs) === null || _a === void 0 ? void 0 : _a.length) {
            this.inputs[0].type = ["IMAGE", "LATENT", "MASK"];
        }
    }
}
RgthreeImageOrLatentSize.title = NodeTypesString.IMAGE_OR_LATENT_SIZE;
RgthreeImageOrLatentSize.type = NodeTypesString.IMAGE_OR_LATENT_SIZE;
RgthreeImageOrLatentSize.comfyClass = NodeTypesString.IMAGE_OR_LATENT_SIZE;
const NODE_CLASS = RgthreeImageOrLatentSize;
app.registerExtension({
    name: "rgthree.ImageOrLatentSize",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === NODE_CLASS.type) {
            NODE_CLASS.setUp(nodeType, nodeData);
        }
    },
});
