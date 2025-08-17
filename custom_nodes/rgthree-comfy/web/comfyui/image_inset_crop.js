import { app } from "../../scripts/app.js";
import { RgthreeBaseServerNode } from "./base_node.js";
import { NodeTypesString } from "./constants.js";
class ImageInsetCrop extends RgthreeBaseServerNode {
    constructor(title = ImageInsetCrop.title) {
        super(title);
    }
    onAdded(graph) {
        const measurementWidget = this.widgets[0];
        let callback = measurementWidget.callback;
        measurementWidget.callback = (...args) => {
            this.setWidgetStep();
            callback && callback.apply(measurementWidget, [...args]);
        };
        this.setWidgetStep();
    }
    configure(info) {
        super.configure(info);
        this.setWidgetStep();
    }
    setWidgetStep() {
        const measurementWidget = this.widgets[0];
        for (let i = 1; i <= 4; i++) {
            if (measurementWidget.value === "Pixels") {
                this.widgets[i].options.step = 80;
                this.widgets[i].options.max = ImageInsetCrop.maxResolution;
            }
            else {
                this.widgets[i].options.step = 10;
                this.widgets[i].options.max = 99;
            }
        }
    }
    async handleAction(action) {
        if (action === "Reset Crop") {
            for (const widget of this.widgets) {
                if (["left", "right", "top", "bottom"].includes(widget.name)) {
                    widget.value = 0;
                }
            }
        }
    }
    static setUp(comfyClass, nodeData) {
        RgthreeBaseServerNode.registerForOverride(comfyClass, nodeData, ImageInsetCrop);
    }
}
ImageInsetCrop.title = NodeTypesString.IMAGE_INSET_CROP;
ImageInsetCrop.type = NodeTypesString.IMAGE_INSET_CROP;
ImageInsetCrop.comfyClass = NodeTypesString.IMAGE_INSET_CROP;
ImageInsetCrop.exposedActions = ["Reset Crop"];
ImageInsetCrop.maxResolution = 8192;
app.registerExtension({
    name: "rgthree.ImageInsetCrop",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === NodeTypesString.IMAGE_INSET_CROP) {
            ImageInsetCrop.setUp(nodeType, nodeData);
        }
    },
});
