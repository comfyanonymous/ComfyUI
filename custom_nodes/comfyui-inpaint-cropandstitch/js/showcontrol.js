import { app } from "../../scripts/app.js";

// Some fragments of this code are from https://github.com/LucianoCirino/efficiency-nodes-comfyui

function inpaintCropAndStitchHandler(node) {
    if (node.comfyClass == "InpaintCropImproved") {
        toggleWidget(node, findWidgetByName(node, "preresize_mode"));
        toggleWidget(node, findWidgetByName(node, "preresize_min_width"));
        toggleWidget(node, findWidgetByName(node, "preresize_min_height"));
        toggleWidget(node, findWidgetByName(node, "preresize_max_width"));
        toggleWidget(node, findWidgetByName(node, "preresize_max_height"));
        if (findWidgetByName(node, "preresize").value == true) {
            toggleWidget(node, findWidgetByName(node, "preresize_mode"), true);
            if (findWidgetByName(node, "preresize_mode").value == "ensure minimum resolution") {
                toggleWidget(node, findWidgetByName(node, "preresize_min_width"), true);
                toggleWidget(node, findWidgetByName(node, "preresize_min_height"), true);
            }
            else if (findWidgetByName(node, "preresize_mode").value == "ensure minimum and maximum resolution") {
                toggleWidget(node, findWidgetByName(node, "preresize_min_width"), true);
                toggleWidget(node, findWidgetByName(node, "preresize_min_height"), true);
                toggleWidget(node, findWidgetByName(node, "preresize_max_width"), true);
                toggleWidget(node, findWidgetByName(node, "preresize_max_height"), true);
            }
            else if (findWidgetByName(node, "preresize_mode").value == "ensure maximum resolution") {
                toggleWidget(node, findWidgetByName(node, "preresize_max_width"), true);
                toggleWidget(node, findWidgetByName(node, "preresize_max_height"), true);
            }
        }
        toggleWidget(node, findWidgetByName(node, "extend_up_factor"));
        toggleWidget(node, findWidgetByName(node, "extend_down_factor"));
        toggleWidget(node, findWidgetByName(node, "extend_left_factor"));
        toggleWidget(node, findWidgetByName(node, "extend_right_factor"));
        if (findWidgetByName(node, "extend_for_outpainting").value == true) {
            toggleWidget(node, findWidgetByName(node, "extend_up_factor"), true);
            toggleWidget(node, findWidgetByName(node, "extend_down_factor"), true);
            toggleWidget(node, findWidgetByName(node, "extend_left_factor"), true);
            toggleWidget(node, findWidgetByName(node, "extend_right_factor"), true);
        }
        toggleWidget(node, findWidgetByName(node, "output_target_width"));
        toggleWidget(node, findWidgetByName(node, "output_target_height"));
        if (findWidgetByName(node, "output_resize_to_target_size").value == true) {
            toggleWidget(node, findWidgetByName(node, "output_target_width"), true);
            toggleWidget(node, findWidgetByName(node, "output_target_height"), true);
        }
    }

    // OLD
    if (node.comfyClass == "InpaintCrop") {
        toggleWidget(node, findWidgetByName(node, "force_width"));
        toggleWidget(node, findWidgetByName(node, "force_height"));
        toggleWidget(node, findWidgetByName(node, "rescale_factor"));
        toggleWidget(node, findWidgetByName(node, "min_width"));
        toggleWidget(node, findWidgetByName(node, "min_height"));
        toggleWidget(node, findWidgetByName(node, "max_width"));
        toggleWidget(node, findWidgetByName(node, "max_height"));
        toggleWidget(node, findWidgetByName(node, "padding"));
        if (findWidgetByName(node, "mode").value == "free size") {
            toggleWidget(node, findWidgetByName(node, "rescale_factor"), true);
            toggleWidget(node, findWidgetByName(node, "padding"), true);
        }
        else if (findWidgetByName(node, "mode").value == "ranged size") {
            toggleWidget(node, findWidgetByName(node, "min_width"), true);
            toggleWidget(node, findWidgetByName(node, "min_height"), true);
            toggleWidget(node, findWidgetByName(node, "max_width"), true);
            toggleWidget(node, findWidgetByName(node, "max_height"), true);
            toggleWidget(node, findWidgetByName(node, "padding"), true);
        }
        else if (findWidgetByName(node, "mode").value == "forced size") {
            toggleWidget(node, findWidgetByName(node, "force_width"), true);
            toggleWidget(node, findWidgetByName(node, "force_height"), true);
        }
    } else if (node.comfyClass == "InpaintExtendOutpaint") {
        toggleWidget(node, findWidgetByName(node, "expand_up_pixels"));
        toggleWidget(node, findWidgetByName(node, "expand_up_factor"));
        toggleWidget(node, findWidgetByName(node, "expand_down_pixels"));
        toggleWidget(node, findWidgetByName(node, "expand_down_factor"));
        toggleWidget(node, findWidgetByName(node, "expand_left_pixels"));
        toggleWidget(node, findWidgetByName(node, "expand_left_factor"));
        toggleWidget(node, findWidgetByName(node, "expand_right_pixels"));
        toggleWidget(node, findWidgetByName(node, "expand_right_factor"));
        if (findWidgetByName(node, "mode").value == "factors") {
            toggleWidget(node, findWidgetByName(node, "expand_up_factor"), true);
            toggleWidget(node, findWidgetByName(node, "expand_down_factor"), true);
            toggleWidget(node, findWidgetByName(node, "expand_left_factor"), true);
            toggleWidget(node, findWidgetByName(node, "expand_right_factor"), true);
        }
        if (findWidgetByName(node, "mode").value == "pixels") {
            toggleWidget(node, findWidgetByName(node, "expand_up_pixels"), true);
            toggleWidget(node, findWidgetByName(node, "expand_down_pixels"), true);
            toggleWidget(node, findWidgetByName(node, "expand_left_pixels"), true);
            toggleWidget(node, findWidgetByName(node, "expand_right_pixels"), true);
        }
    } else if (node.comfyClass == "InpaintResize") {
        toggleWidget(node, findWidgetByName(node, "min_width"));
        toggleWidget(node, findWidgetByName(node, "min_height"));
        toggleWidget(node, findWidgetByName(node, "rescale_factor"));
        if (findWidgetByName(node, "mode").value == "ensure minimum size") {
            toggleWidget(node, findWidgetByName(node, "min_width"), true);
            toggleWidget(node, findWidgetByName(node, "min_height"), true);
        }
        else if (findWidgetByName(node, "mode").value == "factor") {
            toggleWidget(node, findWidgetByName(node, "rescale_factor"), true);
        }
    }
    return;
}

const findWidgetByName = (node, name) => {
    return node.widgets ? node.widgets.find((w) => w.name === name) : null;
};

// Toggle Widget + change size
function toggleWidget(node, widget, show = false, suffix = "") {
    if (!widget) return;
    widget.disabled = !show
    widget.linkedWidgets?.forEach(w => toggleWidget(node, w, ":" + widget.name, show));
}   

app.registerExtension({
    name: "inpaint-cropandstitch.showcontrol",
    nodeCreated(node) {
        if (!node.comfyClass.startsWith("Inpaint")) {
            return;
        }

        inpaintCropAndStitchHandler(node);
        for (const w of node.widgets || []) {
            let widgetValue = w.value;

            // Store the original descriptor if it exists 
            let originalDescriptor = Object.getOwnPropertyDescriptor(w, 'value') || 
                Object.getOwnPropertyDescriptor(Object.getPrototypeOf(w), 'value');
            if (!originalDescriptor) {
                originalDescriptor = Object.getOwnPropertyDescriptor(w.constructor.prototype, 'value');
            }

            Object.defineProperty(w, 'value', {
                get() {
                    // If there's an original getter, use it. Otherwise, return widgetValue.
                    let valueToReturn = originalDescriptor && originalDescriptor.get
                        ? originalDescriptor.get.call(w)
                        : widgetValue;

                    return valueToReturn;
                },
                set(newVal) {
                    // If there's an original setter, use it. Otherwise, set widgetValue.
                    if (originalDescriptor && originalDescriptor.set) {
                        originalDescriptor.set.call(w, newVal);
                    } else { 
                        widgetValue = newVal;
                    }

                    inpaintCropAndStitchHandler(node);
                }
            });
        }
    }
});
