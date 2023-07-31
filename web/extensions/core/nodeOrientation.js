import {app} from "../../scripts/app.js";

app.registerExtension({
        name: "Comfy.NodeOrientation",

        // On node creation, set the horizontal property if it's set in the node properties
        // This is needed as the horizontal status is not serialized
        loadedGraphNode(node, _) {
            if (node?.forced_orientation !== undefined) {
                // Orientation is already handled by nodeCreated
                return;
            }

            if (node?.properties?.orientation !== undefined) {
                node.horizontal = node.properties.orientation !== 'horizontal';
            }
        },
        nodeCreated(node, _) {
            if (node?.forced_orientation !== undefined) {
                // forced_orientation handles node orientation
                // Litegraph uses the opposite of horizontal for the orientation
                node.horizontal = node.forced_orientation !== 'horizontal';
                return;
            }

            if (node?.default_orientation !== undefined) {
                node.horizontal = node.default_orientation !== 'horizontal';
            }
        },
        async beforeRegisterNodeDef(nodeType, nodeData, app) {
            const hasForcedOrientation = Object.hasOwn(nodeData, "forced_orientation");
            const hasDefaultOrientation = Object.hasOwn(nodeData, "default_orientation");
            if (hasForcedOrientation) {
                nodeType.prototype.forced_orientation = nodeData.forced_orientation;
            }
            if (hasDefaultOrientation) {
                nodeType.prototype.default_orientation = nodeData.default_orientation;
            }

            // nodeType.prototype.forced_orientation = false;
            // Add menu option to change node orientation
            const origGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
            const origOnConfigure = nodeType.prototype.onConfigure;
            if (!Object.hasOwn(nodeData, "forced_orientation")) {
                nodeType.prototype.onConfigure = function (settings) {
                    origOnConfigure.apply(this, arguments);
                    if (settings?.properties?.orientation !== undefined) {
                        this.horizontal = settings.properties.orientation !== 'horizontal';
                    }
                    console.log("onConfigure", this, settings);
                }
                nodeType.prototype.getExtraMenuOptions = function (_, options) {
                    const r = origGetExtraMenuOptions ? origGetExtraMenuOptions.apply(this, arguments) : undefined;
                    options.push({
                        content: `Set ${this.horizontal ? "horizontal" : "vertical"}`,
                        callback: () => {
                            // properties get serialized, so we need to set the property on the node
                            this.properties.orientation = this.horizontal ? "horizontal" : "vertical";
                            this.horizontal = !this.horizontal;
                            app.graph.setDirtyCanvas(false, true);
                        },
                    })
                    return r;
                }
            }
        },
    }
)

