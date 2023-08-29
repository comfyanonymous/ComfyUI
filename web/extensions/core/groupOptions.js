import {app} from "../../scripts/app.js";

function setNodeMode(node, mode) {
    node.mode = mode;
    node.graph.change();
}

app.registerExtension({
    name: "Comfy.GroupOptions",
    setup() {
        const orig = LGraphCanvas.prototype.getCanvasMenuOptions;
        // graph_mouse
        LGraphCanvas.prototype.getCanvasMenuOptions = function () {
            const options = orig.apply(this, arguments);
            const group = this.graph.getGroupOnPos(this.graph_mouse[0], this.graph_mouse[1]);
            if (!group) {
                return options;
            }

            // Group nodes aren't recomputed until the group is moved, this ensures the nodes are up-to-date
            group.recomputeInsideNodes();
            const nodesInGroup = group._nodes;

            // No nodes in group, return default options
            if (nodesInGroup.length === 0) {
                return options;
            } else {
                // Add a separator between the default options and the group options
                options.push(null);
            }

            // Check if all nodes are the same mode
            let allNodesAreSameMode = true;
            for (let i = 1; i < nodesInGroup.length; i++) {
                if (nodesInGroup[i].mode !== nodesInGroup[0].mode) {
                    allNodesAreSameMode = false;
                    break;
                }
            }

            // Modes
            // 0: Always
            // 2: Never
            // 4: Bypass
            // If all nodes are the same mode, add a menu option to change the mode
            if (allNodesAreSameMode) {
                const mode = nodesInGroup[0].mode;
                // All nodes are always, option to disable, and bypass
                if (mode === 0) {
                    options.push({
                        content: "Disable Group Nodes",
                        callback: () => {
                            for (const node of nodesInGroup) {
                                setNodeMode(node, 2);
                            }
                        }
                    });
                    options.push({
                        content: "Bypass Group Nodes",
                        callback: () => {
                            for (const node of nodesInGroup) {
                                setNodeMode(node, 4);
                            }
                        }
                    })
                }
                // All nodes are never, option to enable, and bypass
                if (mode === 2) {
                    options.push({
                        content: "Enable Group Nodes",
                        callback: () => {
                            for (const node of nodesInGroup) {
                                setNodeMode(node, 0);
                            }
                        }
                    });
                    options.push({
                        content: "Bypass Group Nodes",
                        callback: () => {
                            for (const node of nodesInGroup) {
                                setNodeMode(node, 4);
                            }
                        }
                    })
                }
                // All nodes are bypass, option to enable, and disable
                if (mode === 4) {
                    options.push({
                        content: "Enable Group Nodes",
                        callback: () => {
                            for (const node of nodesInGroup) {
                                setNodeMode(node, 0);
                            }
                        }
                    });
                    options.push({
                        content: "Disable Group Nodes",
                        callback: () => {
                            for (const node of nodesInGroup) {
                                setNodeMode(node, 2);
                            }
                        }
                    })
                }
            } else {
                // Nodes are not all the same mode, add a menu option to change the mode to always, never, or bypass
                options.push({
                    content: "Enable Group Nodes",
                    callback: () => {
                        for (const node of nodesInGroup) {
                            setNodeMode(node, 0);
                        }
                    }
                });
                options.push({
                    content: "Disable Group Nodes",
                    callback: () => {
                        for (const node of nodesInGroup) {
                            setNodeMode(node, 2);
                        }
                    }
                });
                options.push({
                    content: "Bypass Group Nodes",
                    callback: () => {
                        for (const node of nodesInGroup) {
                            setNodeMode(node, 4);
                        }
                    }
                });
            }

            return options
        }
    }
});
