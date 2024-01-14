import {app} from "../../scripts/app.js";

function setNodeMode(node, mode) {
    node.mode = mode;
    node.graph.change();
}

function addNodesToGroup(group, nodes=[]) {
    var x1, y1, x2, y2;
    var nx1, ny1, nx2, ny2;
    var node;

    x1 = y1 = x2 = y2 = -1;
    nx1 = ny1 = nx2 = ny2 = -1;

    for (var n of [group._nodes, nodes]) {
        for (var i in n) {
            node = n[i]

            nx1 = node.pos[0]
            ny1 = node.pos[1]
            nx2 = node.pos[0] + node.size[0]
            ny2 = node.pos[1] + node.size[1]

            if (node.type != "Reroute") {
                ny1 -= LiteGraph.NODE_TITLE_HEIGHT;
            }

            if (node.flags?.collapsed) {
                ny2 = ny1 + LiteGraph.NODE_TITLE_HEIGHT;

                if (node?._collapsed_width) {
                    nx2 = nx1 + Math.round(node._collapsed_width);
                }
            }

            if (x1 == -1 || nx1 < x1) {
                x1 = nx1;
            }

            if (y1 == -1 || ny1 < y1) {
                y1 = ny1;
            }

            if (x2 == -1 || nx2 > x2) {
                x2 = nx2;
            }

            if (y2 == -1 || ny2 > y2) {
                y2 = ny2;
            }
        }
    }

    var padding = 10;

    y1 = y1 - Math.round(group.font_size * 1.4);

    group.pos = [x1 - padding, y1 - padding];
    group.size = [x2 - x1 + padding * 2, y2 - y1 + padding * 2];
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
                options.push({
                    content: "Add Group For Selected Nodes",
                    disabled: !Object.keys(app.canvas.selected_nodes || {}).length,
                    callback: () => {
                        var group = new LiteGraph.LGraphGroup();
                        addNodesToGroup(group, this.selected_nodes)
                        app.canvas.graph.add(group);
                        this.graph.change();
                    }
                });

                return options;
            }

            // Group nodes aren't recomputed until the group is moved, this ensures the nodes are up-to-date
            group.recomputeInsideNodes();
            const nodesInGroup = group._nodes;

            options.push({
                content: "Add Selected Nodes To Group",
                disabled: !Object.keys(app.canvas.selected_nodes || {}).length,
                callback: () => {
                    addNodesToGroup(group, this.selected_nodes)
                    this.graph.change();
                }
            });

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

            options.push({
                content: "Fit Group To Nodes",
                callback: () => {
                    addNodesToGroup(group)
                    this.graph.change();
                }
            });

            options.push({
                content: "Select Nodes",
                callback: () => {
                    this.selectNodes(nodesInGroup);
                    this.graph.change();
                    this.canvas.focus();
                }
            });

            // Modes
            // 0: Always
            // 1: On Event
            // 2: Never
            // 3: On Trigger
            // 4: Bypass
            // If all nodes are the same mode, add a menu option to change the mode
            if (allNodesAreSameMode) {
                const mode = nodesInGroup[0].mode;
                switch (mode) {
                    case 0:
                        // All nodes are always, option to disable, and bypass
                        options.push({
                            content: "Set Group Nodes to Never",
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
                        break;
                    case 2:
                        // All nodes are never, option to enable, and bypass
                        options.push({
                            content: "Set Group Nodes to Always",
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
                        });
                        break;
                    case 4:
                        // All nodes are bypass, option to enable, and disable
                        options.push({
                            content: "Set Group Nodes to Always",
                            callback: () => {
                                for (const node of nodesInGroup) {
                                    setNodeMode(node, 0);
                                }
                            }
                        });
                        options.push({
                            content: "Set Group Nodes to Never",
                            callback: () => {
                                for (const node of nodesInGroup) {
                                    setNodeMode(node, 2);
                                }
                            }
                        });
                        break;
                    default:
                        // All nodes are On Trigger or On Event(Or other?), option to disable, set to always, or bypass
                        options.push({
                            content: "Set Group Nodes to Always",
                            callback: () => {
                                for (const node of nodesInGroup) {
                                    setNodeMode(node, 0);
                                }
                            }
                        });
                        options.push({
                            content: "Set Group Nodes to Never",
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
                        break;
                }
            } else {
                // Nodes are not all the same mode, add a menu option to change the mode to always, never, or bypass
                options.push({
                    content: "Set Group Nodes to Always",
                    callback: () => {
                        for (const node of nodesInGroup) {
                            setNodeMode(node, 0);
                        }
                    }
                });
                options.push({
                    content: "Set Group Nodes to Never",
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
