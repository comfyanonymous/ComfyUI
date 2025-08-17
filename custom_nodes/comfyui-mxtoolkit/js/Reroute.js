// ComfyUI.mxToolkit.Reroute v.0.9.6 - Max Smirnov 2024
import { app } from "../../scripts/app.js";
import { mergeIfValid, getWidgetConfig, setWidgetConfig } from "../core/widgetInputs.js";

app.registerExtension({
    name: "Comfy.MxRerouteNode",
    registerCustomNodes(app) {
        class MxRerouteNode extends LGraphNode {
            constructor(title = MxRerouteNode.title) {
                super(title);
                if (!this.properties) {
                    this.properties = {};
                }
                this.addInput("", "*");
                this.addOutput("", "*");

                this.properties.inputDir = "LEFT";
                this.properties.outputDir= "RIGHT";
                this.linkType = "*";
                this.bgcolor="#0000";
                this.keyCode = 0;

                this.size = [2*LiteGraph.CANVAS_GRID_SIZE + 1.4*LiteGraph.NODE_SLOT_HEIGHT, 2*LiteGraph.CANVAS_GRID_SIZE + 1.4*LiteGraph.NODE_SLOT_HEIGHT];

                this.onAdded = function ()
                {
                    this.resizable = false;
                    this.ioOrientation();
                }

                this.onConfigure = function ()
                {
                    this.bgcolor="#0000";
                }

                this.onGraphConfigured = function ()
                {
                    this.configured = true;
                    this.onConnectionsChange();
                }

                this.onDrawBackground = function (ctx)
                {
                    const canvas = app.graph.list_of_graphcanvas[0];
                    let linkColor = LGraphCanvas.link_type_colors[this.linkType];
                    if (linkColor === "") linkColor = LiteGraph.LINK_COLOR;

                    if (this.inputs && this.outputs) if (this.inputs[0].pos && this.outputs[0].pos)
                    {
                        const drawLink = (ctx) =>
                        {
                            ctx.beginPath();
                            ctx.moveTo(this.inputs[0].pos[0], this.inputs[0].pos[1]);
                            if (canvas.links_render_mode < 2)
                            {
                                ctx.lineTo(this.size[0]/2, this.size[1]/2);
                                ctx.lineTo(this.outputs[0].pos[0], this.outputs[0].pos[1]);
                            } else ctx.quadraticCurveTo(this.size[0]/2, this.size[1]/2, this.outputs[0].pos[0], this.outputs[0].pos[1]);
                            ctx.stroke();
                        }
                        this.inputs[0].color_on = "#0000";
                        this.outputs[0].color_on = "#0000";
                        this.outputs[0].shape = LiteGraph.ROUND_SHAPE;
                        if (this.outputs[0].links) if (this.outputs[0].links.length) if (this.outputs[0].links.length > 0) this.outputs[0].shape = LiteGraph.GRID_SHAPE;

                        if (canvas) if (canvas.render_connections_border && canvas.ds.scale > 0.6)
                        {
                            ctx.lineWidth = canvas.connections_width + 4;
                            ctx.strokeStyle = "#0007";
                            drawLink(ctx);
                        }
                        ctx.lineWidth = canvas.connections_width;
                        ctx.strokeStyle = linkColor;
                        drawLink(ctx);
                    }

                    if (this.mouseOver) if (this.inputs && this.outputs) if (this.inputs[0].pos && this.outputs[0].pos)
                    {
                        ctx.lineWidth = 1;
                        ctx.strokeStyle = linkColor;
                        ctx.beginPath();
                        ctx.moveTo(this.pos[0], this.pos[1]);
                        ctx.roundRect(0, 0, this.size[0], this.size[1], 10);
                        ctx.stroke();
                        ctx.fillStyle = linkColor;
                        ctx.beginPath();
                        ctx.arc(this.inputs[0].pos[0], this.inputs[0].pos[1], 5, 0, 2 * Math.PI, false);
                        ctx.arc(this.outputs[0].pos[0], this.outputs[0].pos[1], 5, 0, 2 * Math.PI, false);
                        ctx.fill();
                    }
                }

                this.onKeyUp = function(e)
                {
                    if (e.keyCode < 37 || e.keyCode > 40) return;
                    if (this.keyCode > 0)
                    {
                        const arrowKeys = { 37: "LEFT", 38: "UP", 39: "RIGHT", 40: "DOWN"};
                        this.properties.inputDir  = arrowKeys[this.keyCode];
                        this.properties.outputDir = arrowKeys[e.keyCode];
                        this.onPropertyChanged();
                        this.keyCode = 0;
                    } else this.keyCode = e.keyCode;
                }

                this.onDeselected = () => {this.keyCode = 0};

                this.getExtraMenuOptions = function()
                {
                    const that = this;
                    const iDir = ["LEFT",  "RIGHT", "DOWN", "UP",   "LEFT", "LEFT", "RIGHT", "RIGHT", "UP",    "DOWN",  "DOWN", "UP"];
                    const oDir = ["RIGHT", "LEFT",  "UP",   "DOWN", "UP",   "DOWN", "UP",    "DOWN",  "RIGHT", "RIGHT", "LEFT", "LEFT"];
                    const key  = ["🠖",     "🠔",    "🠕",    "🠗",    "⮥",    "⮧",    "⮤",    "⮦",     "⮡",     "⮣",     "⮢",   "⮠"], options = [];
                    for (let i = 0; i < 12; i++) options.push({ content: key[i], callback: () => { that.properties.inputDir = iDir[i]; that.properties.outputDir = oDir[i]; that.onPropertyChanged(); }});
                    return options;
                };

                this.onPropertyChanged = function ()
                {
                    const aValues = ["LEFT","RIGHT","UP","DOWN","TOP"];
                    const sValues = ["L","R","U","D","T"]
                    this.properties.inputDir = this.properties.inputDir.toUpperCase();
                    this.properties.outputDir = this.properties.outputDir.toUpperCase();
                    if (sValues.indexOf(this.properties.inputDir)>=0) this.properties.inputDir = aValues[sValues.indexOf(this.properties.inputDir)];
                    if (sValues.indexOf(this.properties.outputDir)>=0) this.properties.outputDir = aValues[sValues.indexOf(this.properties.outputDir)];
                    if (this.properties.inputDir === "TOP") this.properties.inputDir = "UP";
                    if (this.properties.outputDir === "TOP") this.properties.inputDir = "UP";
                    this.properties.inputDir = aValues.includes(this.properties.inputDir)?this.properties.inputDir:"LEFT";
                    this.properties.outputDir= aValues.includes(this.properties.outputDir)?this.properties.outputDir:"RIGHT";
                    if (this.properties.inputDir === this.properties.outputDir) this.properties.outputDir=(this.properties.outputDir==="RIGHT")?"LEFT":"RIGHT";
                    this.ioOrientation();
                    app.graph.setDirtyCanvas(true, true);
                }

                this.ioOrientation = function ()
                {
                    let i = this.properties.inputDir;
                    let o = this.properties.outputDir;
                    switch (i)
                    {
                        case "LEFT":
                            this.inputs[0].pos = [0,0.7*LiteGraph.NODE_SLOT_HEIGHT+LiteGraph.CANVAS_GRID_SIZE];
                            this.inputs[0].dir = LiteGraph.LEFT;
                            break;
                        case "RIGHT":
                            this.inputs[0].pos = [this.size[0],0.7*LiteGraph.NODE_SLOT_HEIGHT+LiteGraph.CANVAS_GRID_SIZE];
                            this.inputs[0].dir = LiteGraph.RIGHT;
                            break;
                        case "UP":
                            this.inputs[0].pos = [this.size[0]/2,0];
                            this.inputs[0].dir = LiteGraph.UP;
                            break;
                        case "DOWN":
                            this.inputs[0].pos = [this.size[0]/2,this.size[1]];
                            this.inputs[0].dir = LiteGraph.DOWN;
                            break;
                    }
                    switch (o)
                    {
                        case "LEFT":
                            this.outputs[0].pos = [0,0.7*LiteGraph.NODE_SLOT_HEIGHT+LiteGraph.CANVAS_GRID_SIZE];
                            this.outputs[0].dir = LiteGraph.LEFT;
                            break;
                        case "RIGHT":
                            this.outputs[0].pos = [this.size[0],0.7*LiteGraph.NODE_SLOT_HEIGHT+LiteGraph.CANVAS_GRID_SIZE];
                            this.outputs[0].dir = LiteGraph.RIGHT;
                            break;
                        case "UP":
                            this.outputs[0].pos = [this.size[0]/2,0];
                            this.outputs[0].dir = LiteGraph.UP;
                            break;
                        case "DOWN":
                            this.outputs[0].pos = [this.size[0]/2,this.size[1]];
                            this.outputs[0].dir = LiteGraph.DOWN;
                            break;
                    }
                }

                this.onConnectionsChange = function (type, index, connected, link_info) {

                    if (this.configured && connected && type === LiteGraph.OUTPUT) {
                        const types = new Set(this.outputs[0].links.map((l) => app.graph.links[l].type).filter((t) => t !== "*"));
                        if (types.size > 1) {
                            const linksToDisconnect = [];
                            for (let i = 0; i < this.outputs[0].links.length - 1; i++) {
                                const linkId = this.outputs[0].links[i];
                                const link = app.graph.links[linkId];
                                linksToDisconnect.push(link);
                            }
                            for (const link of linksToDisconnect) {
                                const node = app.graph.getNodeById(link.target_id);
                                node.disconnectInput(link.target_slot);
                            }
                        }
                    }

                    let currentNode = this;
                    let updateNodes = [];
                    let inputType = null;
                    let inputNode = null;
                    while (currentNode) {
                        updateNodes.unshift(currentNode);
                        const linkId = currentNode.inputs[0].link;
                        if (linkId !== null) {
                            const link = app.graph.links[linkId];
                            if (!link) return;
                            const node = app.graph.getNodeById(link.origin_id);
                            const type = node.constructor.type;
                            if (type && type.includes ("Reroute")) {
                                if (node === this) {
                                    currentNode.disconnectInput(link.target_slot);
                                    currentNode = null;
                                } else {
                                    currentNode = node;
                                }
                            } else {
                                inputNode = currentNode;
                                inputType = node.outputs[link.origin_slot]?.type ?? null;
                                break;
                            }
                        } else {
                            currentNode = null;
                            break;
                        }
                    }

                    const nodes = [this];
                    let outputType = null;
                    while (nodes.length) {
                        currentNode = nodes.pop();
                        const outputs = (currentNode.outputs ? currentNode.outputs[0].links : []) || [];
                        if (outputs.length) {
                            for (const linkId of outputs) {
                                const link = app.graph.links[linkId];

                                if (!link) continue;

                                const node = app.graph.getNodeById(link.target_id);
                                const type = node.constructor.type;

                                if (type && type.includes("Reroute")) {
                                    nodes.push(node);
                                    updateNodes.push(node);
                                } else {
                                    const nodeOutType =
                                        node.inputs && node.inputs[link?.target_slot] && node.inputs[link.target_slot].type
                                            ? node.inputs[link.target_slot].type
                                            : null;
                                    if (this.configured && inputType && inputType !== "*" && nodeOutType !== inputType) {
                                        node.disconnectInput(link.target_slot);
                                    } else {
                                        outputType = nodeOutType;
                                    }
                                }
                            }
                        }
                    }

                    this.linkType = inputType || outputType || "*";
                    const linkColor = LGraphCanvas.link_type_colors[this.linkType];

                    let widgetConfig;
                    let targetWidget;
                    let widgetType;
                    for (const node of updateNodes) {
                        node.outputs[0].type = inputType || "*";
                        node.__outputType = this.linkType;
                        node.outputs[0].name = node.properties.showOutputText ? this.linkType : "";
                        if (node.linkType) node.linkType = this.linkType;
                        node.size = node.computeSize();
                        if (node.applyOrientation) node.applyOrientation();

                        for (const l of node.outputs[0].links || []) {
                            const link = app.graph.links[l];
                            if (link) {
                                link.color = linkColor;

                                if (app.configuringGraph) continue;
                                const targetNode = app.graph.getNodeById(link.target_id);
                                const targetInput = targetNode.inputs?.[link.target_slot];
                                if (targetInput?.widget) {
                                    const config = getWidgetConfig(targetInput);
                                    if (!widgetConfig) {
                                        widgetConfig = config[1] ?? {};
                                        widgetType = config[0];
                                    }
                                    if (!targetWidget) {
                                        targetWidget = targetNode.widgets?.find((w) => w.name === targetInput.widget.name);
                                    }

                                    const merged = mergeIfValid(targetInput, [config[0], widgetConfig]);
                                    if (merged.customConfig) {
                                        widgetConfig = merged.customConfig;
                                    }
                                }
                            }
                        }
                    }

                    for (const node of updateNodes) {
                        if (widgetConfig && outputType) {
                            node.inputs[0].widget = { name: "value" };
                            setWidgetConfig(node.inputs[0], [widgetType ?? this.linkType, widgetConfig], targetWidget);
                        } else {
                            setWidgetConfig(node.inputs[0], null);
                        }
                    }

                    if (inputNode) {
                        const link = app.graph.links[inputNode.inputs[0].link];
                        if (link) {
                            link.color = linkColor;
                        }
                    }
                };

                this.clone = function () {
                    const cloned = MxRerouteNode.prototype.clone.apply(this);
                    cloned.removeOutput(0);
                    cloned.addOutput("", "*");
                    cloned.size = cloned.computeSize();
                    cloned.ioOrientation();
                    return cloned;
                };

                this.isVirtualNode = true;
            }

            computeSize() { return [2*LiteGraph.CANVAS_GRID_SIZE+1.4*LiteGraph.NODE_SLOT_HEIGHT,2*LiteGraph.CANVAS_GRID_SIZE+1.4*LiteGraph.NODE_SLOT_HEIGHT] }
        }

        LiteGraph.registerNodeType(
            "mxReroute",
            Object.assign(MxRerouteNode, {
                title_mode: LiteGraph.NO_TITLE,
                title: "mxReroute",
                collapsable: false,
            })
        );

        MxRerouteNode.category = "utils";
    },
});
