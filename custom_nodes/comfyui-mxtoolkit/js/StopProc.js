// ComfyUI.mxToolkit.Stop v.0.9.7 - Max Smirnov 2024
import { app } from "../../scripts/app.js";

class MXStop
{
    constructor(node)
    {
        this.node = node;
        this.node.properties = this.node.properties || {};

        this.node.onGraphConfigured = function ()
        {
            this.configured = true;
        }

        this.node.onConnectionsChange = function (type, index, connected, link_info)
        {
            if (link_info)
            {
                if (connected)
                {
                    if (type === LiteGraph.INPUT)
                    {
                        const cnode = app.graph.getNodeById(link_info.origin_id);
                        const ctype = cnode.outputs[link_info.origin_slot].type;
                        const color = LGraphCanvas.link_type_colors[ctype];
                        this.outputs[0].type = ctype;
                        this.outputs[0].name = ctype;
                        this.inputs[0].type = ctype;
                        if (link_info.id) { app.graph.links[link_info.id].color = color; }
                        if (this.outputs[0].links !== null)
                            for (let i = this.outputs[0].links.length; i > 0; i--)
                            {
                                const tlinkId = this.outputs[0].links[i-1];
                                const tlink = app.graph.links[tlinkId];
                                if (this.configured) if ( ctype !== tlink.type ) app.graph.getNodeById(tlink.target_id).disconnectInput(tlink.target_slot);
                            }
                    }
                    if (type === LiteGraph.OUTPUT && this.inputs[0].link === null)
                    {
                        this.inputs[0].type = link_info.type;
                        this.outputs[0].type = link_info.type;
                        this.outputs[0].name = link_info.type;
                    }
                }
                else if ((( type === LiteGraph.INPUT ) && ( this.outputs[0].links === null || this.outputs[0].links.length === 0 )) || (( type === LiteGraph.OUTPUT) && ( this.inputs[0].link === null ))) this.onAdded();
            }
            this.computeSize();
        };

        this.node.onAdded = function ()
        {
            this.inputs[0].type = "*";
            this.outputs[0].name = "";
            this.outputs[0].type = "*";
        };

        this.node.onMouseDown = function(e, pos, canvas)
        {
            let cWidth = this._collapsed_width || LiteGraph.NODE_COLLAPSED_WIDTH;
            if ( e.canvasY-this.pos[1] > 0 ) return false;
            if (this.flags.collapsed && (e.canvasX-this.pos[0] < LiteGraph.NODE_TITLE_HEIGHT)) return false;
            if (!this.flags.collapsed && ((e.canvasX-this.pos[0]) < (this.size[0]-cWidth+LiteGraph.NODE_TITLE_HEIGHT))) return false;
            this.updateThisNodeGraph?.();
            this.onTmpMouseUp(e, pos, canvas);
            return true;
        }

        this.node.onTmpMouseUp = function(e, pos, canvas)
        {
            app.queuePrompt(0);
        }

        this.node.onDrawForeground = function(ctx)
        {
            this.configured = true;
            if (this.size[1] > LiteGraph.NODE_SLOT_HEIGHT*1.3) this.size[1] = LiteGraph.NODE_SLOT_HEIGHT*1.3;
            let titleHeight = LiteGraph.NODE_TITLE_HEIGHT;
            let cWidth = this._collapsed_width || LiteGraph.NODE_COLLAPSED_WIDTH;
            let buttonWidth = cWidth-titleHeight-6;
            let cx = (this.flags.collapsed?cWidth:this.size[0])-buttonWidth-6;

            ctx.fillStyle = this.color || LiteGraph.NODE_DEFAULT_COLOR;
            ctx.beginPath();
            ctx.rect(cx, 2-titleHeight, buttonWidth, titleHeight-4);
            ctx.fill();

            cx += buttonWidth/2;

            ctx.lineWidth = 1;
            if (this.mouseOver)
            {
                ctx.fillStyle = LiteGraph.NODE_SELECTED_TITLE_COLOR
                ctx.beginPath(); ctx.moveTo(cx-8,-titleHeight/2-8); ctx.lineTo(cx+0,-titleHeight/2); ctx.lineTo(cx-8,-titleHeight/2+8); ctx.fill();
                ctx.beginPath(); ctx.moveTo(cx+1,-titleHeight/2-8); ctx.lineTo(cx+9,-titleHeight/2); ctx.lineTo(cx+1,-titleHeight/2+8); ctx.fill();
            }
            else
            {
                ctx.fillStyle = (this.boxcolor || LiteGraph.NODE_DEFAULT_BOXCOLOR);
                ctx.beginPath(); ctx.rect(cx-10,-titleHeight/2-8, 4, 16); ctx.fill();
                ctx.beginPath(); ctx.rect(cx-2,-titleHeight/2-8, 4, 16); ctx.fill();
            }
        }

        this.node.computeSize = function()
        {
            return [ (this.properties.showOutputText && this.outputs && this.outputs.length) ? LiteGraph.NODE_TEXT_SIZE * (this.outputs[0].name.length+5) * 0.6 + 140 : 140, LiteGraph.NODE_SLOT_HEIGHT*1.3 ];
        }
    }
}

app.registerExtension(
{
    name: "mxStop",
    async beforeRegisterNodeDef(nodeType, nodeData, _app)
    {
        if (nodeData.name === "mxStop")
        {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) onNodeCreated.apply(this, []);
                this.mxStop = new MXStop(this);
            }
        }
    }
});
