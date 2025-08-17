// ComfyUI.mxToolkit.Seed v.0.9.9 - Max Smirnov 2024
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

class MXSeed {
    constructor(node) {

        this.node = node;
        this.node.properties = { seed:0, min:0, max:4294967296, autorunQueue: true, interruptQueue: true };
        this.node.size = [210,LiteGraph.NODE_SLOT_HEIGHT*3.4];
        this.node.widgets[0].hidden = true;
        this.node.widgets[0].type = "hidden";

        this.seedWidget = this.node.widgets[0];

        const fontsize = LiteGraph.NODE_TEXT_SIZE;

        this.node.onAdded = function ()
        {
            this.outputs[0].name = this.outputs[0].localized_name = "";
            this.widgets_start_y = 0;
            this.resizable = false;
            this.lastprocessed = null;
            this.configured = false;
            this.history=[this.properties.seed];
        };

        this.node.onConfigure = function ()
        {
            this.history=[this.properties.seed];
            this.configured = true;
        }

        this.node.onMouseDown = function(e, pos, canvas)
        {
            let cWidth = this._collapsed_width || LiteGraph.NODE_COLLAPSED_WIDTH;

            if (this.flags.collapsed && (( e.canvasY-this.pos[1] > 0 ) || (e.canvasX-this.pos[0] < LiteGraph.NODE_TITLE_HEIGHT))) return false;
            if (!this.flags.collapsed && ( e.canvasY-this.pos[1] < 0 ) && ((e.canvasX-this.pos[0]) < (this.size[0]-cWidth+LiteGraph.NODE_TITLE_HEIGHT))) return false;
            if ((e.canvasX-this.pos[0]) < LiteGraph.NODE_TITLE_HEIGHT) return false;

            this.updateThisNodeGraph?.();
            this.onTmpMouseUp(e, pos, canvas);
            return true;
        }

        this.node.onTmpMouseUp = function(e, pos, canvas)
        {
            if (!this.flags.collapsed && ( e.canvasY-this.pos[1] > 0 ))
            {
                let ps = Math.floor((e.canvasY-this.pos[1]-(LiteGraph.NODE_SLOT_HEIGHT-fontsize)/2)/LiteGraph.NODE_SLOT_HEIGHT);
                if (ps > 0 && ps < this.history.length)
                {
                    this.history.unshift(this.history[ps]);
                    this.history.splice(ps+1,1);
                    this.lastprocessed = null;
                    this.processSeed();
                }
                else if (ps === 0)
                {
                    if (this.configured) this.lastprocessed = this.history[0];
                    canvas.prompt("Seed", this.properties.seed, function(v) {if (!isNaN(Number(v))) this.processSeed(Number(v));}.bind(this), e);
                    return;
                }
            }
            if ( e.canvasY-this.pos[1] < 0 ) if (e.shiftKey) { this.processSeed((this.properties.seed<this.properties.max)?this.properties.seed+1:this.properties.min) } else { this.processSeed() }

        }

        this.node.onPropertyChanged = function ()
        {
            if ( this.properties.min < 0 ) this.properties.min = 0;
            if ( this.properties.max > 4294967296 ) this.properties.max = 4294967296;
            if ( this.properties.max < this.properties.min ) this.properties.max = this.properties.min + 1;
            this.randomRange = this.properties.max - this.properties.min;

            if (!this.configured) return;
            if (this.properties.seed !== this.lastprocessed) this.processSeed(this.properties.seed);
        }


        this.node.processSeed = function(s)
        {
            let newSeed;
            if (s === undefined)
            {
                do { newSeed = Math.round( Math.random() * (this.properties.max - this.properties.min) + this.properties.min ) } while (newSeed === this.lastprocessed);
            } else newSeed = s;
            if (this.lastprocessed === null && this.configured ) newSeed = this.history[0];
            if (newSeed !== this.history[0])
            {
                this.history.unshift(newSeed);
                if (this.history.length === 2 && this.lastprocessed === null && !this.configured) { this.history.splice(1); this.configured = true; }
                if (this.history.length > 3) { this.history.splice(3) }
            }

            this.lastprocessed = newSeed;
            this.properties.seed = newSeed;
            this.widgets[0].value = newSeed;

            if (this.properties.interruptQueue) api.interrupt();
            if (this.properties.autorunQueue) app.queuePrompt(0);
        }

        this.node.onDrawForeground = function(ctx)
        {
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
            ctx.fillStyle = this.mouseOver?LiteGraph.NODE_SELECTED_TITLE_COLOR:(this.boxcolor || LiteGraph.NODE_DEFAULT_BOXCOLOR);
            ctx.beginPath(); ctx.moveTo(cx-8,-titleHeight/2-8); ctx.lineTo(cx+3,-titleHeight/2); ctx.lineTo(cx-8,-titleHeight/2+8); ctx.fill();

            if (!this.flags.collapsed)
            {
                ctx.fillStyle="rgba(20,20,20,0.5)";
                ctx.beginPath();
                ctx.roundRect( 20, 5, this.size[0]-40, fontsize+6, 6);
                ctx.fill();

                ctx.strokeStyle=LiteGraph.NODE_TEXT_COLOR;
                ctx.beginPath();
                ctx.roundRect( 20, 5, this.size[0]-40, fontsize+6, 6);
                ctx.stroke();

                ctx.fillStyle=LiteGraph.NODE_SELECTED_TITLE_COLOR;
                ctx.font = (fontsize) + "px Arial";
                ctx.textAlign = "center";
                ctx.fillStyle=LiteGraph.NODE_TEXT_COLOR;
                for (let i=0; i<this.history.length; i++) ctx.fillText(this.history[i], this.size[0]/2, LiteGraph.NODE_SLOT_HEIGHT*(i+1));
            }
        }

        this.node.computeSize = function()
        {
            return [210,LiteGraph.NODE_SLOT_HEIGHT*3.4];
        }
    }
}

app.registerExtension(
{
    name: "mxSeed",
    async beforeRegisterNodeDef(nodeType, nodeData, _app)
    {
        if (nodeData.name === "mxSeed")
        {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function ()
            {
                if (onNodeCreated) onNodeCreated.apply(this, []);
                this.mxSeed = new MXSeed(this);
            }
        }
    }
});
