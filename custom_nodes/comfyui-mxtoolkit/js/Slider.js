// ComfyUI.mxToolkit.Slider v.0.9.92 - Max Smirnov 2025
import { app } from "../../scripts/app.js";

class MXSlider
{
    constructor(node)
    {
        this.node = node;
        this.node.properties = this.node.properties || {};
        this.node.properties.value=20;
        this.node.properties.min=0;
        this.node.properties.max=100;
        this.node.properties.step=1;
        this.node.properties.decimals=0;
        this.node.properties.snap=true;

        this.node.intpos = { x:0.2 };
        this.node.size = [210, Math.floor(LiteGraph.NODE_SLOT_HEIGHT*1.5)];
        const fontsize = LiteGraph.NODE_SUBTEXT_SIZE;
        const shX = (this.node.slot_start_y || 0)+fontsize*1.5;
        const shY = LiteGraph.NODE_SLOT_HEIGHT/1.5;
        const shiftLeft = 10;
        const shiftRight = 60;

        for (let i=0; i<3; i++) { this.node.widgets[i].hidden = true; this.node.widgets[i].type = "hidden"; }

        this.node.onAdded = function ()
        {
            this.outputs[0].name = this.outputs[0].localized_name = "";
            this.widgets_start_y = -2.4e8*LiteGraph.NODE_SLOT_HEIGHT;
            this.intpos.x = Math.max(0, Math.min(1, (this.properties.value-this.properties.min)/(this.properties.max-this.properties.min)));
            if (this.size) if (this.size.length) if (this.size[1] > LiteGraph.NODE_SLOT_HEIGHT*1.5) this.size[1] = LiteGraph.NODE_SLOT_HEIGHT*1.5;
            this.outputs[0].type = (this.properties.decimals > 0)?"FLOAT":"INT";
        };

        this.node.onConfigure = function ()
        {
            this.outputs[0].type = (this.properties.decimals > 0)?"FLOAT":"INT";
        }

        this.node.onGraphConfigured = function ()
        {
            this.configured = true;
            this.onPropertyChanged();
        }

        this.node.onPropertyChanged = function (propName)
        {
            if (!this.configured) return;
            if (this.properties.step <= 0) this.properties.step = 1;
            if ( isNaN(this.properties.value) ) this.properties.value = this.properties.min;
            if ( this.properties.min >= this.properties.max ) this.properties.max = this.properties.min+this.properties.step;
            if ((propName === "min") && (this.properties.value < this.properties.min)) this.properties.value = this.properties.min;
            if ((propName === "max") && (this.properties.value > this.properties.max)) this.properties.value = this.properties.max;
            this.properties.decimals = Math.floor(this.properties.decimals);
            if (this.properties.decimals>4) this.properties.decimals = 4;
            if (this.properties.decimals<0) this.properties.decimals = 0;
            this.properties.value = Math.round(Math.pow(10,this.properties.decimals)*this.properties.value)/Math.pow(10,this.properties.decimals);
            this.intpos.x = Math.max(0, Math.min(1, (this.properties.value-this.properties.min)/(this.properties.max-this.properties.min)));
            if ((this.properties.decimals > 0 && this.outputs[0].type !== "FLOAT") || (this.properties.decimals === 0 && this.outputs[0].type !== "INT"))
                if (this.outputs[0].links !== null)
                    for (let i = this.outputs[0].links.length; i > 0; i--)
                    {
                        const tlinkId = this.outputs[0].links[i-1];
                        const tlink = app.graph.links[tlinkId];
                        app.graph.getNodeById(tlink.target_id).disconnectInput(tlink.target_slot);
                    }
            this.outputs[0].type = (this.properties.decimals > 0)?"FLOAT":"INT";
            this.widgets[2].value = (this.properties.decimals > 0)?1:0;
            this.widgets[1].value = this.properties.value;
            this.widgets[0].value = Math.floor(this.properties.value);
        }

        this.node.onDrawForeground = function(ctx)
        {
            this.configured = true;
            if ( this.flags.collapsed ) return false;
            if (this.size[1] > LiteGraph.NODE_SLOT_HEIGHT*1.5) this.size[1] = LiteGraph.NODE_SLOT_HEIGHT*1.5;
            let dgt = parseInt(this.properties.decimals);

            ctx.fillStyle="rgba(20,20,20,0.5)";
            ctx.beginPath();
            ctx.roundRect( shiftLeft, shY-1, this.size[0]-shiftRight-shiftLeft, 4, 2);
            ctx.fill();

            ctx.fillStyle=LiteGraph.NODE_TEXT_COLOR;
            ctx.beginPath();
            ctx.arc(shiftLeft+(this.size[0]-shiftRight-shiftLeft)*this.intpos.x, shY+1, 7, 0, 2 * Math.PI, false);
            ctx.fill();

            ctx.lineWidth = 1.5;
            ctx.strokeStyle=node.bgcolor || LiteGraph.NODE_DEFAULT_BGCOLOR;
            ctx.beginPath();
            ctx.arc(shiftLeft+(this.size[0]-shiftRight-shiftLeft)*this.intpos.x, shY+1, 5, 0, 2 * Math.PI, false);
            ctx.stroke();

            ctx.fillStyle=LiteGraph.NODE_TEXT_COLOR;
            ctx.font = (fontsize) + "px Arial";
            ctx.textAlign = "center";
            ctx.fillText(this.properties.value.toFixed(dgt), this.size[0]-shiftRight+24, shX);
        }

        this.node.onDblClick = function(e, pos, canvas)
        {
            if ( e.canvasX > this.pos[0]+this.size[0]-shiftRight+10 )
            {
                canvas.prompt("value", this.properties.value, function(v) {if (!isNaN(Number(v))) { this.properties.value = Number(v); this.onPropertyChanged("value");}}.bind(this), e);
                return true;
            }
        }

        this.node.onMouseDown = function(e)
        {
            if ( e.canvasY - this.pos[1] < 0 ) return false;
            if ( e.canvasX < this.pos[0]+shiftLeft-5 || e.canvasX > this.pos[0]+this.size[0]-shiftRight+5 ) return false;
            if ( e.canvasY < this.pos[1]+shiftLeft-5 || e.canvasY > this.pos[1]+this.size[1]-shiftLeft+5 ) return false;

            this.capture = true;
            this.unlock = false;
            this.captureInput(true);
            this.valueUpdate(e);
            return true;
        }

        this.node.onMouseMove = function(e, pos, canvas)
        {
            if (!this.capture) return;
            if ( canvas.pointer.isDown === false ) { this.onMouseUp(e); return; }
            this.valueUpdate(e);
        }

        this.node.onMouseUp = function(e)
        {
            if (!this.capture) return;
            this.capture = false;
            this.captureInput(false);
            this.widgets[0].value = Math.floor(this.properties.value);
            this.widgets[1].value = this.properties.value;
        }

        this.node.valueUpdate = function(e)
        {
            let prevX = this.properties.value;
            let rn = Math.pow(10,this.properties.decimals);
            let vX = (e.canvasX - this.pos[0] - shiftLeft)/(this.size[0]-shiftRight-shiftLeft);

            if (e.ctrlKey) this.unlock = true;
            if (e.shiftKey !== this.properties.snap)
            {
                let step = this.properties.step/(this.properties.max - this.properties.min);
                vX = Math.round(vX/step)*step;
            }

            this.intpos.x = Math.max(0, Math.min(1, vX));
            this.properties.value = Math.round(rn*(this.properties.min + (this.properties.max - this.properties.min) * ((this.unlock)?vX:this.intpos.x)))/rn;

            this.updateThisNodeGraph?.();
            if ( this.properties.value !== prevX ) this.graph.setisChangedFlag(this.id);
        }

        this.node.onSelected = function(e) { this.onMouseUp(e) }
        this.node.computeSize = () => [LiteGraph.NODE_WIDTH,Math.floor(LiteGraph.NODE_SLOT_HEIGHT*1.5)];
    }
}

app.registerExtension(
{
    name: "mxSlider",
    async beforeRegisterNodeDef(nodeType, nodeData, _app)
    {
        if (nodeData.name === "mxSlider")
        {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) onNodeCreated.apply(this, []);
                this.mxSlider = new MXSlider(this);
            }
        }
    }
});
