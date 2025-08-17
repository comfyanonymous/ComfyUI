// ComfyUI.mxToolkit.Slider2D v.0.9.92 - Max Smirnov 2025
import { app } from "../../scripts/app.js";

class MXSlider2D
{
    constructor(node)
    {
        this.node = node;
        this.node.properties = { valueX:512, valueY:512, minX:0, minY:0, maxX:1024, maxY:1024, stepX:128, stepY:128, decimalsX:0, decimalsY:0, snap: true, dots: true, frame: true, frameAlert:0 };
        this.node.intpos = { x:0.5, y:0.5 };
        this.node.size = [210, 210];
        const fontsize = LiteGraph.NODE_SUBTEXT_SIZE;
        const shX = (this.node.slot_start_y || 0)+fontsize*1.5;
        const shY = shX + LiteGraph.NODE_SLOT_HEIGHT;
        const minSize = 60;
        const shiftLeft = 10;
        const shiftRight = 60;

        for (let i=0; i<6; i++) { this.node.widgets[i].hidden = true; this.node.widgets[i].type = "hidden"; }

        this.node.onAdded = function ()
        {
            this.outputs[0].name = this.outputs[0].localized_name = "";
            this.outputs[1].name = this.outputs[1].localized_name = "";
            this.widgets_start_y = -4.8e8*LiteGraph.NODE_SLOT_HEIGHT;
            this.intpos.x = (this.properties.valueX - this.properties.minX) / (this.properties.maxX - this.properties.minX);
            this.intpos.y = (this.properties.valueY - this.properties.minY) / (this.properties.maxY - this.properties.minY);
            if (this.size[1] > this.size[0]-shiftRight+shiftLeft) {this.size[1] = this.size[0]-shiftRight+shiftLeft} else {this.size[0] = this.size[1]+shiftRight-shiftLeft}
            this.outputs[0].type = (this.properties.decimalsX > 0)?"FLOAT":"INT";
            this.outputs[1].type = (this.properties.decimalsY > 0)?"FLOAT":"INT";
        };

        this.node.onConfigure = function ()
        {
            this.outputs[0].type = (this.properties.decimalsX > 0)?"FLOAT":"INT";
            this.outputs[1].type = (this.properties.decimalsY > 0)?"FLOAT":"INT";
        }

        this.node.onGraphConfigured = function ()
        {
            this.configured = true;
            this.onPropertyChanged();
        }

        this.node.onPropertyChanged = function (propName)
        {
            if (!this.configured) return;
            if (this.properties.stepX <= 0) this.properties.stepX = 1;
            if (this.properties.stepY <= 0) this.properties.stepY = 1;
            if ( isNaN(this.properties.valueX) ) this.properties.valueX = this.properties.minX;
            if ( isNaN(this.properties.valueY) ) this.properties.valueY = this.properties.minY;
            if ( this.properties.minX >= this.properties.maxX ) this.properties.maxX = this.properties.minX+1;
            if ( this.properties.minY >= this.properties.maxY ) this.properties.maxY = this.properties.minY+1;
            if ((propName === "minX") && ( this.properties.valueX < this.properties.minX )) this.properties.valueX = this.properties.minX;
            if ((propName === "minY") && ( this.properties.valueY < this.properties.minY )) this.properties.valueY = this.properties.minY;
            if ((propName === "maxX") && ( this.properties.valueX > this.properties.maxX )) this.properties.valueX = this.properties.maxX;
            if ((propName === "maxY") && ( this.properties.valueY > this.properties.maxY )) this.properties.valueY = this.properties.maxY;
            this.properties.decimalsX = Math.floor(this.properties.decimalsX);
            this.properties.decimalsY = Math.floor(this.properties.decimalsY);
            if (this.properties.decimalsX>4) this.properties.decimalsX = 4;
            if (this.properties.decimalsY>4) this.properties.decimalsY = 4;
            if (this.properties.decimalsX<0) this.properties.decimalsX = 0;
            if (this.properties.decimalsY<0) this.properties.decimalsY = 0;
            this.properties.valueX = Math.round(Math.pow(10,this.properties.decimalsX)*this.properties.valueX)/Math.pow(10,this.properties.decimalsX);
            this.properties.valueY = Math.round(Math.pow(10,this.properties.decimalsY)*this.properties.valueY)/Math.pow(10,this.properties.decimalsY);
            this.intpos.x = Math.max(0, Math.min(1, (this.properties.valueX-this.properties.minX)/(this.properties.maxX-this.properties.minX)));
            this.intpos.y = Math.max(0, Math.min(1, (this.properties.valueY-this.properties.minY)/(this.properties.maxY-this.properties.minY)));

            if ((this.properties.decimalsX > 0 && this.outputs[0].type !== "FLOAT") || (this.properties.decimalsX === 0 && this.outputs[0].type !== "INT"))
                if (this.outputs[0].links !== null)
                    for (let i = this.outputs[0].links.length; i > 0; i--)
                    {
                        const tlinkId = this.outputs[0].links[i-1];
                        const tlink = app.graph.links[tlinkId];
                        app.graph.getNodeById(tlink.target_id).disconnectInput(tlink.target_slot);
                    }
            if ((this.properties.decimalsY > 0 && this.outputs[1].type !== "FLOAT") || (this.properties.decimalsY === 0 && this.outputs[1].type !== "INT"))
                if (this.outputs[1].links !== null)
                    for (let i = this.outputs[1].links.length; i > 0; i--)
                    {
                        const tlinkId = this.outputs[1].links[i-1];
                        const tlink = app.graph.links[tlinkId];
                        app.graph.getNodeById(tlink.target_id).disconnectInput(tlink.target_slot);
                    }

            this.outputs[0].type = (this.properties.decimalsX > 0)?"FLOAT":"INT";
            this.outputs[1].type = (this.properties.decimalsY > 0)?"FLOAT":"INT";
            this.widgets[5].value = (this.properties.decimalsY > 0)?1:0;
            this.widgets[4].value = (this.properties.decimalsX > 0)?1:0;
            this.widgets[3].value = this.properties.valueY;
            this.widgets[2].value = Math.floor(this.properties.valueY);
            this.widgets[1].value = this.properties.valueX;
            this.widgets[0].value = Math.floor(this.properties.valueX);
        }

        this.node.onDrawForeground = function(ctx)
        {
            this.configured = true;
            if ( this.flags.collapsed ) return false;

            let dgtX = parseInt(this.properties.decimalsX);
            let dgtY = parseInt(this.properties.decimalsY);

            ctx.fillStyle="rgba(20,20,20,0.8)";
            ctx.beginPath();
            ctx.roundRect( shiftLeft-4, shiftLeft-4, this.size[0]-shiftRight-shiftLeft+8, this.size[1]-shiftLeft-shiftLeft+8,4);
            ctx.fill();

            // Dots
            if (this.properties.dots)
            {
                ctx.fillStyle="rgba(200,200,200,0.7)";
                ctx.beginPath();
                let swX = (this.size[0]-shiftRight-shiftLeft);
                let swY = (this.size[1]-shiftLeft-shiftLeft);
                let stX = (swX*this.properties.stepX/(this.properties.maxX-this.properties.minX));
                let stY = (swY*this.properties.stepY/(this.properties.maxY-this.properties.minY));
                for (var ix=0; ix<swX+stX/2; ix+=stX) for (var iy=0; iy<swY+stY/2; iy+=stY) ctx.rect(shiftLeft+ix-0.5, shiftLeft+iy-0.5, 1, 1);
                ctx.fill();
            }

            if (this.properties.frame)
            {
                if ((this.properties.frameAlert > 0) && (this.properties.valueX*this.properties.valueY > this.properties.frameAlert))
                {
                    ctx.fillStyle="rgba(250,0,0,0.2)";
                    ctx.strokeStyle="rgba(250,0,0,0.7)";
                }
                else
                {
                    ctx.fillStyle="rgba(200,200,200,0.1)";
                    ctx.strokeStyle="rgba(200,200,200,0.7)";
                }
                ctx.beginPath();
                ctx.rect(shiftLeft, shiftLeft+(this.size[1]-shiftLeft-shiftLeft)*(1-this.intpos.y),(this.size[0]-shiftRight-shiftLeft)*this.intpos.x,(this.size[1]-shiftLeft-shiftLeft)*(this.intpos.y));
                ctx.fill();
                ctx.stroke();
            }

            ctx.fillStyle=LiteGraph.NODE_TEXT_COLOR;
            ctx.beginPath();
            ctx.arc(shiftLeft+(this.size[0]-shiftRight-shiftLeft)*this.intpos.x, shiftLeft+(this.size[1]-shiftLeft-shiftLeft)*(1-this.intpos.y), 7, 0, 2 * Math.PI, false);
            ctx.fill();

            ctx.lineWidth = 1.5;
            ctx.strokeStyle=node.bgcolor || LiteGraph.NODE_DEFAULT_BGCOLOR;
            ctx.beginPath();
            ctx.arc(shiftLeft+(this.size[0]-shiftRight-shiftLeft)*this.intpos.x, shiftLeft+(this.size[1]-shiftLeft-shiftLeft)*(1-this.intpos.y), 5, 0, 2 * Math.PI, false);
            ctx.stroke();

            ctx.fillStyle=LiteGraph.NODE_TEXT_COLOR;
            ctx.font = (fontsize) + "px Arial";
            ctx.textAlign = "center";
            ctx.fillText(this.properties.valueX.toFixed(dgtX), this.size[0]-shiftRight+24, (shX));
            ctx.fillText(this.properties.valueY.toFixed(dgtY), this.size[0]-shiftRight+24, (shY));
        }

        this.node.onDblClick = function(e, pos, canvas)
        {
            if ( e.canvasY - this.pos[1] < 0 ) return false;
            if ( e.canvasX > this.pos[0]+this.size[0]-shiftRight+10 )
            {
                if (e.canvasY - this.pos[1] - 5 < shX)
                {
                    canvas.prompt("valueX", this.properties.valueX, function(v) {if (!isNaN(Number(v))) { this.properties.valueX = Number(v); this.onPropertyChanged("valueX");}}.bind(this), e);
                    return true;
                }
                else if (e.canvasY - this.pos[1] - 5 < shY)
                {
                    canvas.prompt("valueY", this.properties.valueY, function(v) {if (!isNaN(Number(v))) { this.properties.valueY = Number(v); this.onPropertyChanged("valueY");}}.bind(this), e);
                    return true;
                }
            }
        }

        this.node.onMouseDown = function(e)
        {
            if (e.canvasY - this.pos[1] < 0) return false;
            if (e.shiftKey &&
                e.canvasX > this.pos[0]+this.size[0]-shiftRight+10 &&
                e.canvasX < this.pos[0]+this.size[0]-15 &&
                e.canvasX > this.pos[0]+this.size[0]-shiftRight+15 &&
                e.canvasY < this.pos[1]+shY &&
                this.properties.decimalsX === this.properties.decimalsY &&
                this.properties.valueX <= this.properties.maxY &&
                this.properties.valueX >= this.properties.minY &&
                this.properties.valueY <= this.properties.maxX &&
                this.properties.valueY >= this.properties.minX)
            {
                let tmpX = this.properties.valueX;
                this.properties.valueX = this.properties.valueY;
                this.properties.valueY = tmpX;
                this.intpos.x = (this.properties.valueX-this.properties.minX)/(this.properties.maxX-this.properties.minX);
                this.intpos.y = (this.properties.valueY-this.properties.minY)/(this.properties.maxY-this.properties.minY);
                this.onPropertyChanged("valueX");
                this.onPropertyChanged("valueY");
                this.updateThisNodeGraph();
                this.graph.setisChangedFlag(this.id);
                return true;
            }

            if ( e.canvasX < this.pos[0]+shiftLeft-5 || e.canvasX > this.pos[0]+this.size[0]-shiftRight+5 ) return false;
            if ( e.canvasY < this.pos[1]+shiftLeft-5 || e.canvasY > this.pos[1]+this.size[1]-shiftLeft+5 )  return false;
            this.capture = true;
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

        this.node.valueUpdate = function(e)
        {
            let prevX = this.properties.valueX;
            let prevY = this.properties.valueY;
            let rnX = Math.pow(10,this.properties.decimalsX);
            let rnY = Math.pow(10,this.properties.decimalsY);
            let vX = (e.canvasX - this.pos[0] - shiftLeft)/(this.size[0]-shiftRight-shiftLeft);
            let vY = 1-(e.canvasY - this.pos[1] - shiftLeft)/(this.size[1]-shiftLeft-shiftLeft);
            if (e.shiftKey !== this.properties.snap)
            {
                let sX = this.properties.stepX/(this.properties.maxX - this.properties.minX);
                let sY = this.properties.stepY/(this.properties.maxY - this.properties.minY);
                vX = Math.round(vX/sX)*sX;
                vY = Math.round(vY/sY)*sY;
            }
            if ( vX < 0 ) { vX = 0 } else if ( vX > 1 ) { vX = 1 }
            if ( vY < 0 ) { vY = 0 } else if ( vY > 1 ) { vY = 1 }
            this.intpos.x = vX;
            this.intpos.y = vY;
            this.properties.valueX = Math.round(rnX*(this.properties.minX + (this.properties.maxX - this.properties.minX) * this.intpos.x))/rnX;
            this.properties.valueY = Math.round(rnY*(this.properties.minY + (this.properties.maxY - this.properties.minY) * this.intpos.y))/rnY;
            this.updateThisNodeGraph?.();
            if ( this.properties.valueX !== prevX || this.properties.valueY !== prevY ) this.graph.setisChangedFlag(this.id);
        }

        this.node.onMouseUp = function()
        {
            if (!this.capture) return;
            this.capture = false;
            this.captureInput(false);
            this.widgets[0].value = Math.floor(this.properties.valueX);
            this.widgets[1].value = this.properties.valueX;
            this.widgets[2].value = Math.floor(this.properties.valueY);
            this.widgets[3].value = this.properties.valueY;
        }

        this.node.onSelected = function(e) { this.onMouseUp(e) }
        this.node.computeSize = () => [minSize + shiftRight - shiftLeft, minSize];
    }
}

app.registerExtension(
{
    name: "mxSlider2D",
    async beforeRegisterNodeDef(nodeType, nodeData, _app)
    {
        if (nodeData.name === "mxSlider2D")
        {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) onNodeCreated.apply(this, []);
                this.mxSlider2d = new MXSlider2D(this);
            }
        }
    }
});
