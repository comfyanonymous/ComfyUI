import { app } from "../../../scripts/app.js";


app.registerExtension({
    name: "easy bookmark",
    registerCustomNodes() {
        class Bookmark {
            type = 'easy bookmark'
            title = "🔖";

            slot_start_y = -20;

            ___collapsed_width = 0;

            get _collapsed_width() {
                return this.___collapsed_width;
            }

            set _collapsed_width(width){
                const canvas = app.canvas ;
                const ctx = canvas.canvas.getContext('2d');
                if(ctx){
                   const oldFont = ctx.font;
                    ctx.font = canvas.title_text_font;
                    this.___collapsed_width = 40 +  ctx.measureText(this.title).width;
                    ctx.font = oldFont;
                }
            }

            isVirtualNode = true;
            serialize_widgets = true;
            keypressBound = null;

            constructor() {

                this.addWidget('text', 'shortcut_key', '1', (value) => {
                  value = value.trim()[0] || '1';
                  if(value !== ''){
                      this.title = "🔖 " + value;
                  }
                },{
                  y: 8,
                });
                this.addWidget('number', 'zoom', 1, (value) => {}, {
                  y: 8 + LiteGraph.NODE_WIDGET_HEIGHT + 4,
                  max: 2,
                  min: 0.5,
                  precision: 2,
                });
                this.keypressBound = this.onKeypress.bind(this);
            }

            onAdded(){
                setTimeout(_=>{
                    const value = this.widgets[0].value
                    if(value){
                        this.title = "🔖 " + value;
                    }
                },1)
                window.addEventListener("keydown", this.keypressBound);
            }

            onRemoved() {
                window.removeEventListener("keydown", this.keypressBound);
            }

            onKeypress(event){
                const target = event.target;
                if (['input','textarea'].includes(target.localName)) {
                  return;
                }
                if (this.widgets[0] && event.key.toLocaleLowerCase() === this.widgets[0].value.toLocaleLowerCase()) {
                  this.canvasToBookmark();
                }
            }

            canvasToBookmark() {
                const canvas = app.canvas;
                // ComfyUI seemed to break us again, but couldn't repro. No reason to not check, I guess.
                // https://github.com/rgthree/rgthree-comfy/issues/71
                if (canvas?.ds?.offset) {
                  canvas.ds.offset[0] = -this.pos[0]  + 16;
                  canvas.ds.offset[1] = -this.pos[1]  + 40;
                }
                if (canvas?.ds?.scale != null) {
                  canvas.ds.scale = Number(this.widgets[1].value || 1);
                }
                canvas.setDirty(true, true);
              }
        }

        LiteGraph.registerNodeType(
			"easy bookmark",
			Object.assign(Bookmark,{
                title: "Bookmark 🔖",
            })
		);

        Bookmark.category = "EasyUse/Util"
    }
})