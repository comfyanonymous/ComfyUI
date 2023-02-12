var COMFY_WIDGETS = [];


class BaseWidget {
  addWidget(node) {
  }
}


class IntWidget extends BaseWidget {
  constructor(opts) {
    super();
    this.default_val = opts['default'] || 0;
    this.min_val = opts['min'] || 0;
    this.max_val = opts['max'] || 2048;
    this.step_val = opts['step'] || 1;
  }

  addWidget(node, x) {
    let onSet = function(v) {
      let s = this.options.step / 10;
      this.value = Math.round( v / s ) * s;
    };
    let w = node.addWidget("number", x, this.default_val, onSet, { min: this.min_val, max: this.max_val, step: this.step_val});
    node._widgets += [w]
    if (x == "seed" || x == "noise_seed") {
      let w1 = node.addWidget("toggle", "Random seed after every gen", true, function(v){}, { on: "enabled", off: "disabled" } );
      w1.to_randomize = w;
      node._widgets += [w1]
    }
  }
}
COMFY_WIDGETS["INT"] = IntWidget;


class FloatWidget extends BaseWidget {
  constructor(opts) {
    super();
    this.default_val = opts['default'] || 0;
    this.min_val = opts['min'] || 0;
    this.max_val = opts['max'] || 2048;
    this.step_val = opts['step'] || 0.5;
  }

  addWidget(node, x) {
    // if (min_val == 0.0 && max_val == 1.0) {
    //     w = this.slider = this.addWidget("slider", x, default_val, function(v){}, { min: min_val, max: max_val} );
    // } else {
    let w = node.addWidget("number", x, this.default_val, function(v){}, { min: this.min_val, max: this.max_val, step: this.step_val} );
    // }
    node._widgets += [w];
  }
}
COMFY_WIDGETS["FLOAT"] = FloatWidget;


class StringWidget extends BaseWidget {
  constructor(opts) {
    super();
    this.default_val = opts['default'] || "";
    this.multiline = opts['multiline'] || false;
  }

  addWidget(node, x) {
    if (this.multiline) {
      var w = {
        type: "customtext",
        name: x,
        get value() { return this.input_div.innerText;},
        set value(x) { this.input_div.innerText = x;},
        callback: function(v){console.log(v);},
        options: {},
        draw: function(ctx, node, widget_width, y, H){
          var show_text = canvas.ds.scale > 0.5;
          // this.input_div.style.top = `${y}px`;
          let t = ctx.getTransform();
          let margin = 10;
          let x_div = t.a * margin * 2 + t.e;
          let y_div = t.d * (y + H) + t.f;
          let width_div = (widget_width - margin * 2 - 3) * t.a;
          let height_div = (this.parent.size[1] - (y + H) - 3)* t.d;
          this.input_div.style.left = `${x_div}px`;
          this.input_div.style.top = `${y_div}px`;
          this.input_div.style.width = width_div;
          this.input_div.style.height = height_div;
          this.input_div.style.position = 'absolute';
          this.input_div.style.zIndex = 1;
          this.input_div.style.fontSize = t.d * 10.0;

          if (show_text) {
            this.input_div.hidden = false;
          } else {
            this.input_div.hidden = true;
          }

          ctx.save();
          // ctx.fillText(String(this.value).substr(0,30), 0, y + H * 0.7);
          ctx.restore();
        },
      };
      w.input_div = document.createElement('div');
      w.input_div.contentEditable = true;
      w.input_div.style.backgroundColor = "#FFFFFF";
      w.input_div.style.overflow = 'hidden';
      w.input_div.style.overflowY = 'auto';
      w.input_div.style.padding = 2;
      w.input_div.innerText = this.default_val;
      document.addEventListener('click', function(event) {
        if (!w.input_div.contains(event.target)) {
          w.input_div.blur();
        }
      });
      w.parent = node;
      min_height = Math.max(min_height, 200);
      min_width = Math.max(min_width, 400);
      ccc.parentNode.appendChild(w.input_div);

      w = node.addCustomWidget(w);
      canvas.onDrawBackground = function() {
        for (let n in graph._nodes) {
          n = graph._nodes[n];
          for (let w in n.widgets) {
            let wid = n.widgets[w];
            if (Object.hasOwn(wid, 'input_div')) {
              wid.input_div.style.left = -8000;
              wid.input_div.style.position = 'absolute';
            }
          }
        }
      }
      // w = node.addWidget("text", x, "", function(v){}, { multiline:true } );
      console.log(w, node);
      node._widgets += [w]
      node.onRemoved = function() {
        for (let y in node.widgets) {
          if (node.widgets[y].input_div) {
            node.widgets[y].input_div.remove();
          }
        }
      }
    }
    else {
      w = node.addWidget("text", x, this.default_val, function(v){}, { multiline:false } );
      node._widgets += [w];
    }
  }
}
COMFY_WIDGETS["STRING"] = StringWidget;


class ComboWidget extends BaseWidget {
  constructor(opts) {
    super();
    this.choices = opts['choices'] || [];
  }

  addWidget(node, x) {
    let w = node.addWidget("combo", x, this.choices[0], function(v){}, { values: this.choices } );
    node._widgets += [w]
  }
}
COMFY_WIDGETS["COMBO"] = ComboWidget;


class RegionWidget extends BaseWidget {
  constructor(opts) {
    super();
  }

  addWidget(node, x) {
    var w = {
      type: "region",
      name: x,
      region: { x: 0, y: 0, width: 2048, height: 2048 },
      get value() {
          return this.region;
      },
      set value(x) {
          this.region = x;
      },
      callback: function(v){console.log("CB!", v);},
      options: {},
      draw: function(ctx, node, widget_width, y, H){
        ctx.save();

        var size = this.size[1] * 0.5;
        var margin = 0.25;
        var h = this.size[1] * 0.8;
        // ctx.font = this.properties.font || (size * 0.8).toFixed(0) + "px Arial";
        var w = ctx.measureText(this.title).width;
        var x = margin * this.size[0] * 0.25;

        var latentParams = node.getInputData(1, true);
        if (latentParams) {
            var latentWidth = latentParams[0];
            var latentHeight = latentParams[1];

            var widgetWidth = node.widgets[1]
            var widgetHeight = node.widgets[2]
            var widgetX = node.widgets[3]
            var widgetY = node.widgets[4]
            widgetWidth.options.max = latentWidth;
            widgetWidth.value = Math.min(widgetWidth.value, latentWidth);
            this.region.width = widgetWidth.value;

            widgetHeight.options.max = latentHeight;
            widgetHeight.value = Math.min(widgetHeight.value, latentHeight);
            this.region.height = widgetHeight.value;

            widgetX.options.max = latentWidth;
            widgetX.value = Math.min(widgetX.value, latentWidth - widgetWidth.value);
            this.region.x = widgetX.value;

            widgetY.options.max = latentHeight;
            widgetY.value = Math.min(widgetY.value, latentHeight - widgetHeight.value);
            this.region.y = widgetY.value;

            ctx.fillStyle = "#FFF";
            ctx.strokeRect(x, y, this.size[0], this.size[1]);

            ctx.fillStyle = "#AAF";
            ctx.fillRect(
                x + (this.region.x / latentWidth) * this.size[0],
                y + (this.region.y / latentHeight) * this.size[1],
                (this.region.width / latentWidth) * this.size[0],
                (this.region.height / latentHeight) * this.size[1]
            );
        }

        // ctx.textAlign = "left";
        // ctx.fillStyle = "#AAA";
        // ctx.fillText(this.title, size * 1.2 + x, h * 0.85);
        // ctx.textAlign = "left";

        ctx.restore();
      },
      mouse: function(event, pos, node) {
        return true;
      },
      computeSize() {
        return this.size;
      }
    };
    w.size = [150, 150];
    w.parent = node;
    w = node.addCustomWidget(w);
    node.addWidget("number", "width",  512, (v) => w.region.width = v,  { property: "region_width",  min: 64, max: 2048, step: 64 });
    node.addWidget("number", "height", 512, (v) => w.region.height = v, { property: "region_height", min: 64, max: 2048, step: 64 });
    node.addWidget("number", "x",      0,   (v) => w.region.x = v,      { property: "region_x",      min: 0,  max: 2048, step: 64 });
    node.addWidget("number", "y",      0,   (v) => w.region.y = v,      { property: "region_y",      min: 0,  max: 2048, step: 64 });
    // console.log(node.getInputData("Width"))
    node._widgets += [w]
    console.log(node);
  }
}
COMFY_WIDGETS["REGION"] = RegionWidget;
