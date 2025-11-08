/**
 * Uses code adapted from https://github.com/yorkane/ComfyUI-KYNode
 *
 * MIT License
 *
 * Copyright (c) 2024 Kevin Yuan
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
import { app } from "../../scripts/app.js";

import * as ace from "https://cdn.jsdelivr.net/npm/ace-code@1.43.4/+esm";
import { makeElement, findWidget } from "./ace_utils.js";

// Constants
const varTypes = ["int", "boolean", "string", "float", "json", "list", "dict"];
const typeMap = {
  int: "int",
  boolean: "bool",
  string: "str",
  float: "float",
  json: "json",
  list: "list",
  dict: "dict",
};

ace.config.setModuleLoader('ace/mode/python', () =>
    import('https://cdn.jsdelivr.net/npm/ace-builds@1.43.4/src/mode-python.js')
);

ace.config.setModuleLoader('ace/theme/monokai', () =>
    import('https://cdn.jsdelivr.net/npm/ace-builds@1.43.4/src/theme-monokai.js')
);

function getPostition(node, ctx, w_width, y, n_height) {
  const margin = 5;

  const rect = ctx.canvas.getBoundingClientRect();
  const transform = new DOMMatrix()
    .scaleSelf(rect.width / ctx.canvas.width, rect.height / ctx.canvas.height)
    .multiplySelf(ctx.getTransform())
    .translateSelf(margin, margin + y);
  const scale = new DOMMatrix().scaleSelf(transform.a, transform.d);

  return {
    transformOrigin: "0 0",
    transform: scale,
    left: `${transform.a + transform.e + rect.left}px`,
    top: `${transform.d + transform.f + rect.top}px`,
    maxWidth: `${w_width - margin * 2}px`,
    maxHeight: `${n_height - margin * 2 - y - 15}px`,
    width: `${w_width - margin * 2}px`,
    height: "90%",
    position: "absolute",
    scrollbarColor: "var(--descrip-text) var(--bg-color)",
    scrollbarWidth: "thin",
    zIndex: app.graph._nodes.indexOf(node),
  };
}

// Create editor code
function codeEditor(node, inputName, inputData) {
  const widget = {
    type: "pycode",
    name: inputName,
    options: { hideOnZoom: true },
    value:
      inputData[1]?.default ||
      `def my(a, b=1):
  return a * b<br>
    
r0 = str(my(23, 9))`,
    draw(ctx, node, widget_width, y, widget_height) {
      const hidden = node.flags?.collapsed || (!!widget.options.hideOnZoom && app.canvas.ds.scale < 0.5) || widget.type === "converted-widget" || widget.type === "hidden";

      widget.codeElement.hidden = hidden;

      if (hidden) {
        widget.options.onHide?.(widget);
        return;
      }

      Object.assign(this.codeElement.style, getPostition(node, ctx, widget_width, y, node.size[1]));
    },
    computeSize(...args) {
      return [500, 250];
    },
  };

  widget.codeElement = makeElement("pre", {
    innerHTML: widget.value,
  });

  widget.editor = ace.edit(widget.codeElement);
  widget.editor.setTheme("ace/theme/monokai");
  widget.editor.session.setMode("ace/mode/python");
  widget.editor.setOptions({
      enableAutoIndent: true,
      enableLiveAutocompletion: true,
      enableBasicAutocompletion: true,
    fontFamily: "monospace",
  });
  widget.codeElement.hidden = true;

  document.body.appendChild(widget.codeElement);

  const collapse = node.collapse;
  node.collapse = function () {
    collapse.apply(this, arguments);
    if (this.flags?.collapsed) {
      widget.codeElement.hidden = true;
    } else {
      if (this.flags?.collapsed === false) {
        widget.codeElement.hidden = false;
      }
    }
  };

  return widget;
}

// Save data to workflow forced!
function saveValue() {
  app?.extensionManager?.workflow?.activeWorkflow?.changeTracker?.checkState();
}

// Register extensions
app.registerExtension({
  name: "KYNode.KY_Eval_Python",
  getCustomWidgets(app) {
    return {
      PYCODE: (node, inputName, inputData, app) => {
        const widget = codeEditor(node, inputName, inputData);

        widget.editor.getSession().on("change", function (e) {
          widget.value = widget.editor.getValue();
          saveValue();
        });

        const varTypeList = node.addWidget(
          "combo",
          "select_type",
          "string",
          (v) => {
            // widget.editor.setTheme(`ace/theme/${varTypeList.value}`);
          },
          {
            values: varTypes,
            serialize: false,
          },
        );

        // 6. 使用 addDOMWidget 将容器添加到节点上
        //    - 第一个参数是 widget 的名称，在节点内部需要是唯一的。
        //    - 第二个参数是 widget 的类型，对于自定义 DOM 元素，通常是 "div"。
        //    - 第三个参数是您创建的 DOM 元素。
        //    - 第四个参数是一个选项对象，可以用来配置 widget。
        // node.addDOMWidget("rowOfButtons", "div", container, {
        // });
        node.addWidget("button", "Add Input variable", "add_input_variable", async () => {
          // Input name variable and check
          let nameInput = node?.inputs?.length ? `p${node.inputs.length - 1}` : "p0";

          const currentWidth = node.size[0];
          let tp = varTypeList.value;
          nameInput = nameInput + "_" + typeMap[tp];
          node.addInput(nameInput, "*");
          node.setSize([currentWidth, node.size[1]]);
          let cv = widget.editor.getValue();
          if (tp === "json") {
            cv = cv + "\n" + nameInput + " = json.loads(" + nameInput + ")";
          } else if (tp === "list") {
            cv = cv + "\n" + nameInput + " = []";
          } else if (tp === "dict") {
            cv = cv + "\n" + nameInput + " = {}";
          } else {
            cv = cv + "\n" + nameInput + " = " + typeMap[tp] + "(" + nameInput + ")";
          }
          widget.editor.setValue(cv);
          saveValue();
        });

        node.addWidget("button", "Add Output variable", "add_output_variable", async () => {
          const currentWidth = node.size[0];
          // Output name variable
          let nameOutput = node?.outputs?.length ? `r${node.outputs.length}` : "r0";
          let tp = varTypeList.value;
          nameOutput = nameOutput + "_" + typeMap[tp];
          node.addOutput(nameOutput, tp);
          node.setSize([currentWidth, node.size[1]]);
          let cv = widget.editor.getValue();
          if (tp === "json") {
            cv = cv + "\n" + nameOutput + " = json.dumps(" + nameOutput + ")";
          } else if (tp === "list") {
            cv = cv + "\n" + nameOutput + " = []";
          } else if (tp === "dict") {
            cv = cv + "\n" + nameOutput + " = {}";
          } else {
            cv = cv + "\n" + nameOutput + " = " + typeMap[tp] + "(" + nameOutput + ")";
          }
          widget.editor.setValue(cv);
          saveValue();
        });

        node.onRemoved = function () {
          for (const w of node?.widgets) {
            if (w?.codeElement) w.codeElement.remove();
          }
        };

        node.addCustomWidget(widget);

        return widget;
      },
    };
  },

  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    // --- IDENode
    if (nodeData.name === "KY_Eval_Python") {
      // Node Created
      const onNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = async function () {
        const ret = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

        const node_title = await this.getTitle();
        const nodeName = `${nodeData.name}_${this.id}`;

        this.name = nodeName;

        // Create default inputs, when first create node
        if (this?.inputs?.length < 2) {
          ["p0_str"].forEach((inputName) => {
            const currentWidth = this.size[0];
            this.addInput(inputName, "*");
            this.setSize([currentWidth, this.size[1]]);
          });
        }

        const widgetEditor = findWidget(this, "pycode", "type");

        this.setSize([530, this.size[1]]);

        return ret;
      };

      const onDrawForeground = nodeType.prototype.onDrawForeground;
      nodeType.prototype.onDrawForeground = function (ctx) {
        const r = onDrawForeground?.apply?.(this, arguments);

        // if (this.flags?.collapsed) return r;

        if (this?.outputs?.length) {
          for (let o = 0; o < this.outputs.length; o++) {
            const { name, type } = this.outputs[o];
            const colorType = LGraphCanvas.link_type_colors[type.toUpperCase()];
            const nameSize = ctx.measureText(name);
            const typeSize = ctx.measureText(`[${type === "*" ? "any" : type.toLowerCase()}]`);

            ctx.fillStyle = colorType === "" ? "#AAA" : colorType;
            ctx.font = "12px Arial, sans-serif";
            ctx.textAlign = "right";
            ctx.fillText(`[${type === "*" ? "any" : type.toLowerCase()}]`, this.size[0] - nameSize.width - typeSize.width, o * 20 + 19);
          }
        }

        if (this?.inputs?.length) {
          const not_showing = ["select_type", "pycode"];
          for (let i = 1; i < this.inputs.length; i++) {
            const { name, type } = this.inputs[i];
            if (not_showing.includes(name)) continue;
            const colorType = LGraphCanvas.link_type_colors[type.toUpperCase()];
            const nameSize = ctx.measureText(name);

            ctx.fillStyle = !colorType || colorType === "" ? "#AAA" : colorType;
            ctx.font = "12px Arial, sans-serif";
            ctx.textAlign = "left";
            ctx.fillText(`[${type === "*" ? "any" : type.toLowerCase()}]`, nameSize.width + 25, i * 20);
          }
        }
        return r;
      };

      // Node Configure
      const onConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function (node) {
        onConfigure?.apply(this, arguments);
        if (node?.widgets_values?.length) {
          const widget_code_id = findWidget(this, "pycode", "type", "findIndex");
          const widget_theme_id = findWidget(this, "varTypeList", "name", "findIndex");
          const widget_language_id = findWidget(this, "language", "name", "findIndex");

          const editor = this.widgets[widget_code_id]?.editor;

          if (editor) {
            // editor.setTheme(
            //   `ace/theme/${this.widgets_values[widget_theme_id]}`
            // );
            // editor.session.setMode(
            //   `ace/mode/${this.widgets_values[widget_language_id]}`
            // );
            editor.setValue(this.widgets_values[widget_code_id]);
            editor.clearSelection();
          }
        }
      };

      // ExtraMenuOptions
      const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
      nodeType.prototype.getExtraMenuOptions = function (_, options) {
        getExtraMenuOptions?.apply(this, arguments);

        const past_index = options.length - 1;
        const past = options[past_index];

        if (!!past) {
          // Inputs remove
          for (const input_idx in this.inputs) {
            const input = this.inputs[input_idx];

            if (["language", "select_type"].includes(input.name)) continue;

            options.splice(past_index + 1, 0, {
              content: `Remove Input ${input.name}`,
              callback: (e) => {
                const currentWidth = this.size[0];
                if (input.link) {
                  app.graph.removeLink(input.link);
                }
                this.removeInput(input_idx);
                this.setSize([80, this.size[1]]);
                saveValue();
              },
            });
          }

          // Output remove
          for (const output_idx in this.outputs) {
            const output = this.outputs[output_idx];

            if (output.name === "r0") continue;

            options.splice(past_index + 1, 0, {
              content: `Remove Output ${output.name}`,
              callback: (e) => {
                const currentWidth = this.size[0];
                if (output.link) {
                  app.graph.removeLink(output.link);
                }
                this.removeOutput(output_idx);
                this.setSize([currentWidth, this.size[1]]);
                saveValue();
              },
            });
          }
        }
      };
      // end - ExtraMenuOptions
    }
  },
});
