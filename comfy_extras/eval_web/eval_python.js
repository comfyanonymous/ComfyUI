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
import { makeElement, findWidget } from "./ace_utils.js";

// Load Ace editor using script tag for Safari compatibility
// The noconflict build includes AMD loader that works in all browsers
let ace;
const aceLoadPromise = new Promise((resolve) => {
  if (window.ace) {
    ace = window.ace;
    resolve();
  } else {
    const script = document.createElement('script');
    script.src = "https://cdn.jsdelivr.net/npm/ace-builds@1.43.4/src-noconflict/ace.js";
    script.onload = () => {
      ace = window.ace;
      ace.config.set("basePath", "https://cdn.jsdelivr.net/npm/ace-builds@1.43.4/src-noconflict");
      resolve();
    };
    document.head.appendChild(script);
  }
});

await aceLoadPromise;


function getPosition(node, ctx, w_width, y, n_height) {
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

// Create code editor widget
function codeEditor(node, inputName, inputData) {
  const widget = {
    type: "pycode",
    name: inputName,
    options: { hideOnZoom: true },
    value: inputData[1]?.default || "",
    draw(ctx, node, widgetWidth, y) {
      const hidden = node.flags?.collapsed || (!!this.options.hideOnZoom && app.canvas.ds.scale < 0.5) || this.type === "converted-widget" || this.type === "hidden";

      this.codeElement.hidden = hidden;

      if (hidden) {
        this.options.onHide?.(this);
        return;
      }

      Object.assign(this.codeElement.style, getPosition(node, ctx, widgetWidth, y, node.size[1]));
    },
    computeSize() {
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

  const originalCollapse = node.collapse;
  node.collapse = function () {
    originalCollapse.apply(this, arguments);
    widget.codeElement.hidden = !!this.flags?.collapsed;
  };

  return widget;
}

// Trigger workflow change tracking
function markWorkflowChanged() {
  app?.extensionManager?.workflow?.activeWorkflow?.changeTracker?.checkState();
}

// Register extensions
app.registerExtension({
  name: "Comfy.EvalPython",
  getCustomWidgets(app) {
    return {
      PYCODE: (node, inputName, inputData) => {
        const widget = codeEditor(node, inputName, inputData);

        widget.editor.getSession().on("change", () => {
          widget.value = widget.editor.getValue();
          markWorkflowChanged();
        });

        node.onRemoved = function () {
          for (const w of this.widgets) {
            if (w?.codeElement) {
              w.codeElement.remove();
            }
          }
        };

        node.addCustomWidget(widget);

        return widget;
      },
    };
  },

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name === "EvalPython") {
      const originalOnConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function (info) {
        originalOnConfigure?.apply(this, arguments);

        if (info?.widgets_values?.length) {
          const widgetCodeIndex = findWidget(this, "pycode", "type", "findIndex");
          const editor = this.widgets[widgetCodeIndex]?.editor;

          if (editor) {
            editor.setValue(info.widgets_values[widgetCodeIndex]);
            editor.clearSelection();
          }
        }
      };
    }
  },
});
