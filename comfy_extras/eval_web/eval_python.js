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

// Load Ace editor using script tag for Safari compatibility
// The noconflict build includes AMD loader that works in all browsers
let ace;
const aceLoadPromise = new Promise((resolve) => {
  if (window.ace) {
    ace = window.ace;
    resolve();
  } else {
    const script = document.createElement("script");
    script.src = "https://cdn.jsdelivr.net/npm/ace-builds@1.43.4/src-noconflict/ace.js";
    script.onload = () => {
      ace = window.ace;
      ace.config.set("basePath", "https://cdn.jsdelivr.net/npm/ace-builds@1.43.4/src-noconflict");
      resolve();
    };
    document.head.appendChild(script);
  }
});

// todo: do we really want to do this here?
await aceLoadPromise;
const findWidget = (node, value, attr = "name", func = "find") => {
  return node?.widgets ? node.widgets[func]((w) => (Array.isArray(value) ? value.includes(w[attr]) : w[attr] === value)) : null;
};

const makeElement = (tag, attrs = {}) => {
  if (!tag) tag = "div";
  const element = document.createElement(tag);
  Object.keys(attrs).forEach((key) => {
    const currValue = attrs[key];
    if (key === "class") {
      if (Array.isArray(currValue)) {
        element.classList.add(...currValue);
      } else if (currValue instanceof String || typeof currValue === "string") {
        element.className = currValue;
      }
    } else if (key === "dataset") {
      try {
        if (Array.isArray(currValue)) {
          currValue.forEach((datasetArr) => {
            const [prop, propval] = Object.entries(datasetArr)[0];
            element.dataset[prop] = propval;
          });
        } else {
          Object.entries(currValue).forEach((datasetArr) => {
            const [prop, propval] = datasetArr;
            element.dataset[prop] = propval;
          });
        }
      } catch (err) {
        // todo: what is this trying to do?
      }
    } else if (key === "style") {
      if (typeof currValue === "object" && !Array.isArray(currValue) && Object.keys(currValue).length) {
        Object.assign(element[key], currValue);
      } else if (typeof currValue === "object" && Array.isArray(currValue) && currValue.length) {
        element[key] = [...currValue];
      } else if (currValue instanceof String || typeof currValue === "string") {
        element[key] = currValue;
      }
    } else if (["for"].includes(key)) {
      element.setAttribute(key, currValue);
    } else if (key === "children") {
      element.append(...(currValue instanceof Array ? currValue : [currValue]));
    } else if (key === "parent") {
      currValue.append(element);
    } else {
      element[key] = currValue;
    }
  });
  return element;
};

const getPosition = (node, ctx, w_width, y, n_height) => {
  const margin = 5;

  const rect = ctx.canvas.getBoundingClientRect();
  const transform = ctx.getTransform();
  const scale = app.canvas.ds.scale;

  // The context is already transformed to draw at the widget position
  // transform.e and transform.f give us the canvas coordinates (in canvas pixels)
  // We need to convert these to screen pixels by accounting for the canvas scale
  // rect gives us the canvas element's position on the page

  // The transform matrix has scale baked in (transform.a = transform.d = scale)
  // transform.e and transform.f are the translation in canvas-pixel space
  const canvasPixelToScreenPixel = rect.width / ctx.canvas.width;

  const x = transform.e * canvasPixelToScreenPixel + rect.left;
  const y_pos = transform.f * canvasPixelToScreenPixel + rect.top;

  // Convert widget dimensions from canvas coordinates to screen pixels
  const scaledWidth = w_width * scale;
  const scaledHeight = (n_height - y - 15) * scale;
  const scaledMargin = margin * scale;
  const scaledY = y * scale;

  return {
    left: `${x + scaledMargin}px`,
    top: `${y_pos + scaledY + scaledMargin}px`,
    width: `${scaledWidth - scaledMargin * 2}px`,
    maxWidth: `${scaledWidth - scaledMargin * 2}px`,
    height: `${scaledHeight - scaledMargin * 2}px`,
    maxHeight: `${scaledHeight - scaledMargin * 2}px`,
    position: "absolute",
    scrollbarColor: "var(--descrip-text) var(--bg-color)",
    scrollbarWidth: "thin",
    zIndex: app.graph._nodes.indexOf(node),
  };
};

// Create code editor widget
const codeEditor = (node, inputName, inputData) => {
  const widget = {
    type: "code_block_python",
    name: inputName,
    options: { hideOnZoom: true },
    value: inputData[1]?.default || "",
    draw(ctx, node, widgetWidth, y) {
      const hidden = node.flags?.collapsed || (!!this.options.hideOnZoom && app.canvas.ds.scale < 0.5) || this.type === "converted-widget" || this.type === "hidden" || this.type === "converted-widget";

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
};

// Trigger workflow change tracking
const markWorkflowChanged = () => {
  app?.extensionManager?.workflow?.activeWorkflow?.changeTracker?.checkState();
};

// Register extensions
app.registerExtension({
  name: "Comfy.EvalPython",
  getCustomWidgets(app) {
    return {
      CODE_BLOCK_PYTHON: (node, inputName, inputData) => {
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
    // Handle all EvalPython node variants
    if (nodeData.name.startsWith("EvalPython")) {
      const originalOnConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function (info) {
        originalOnConfigure?.apply(this, arguments);

        if (info?.widgets_values?.length) {
          const widgetCodeIndex = findWidget(this, "code_block_python", "type", "findIndex");
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
