import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

const tokenPreviewWidgetName = "__tokens";

class TokenProgressHandler {
  constructor() {
    this.nodeOutputs = {};
    this.initEventListeners();
  }

  initEventListeners() {
    api.addEventListener("executing", ({ detail }) => {
      if (!detail) {
        return;
      }
      const nodeId = detail;
      if (!this.nodeOutputs[nodeId]) {
        this.nodeOutputs[nodeId] = {};
      }
      this.nodeOutputs[nodeId].tokens = null;
    });

    api.addEventListener("progress", ({ detail }) => {
      const nodeId = detail.node;
      if (!this.nodeOutputs[nodeId]) {
        this.nodeOutputs[nodeId] = {};
      }
      if (detail.output && detail.output.next_token) {
        if (!this.nodeOutputs[nodeId].tokens) {
          this.nodeOutputs[nodeId].tokens = "";
        }
        this.nodeOutputs[nodeId].tokens += detail.output.next_token;
        this.updateTokenWidget(nodeId, this.nodeOutputs[nodeId].tokens);
      }
      app.graph.setDirtyCanvas(true, false);
    });
  }

  updateTokenWidget(nodeId, tokens) {
    const node = app.graph.getNodeById(nodeId);
    if (node && node.widgets) {
      let widget = node.widgets.find((w) => w.name === tokenPreviewWidgetName);

      if (!widget) {
        widget = ComfyWidgets["STRING"](node, tokenPreviewWidgetName, ["STRING", { multiline: true }], app).widget;
        widget.inputEl.readOnly = true;
        widget.inputEl.style.opacity = 0.7;
      }
      widget.value = tokens;
      app.graph.setDirtyCanvas(true, false);
    }
  }
}

app.registerExtension({
  name: "Comfy.TokenProgress",
  setup() {
    this.tokenProgressHandler = new TokenProgressHandler();
  },
});
