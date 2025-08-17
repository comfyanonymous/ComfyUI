import { app } from "../../../scripts/app.js";


app.registerExtension({
    name: "comfy.easyUse.imageWidgets",

    nodeCreated(node) {
        if (["easy imageSize","easy imageSizeBySide","easy imageSizeByLongerSide","easy imageSizeShow", "easy imageRatio", "easy imagePixelPerfect"].includes(node.comfyClass)) {

			const inputEl = document.createElement("textarea");
			inputEl.className = "comfy-multiline-input";
			inputEl.readOnly = true

			const widget = node.addDOMWidget("info", "customtext", inputEl, {
				getValue() {
					return inputEl.value;
				},
				setValue(v) {
					inputEl.value = v;
				},
				serialize: false
			});
			widget.inputEl = inputEl;

			inputEl.addEventListener("input", () => {
				widget.callback?.(widget.value);
			});
        }
    },

    beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (["easy imageSize","easy imageSizeBySide","easy imageSizeByLongerSide", "easy imageSizeShow", "easy imageRatio", "easy imagePixelPerfect"].includes(nodeData.name)) {
			function populate(arr_text) {
				var text = '';
				for (let i = 0; i < arr_text.length; i++){
					text += arr_text[i];
				}
				if (this.widgets) {
					const pos = this.widgets.findIndex((w) => w.name === "info");
					if (pos !== -1 && this.widgets[pos]) {
						const w = this.widgets[pos]
						w.value = text;
					}
				}
				requestAnimationFrame(() => {
					const sz = this.computeSize();
					if (sz[0] < this.size[0]) {
						sz[0] = this.size[0];
					}
					if (sz[1] < this.size[1]) {
						sz[1] = this.size[1];
					}
					this.onResize?.(sz);
					app.graph.setDirtyCanvas(true, false);
				});
			}

			// When the node is executed we will be sent the input text, display this in the widget
			const onExecuted = nodeType.prototype.onExecuted;
			nodeType.prototype.onExecuted = function (message) {
				onExecuted?.apply(this, arguments);
				populate.call(this, message.text);
			};
		}
    }
})