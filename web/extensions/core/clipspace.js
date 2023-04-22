import { app } from "/scripts/app.js";
import { ComfyDialog, $el } from "/scripts/ui.js";
import { ComfyApp } from "/scripts/app.js";

export class ClipspaceDialog extends ComfyDialog {
	static items = [];
	static is_opened = false; // prevent redundant popup

	static registerButton(name, callback) {
		const item =
			$el("button", {
				type: "button",
				textContent: name,
				onclick: callback
			})

		ClipspaceDialog.items.push(item);
	}

	constructor() {
		super();
		this.element =
			$el("div.comfy-modal", { parent: document.body },
				[$el("div.comfy-modal-content",[...this.createButtons()]),]
				);
	}

	createButtons() {
		const buttons = [];

		for(let idx in ClipspaceDialog.items) {
			const item = ClipspaceDialog.items[idx];
			buttons.push(ClipspaceDialog.items[idx]);
		}

		buttons.push(
			$el("button", {
				type: "button",
				textContent: "Close",
				onclick: () => {
					ClipspaceDialog.is_opened = false;
					this.close();
				}
			})
		);

		return buttons;
	}

	show() {
		ClipspaceDialog.is_opened = true;
		this.element.style.display = "block";
	}
}

app.registerExtension({
	name: "Comfy.Clipspace",
	init(app) {
		app.openClipspace =
			function () {
				if(!ClipspaceDialog.is_opened) {
					let dlg = new ClipspaceDialog(app);
					if(ComfyApp.clipspace)
						dlg.show();
					else
						app.ui.dialog.show("Clipspace is Empty!");
				}
			};
	}
});