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

	static invalidatePreview() {
		const img_preview = document.getElementById("clipspace_preview");
		img_preview.src = ComfyApp.clipspace.imgs[ComfyApp.clipspace['selectedIndex']].src;
		img_preview.style.height = "100px";
	}

	constructor() {
		super();
		this.element =
			$el("div.comfy-modal", { parent: document.body },
				[$el("div.comfy-modal-content",[
					this.createImgSelector(),
					this.createImgPreview(),
					...this.createButtons()]),]
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

	createImgSelector() {
		if(ComfyApp.clipspace.imgs != undefined) {
			const combo_items = [];
			const imgs = ComfyApp.clipspace.imgs;

			for(let i=0; i < imgs.length; i++) {
				combo_items.push($el("option", {value:i}, [`${i}`]));
			}

			const combo = $el("select",
				{id:"clipspace_img_selector", onchange:(event) => {
					ComfyApp.clipspace['selectedIndex'] = event.target.selectedIndex;
					ClipspaceDialog.invalidatePreview();
				} }, combo_items);
			return combo;
		}
		else {
			return [];
		}
	}

	createImgPreview() {
		if(ComfyApp.clipspace.imgs != undefined) {
			return $el("img",{id:"clipspace_preview"});
		}
		else
			return [];
	}

	show() {
		ClipspaceDialog.is_opened = true;
		const img_preview = document.getElementById("clipspace_preview");
		img_preview.src = ComfyApp.clipspace.imgs[0].src;
		img_preview.style.height = "100px";
		
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