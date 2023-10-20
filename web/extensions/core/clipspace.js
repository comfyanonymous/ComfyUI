import { app } from "../../scripts/app.js";
import { ComfyDialog, $el } from "../../scripts/ui.js";
import { ComfyApp } from "../../scripts/app.js";

export class ClipspaceDialog extends ComfyDialog {
	static items = [];
	static instance = null;

	static registerButton(name, contextPredicate, callback) {
		const item =
			$el("button", {
				type: "button",
				textContent: name,
				contextPredicate: contextPredicate,
				onclick: callback
			})

		ClipspaceDialog.items.push(item);
	}

	static invalidatePreview() {
		if(ComfyApp.clipspace && ComfyApp.clipspace.imgs && ComfyApp.clipspace.imgs.length > 0) {
			const img_preview = document.getElementById("clipspace_preview");
			if(img_preview) {
				img_preview.src = ComfyApp.clipspace.imgs[ComfyApp.clipspace['selectedIndex']].src;
				img_preview.style.maxHeight = "100%";
				img_preview.style.maxWidth = "100%";
			}
		}
	}

	static invalidate() {
		if(ClipspaceDialog.instance) {
			const self = ClipspaceDialog.instance;
			// allow reconstruct controls when copying from non-image to image content.
			const children = $el("div.comfy-modal-content", [ self.createImgSettings(), ...self.createButtons() ]);

			if(self.element) {
				// update
				self.element.removeChild(self.element.firstChild);
				self.element.appendChild(children);
			}
			else {
				// new
				self.element = $el("div.comfy-modal", { parent: document.body }, [children,]);
			}

			if(self.element.children[0].children.length <= 1) {
				self.element.children[0].appendChild($el("p", {}, ["Unable to find the features to edit content of a format stored in the current Clipspace."]));
			}

			ClipspaceDialog.invalidatePreview();
		}
	}

	constructor() {
		super();
	}

	createButtons(self) {
		const buttons = [];

		for(let idx in ClipspaceDialog.items) {
			const item = ClipspaceDialog.items[idx];
			if(!item.contextPredicate || item.contextPredicate())
				buttons.push(ClipspaceDialog.items[idx]);
		}

		buttons.push(
			$el("button", {
				type: "button",
				textContent: "Close",
				onclick: () => { this.close(); }
			})
		);

		return buttons;
	}

	createImgSettings() {
		if(ComfyApp.clipspace.imgs) {
			const combo_items = [];
			const imgs = ComfyApp.clipspace.imgs;

			for(let i=0; i < imgs.length; i++) {
				combo_items.push($el("option", {value:i}, [`${i}`]));
			}

			const combo1 = $el("select",
				{id:"clipspace_img_selector", onchange:(event) => {
					ComfyApp.clipspace['selectedIndex'] = event.target.selectedIndex;
					ClipspaceDialog.invalidatePreview();
				} }, combo_items);

			const row1 =
				$el("tr", {},
						[
							$el("td", {}, [$el("font", {color:"white"}, ["Select Image"])]),
							$el("td", {}, [combo1])
						]);


			const combo2 = $el("select",
								{id:"clipspace_img_paste_mode", onchange:(event) => {
									ComfyApp.clipspace['img_paste_mode'] = event.target.value;
								} },
								[
									$el("option", {value:'selected'}, 'selected'),
									$el("option", {value:'all'}, 'all')
								]);
			combo2.value = ComfyApp.clipspace['img_paste_mode'];

			const row2 =
				$el("tr", {},
						[
							$el("td", {}, [$el("font", {color:"white"}, ["Paste Mode"])]),
							$el("td", {}, [combo2])
						]);

			const td = $el("td", {align:'center', width:'100px', height:'100px', colSpan:'2'},
								[ $el("img",{id:"clipspace_preview", ondragstart:() => false},[]) ]);

			const row3 =
				$el("tr", {}, [td]);

			return $el("table", {}, [row1, row2, row3]);
		}
		else {
			return [];
		}
	}

	createImgPreview() {
		if(ComfyApp.clipspace.imgs) {
			return $el("img",{id:"clipspace_preview", ondragstart:() => false});
		}
		else
			return [];
	}

	show() {
		const img_preview = document.getElementById("clipspace_preview");
		ClipspaceDialog.invalidate();
		
		this.element.style.display = "block";
	}
}

app.registerExtension({
	name: "Comfy.Clipspace",
	init(app) {
		app.openClipspace =
			function () {
				if(!ClipspaceDialog.instance) {
					ClipspaceDialog.instance = new ClipspaceDialog(app);
					ComfyApp.clipspace_invalidate_handler = ClipspaceDialog.invalidate;
				}

				if(ComfyApp.clipspace) {
					ClipspaceDialog.instance.show();
				}
				else
					app.ui.dialog.show("Clipspace is Empty!");
			};
	}
});