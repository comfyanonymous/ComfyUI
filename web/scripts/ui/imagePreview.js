import { $el } from "../ui.js";

export function calculateImageGrid(imgs, dw, dh) {
	let best = 0;
	let w = imgs[0].naturalWidth;
	let h = imgs[0].naturalHeight;
	const numImages = imgs.length;

	let cellWidth, cellHeight, cols, rows, shiftX;
	// compact style
	for (let c = 1; c <= numImages; c++) {
		const r = Math.ceil(numImages / c);
		const cW = dw / c;
		const cH = dh / r;
		const scaleX = cW / w;
		const scaleY = cH / h;

		const scale = Math.min(scaleX, scaleY, 1);
		const imageW = w * scale;
		const imageH = h * scale;
		const area = imageW * imageH * numImages;

		if (area > best) {
			best = area;
			cellWidth = imageW;
			cellHeight = imageH;
			cols = c;
			rows = r;
			shiftX = c * ((cW - imageW) / 2);
		}
	}

	return { cellWidth, cellHeight, cols, rows, shiftX };
}

export function createImageHost(node) {
	const el = $el("div.comfy-img-preview");
	let currentImgs;
	let first = true;

	function updateSize() {
		let w = null;
		let h = null;

		if (currentImgs) {
			let elH = el.clientHeight;
			if (first) {
				first = false;
				// On first run, if we are small then grow a bit
				if (elH < 190) {
					elH = 190;
				}
				el.style.setProperty("--comfy-widget-min-height", elH);
			} else {
				el.style.setProperty("--comfy-widget-min-height", null);
			}

			const nw = node.size[0];
			({ cellWidth: w, cellHeight: h } = calculateImageGrid(currentImgs, nw - 20, elH));
			w += "px";
			h += "px";

			el.style.setProperty("--comfy-img-preview-width", w);
			el.style.setProperty("--comfy-img-preview-height", h);
		}
	}
	return {
		el,
		updateImages(imgs) {
			if (imgs !== currentImgs) {
				if (currentImgs == null) {
					requestAnimationFrame(() => {
						updateSize();
					});
				}
				el.replaceChildren(...imgs);
				currentImgs = imgs;
				node.onResize(node.size);
				node.graph.setDirtyCanvas(true, true);
			}
		},
		getHeight() {
			updateSize();
		},
		onDraw() {
			// Element from point uses a hittest find elements so we need to toggle pointer events
			el.style.pointerEvents = "all";
			const over = document.elementFromPoint(app.canvas.mouse[0], app.canvas.mouse[1]);
			el.style.pointerEvents = "none";

			if(!over) return;
			// Set the overIndex so Open Image etc work
			const idx = currentImgs.indexOf(over);
			node.overIndex = idx;
		},
	};
}
