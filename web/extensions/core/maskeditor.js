import { app } from "/scripts/app.js";
import { ComfyDialog, $el } from "/scripts/ui.js";
import { ComfyApp } from "/scripts/app.js";
import { ClipspaceDialog } from "/extensions/core/clipspace.js";

// Helper function to convert a data URL to a Blob object
function dataURLToBlob(dataURL) {
	const parts = dataURL.split(';base64,');
	const contentType = parts[0].split(':')[1];
	const byteString = atob(parts[1]);
	const arrayBuffer = new ArrayBuffer(byteString.length);
	const uint8Array = new Uint8Array(arrayBuffer);
	for (let i = 0; i < byteString.length; i++) {
		uint8Array[i] = byteString.charCodeAt(i);
	}
	return new Blob([arrayBuffer], { type: contentType });
}

function loadedImageToBlob(image) {
	const canvas = document.createElement('canvas');

	canvas.width = image.width;
	canvas.height = image.height;

	const ctx = canvas.getContext('2d');

	ctx.drawImage(image, 0, 0);

	const dataURL = canvas.toDataURL('image/png', 1);
	const blob = dataURLToBlob(dataURL);

	return blob;
}

async function uploadMask(filepath, formData) {
	await fetch('/upload/mask', {
		method: 'POST',
		body: formData
	}).then(response => {}).catch(error => {
		console.error('Error:', error);
	});

	ComfyApp.clipspace.imgs[0] = new Image();
	ComfyApp.clipspace.imgs[0].src = `view?filename=${filepath.filename}&type=${filepath.type}`;
	ComfyApp.clipspace.images = [filepath];
}

function removeRGB(image, backupCanvas, backupCtx, maskCtx) {
	// paste mask data into alpha channel
	backupCtx.drawImage(image, 0, 0, backupCanvas.width, backupCanvas.height);
	const backupData = backupCtx.getImageData(0, 0, backupCanvas.width, backupCanvas.height);

	// refine mask image
	for (let i = 0; i < backupData.data.length; i += 4) {
		if(backupData.data[i+3] == 255)
			backupData.data[i+3] = 0;
		else
			backupData.data[i+3] = 255;

		backupData.data[i] = 0;
		backupData.data[i+1] = 0;
		backupData.data[i+2] = 0;
	}

	backupCtx.globalCompositeOperation = 'source-over';
	backupCtx.putImageData(backupData, 0, 0);
}

class MaskEditorDialog extends ComfyDialog {
	constructor() {
		super();
		this.element = $el("div.comfy-modal", { parent: document.body }, 
	[
			$el("div.comfy-modal-content", 
		[
			...this.createButtons()]),
		]);
	}

	createButtons() {
		return [];
//			$el("button", {
//				type: "button",
//				textContent: "Save",
//				onclick: () => this.save(),
//			}),
//			$el("button", {
//				type: "button",
//				textContent: "Cancel",
//				onclick: () => this.close(),
//			}),
//			$el("button", {
//				type: "button",
//				textContent: "Clear",
//				onclick: () => {
//					this.maskCtx.clearRect(0, 0, this.maskCanvas.width, this.maskCanvas.height);
//					this.backupCtx.clearRect(0, 0, this.backupCanvas.width, this.backupCanvas.height);
//				},
//			}),
//		];
	}

	clearMask(self) {
	}

	setlayout(imgCanvas, maskCanvas) {
		const self = this;
		var bottom_panel = document.createElement("div");
		bottom_panel.style.position = "fixed";
		bottom_panel.style.bottom = "0";
		bottom_panel.style.left = "0";
		bottom_panel.style.right = "0";
		bottom_panel.style.height = "50px";

		var clearButton = document.createElement("button");
		clearButton.innerText = "Clear";
		clearButton.style.position = "absolute";
		clearButton.style.left = "20px";
		clearButton.addEventListener("click", function() {
			self.maskCtx.clearRect(0, 0, self.maskCanvas.width, self.maskCanvas.height);
			self.backupCtx.clearRect(0, 0, self.backupCanvas.width, self.backupCanvas.height);
		});

		var saveButton = document.createElement("button");
		saveButton.innerText = "Save";
		saveButton.style.position = "absolute";
		saveButton.style.right = "110px";
		saveButton.addEventListener("click", function() { self.save(); });

		var cancelButton = document.createElement("button");
		cancelButton.innerText = "Cancel";
		cancelButton.style.position = "absolute";
		cancelButton.style.right = "20px";
		cancelButton.addEventListener("click", function() { self.close(); });

		this.element.appendChild(imgCanvas);
		this.element.appendChild(maskCanvas);
		this.element.appendChild(bottom_panel);

		bottom_panel.appendChild(clearButton);
		bottom_panel.appendChild(saveButton);
		bottom_panel.appendChild(cancelButton);

		this.element.style.display = "block";
		imgCanvas.style.position = "relative";
		imgCanvas.style.top = "200";
		imgCanvas.style.left = "0";

		maskCanvas.style.position = "absolute";
	}

	show() {
		// layout
		const imgCanvas = document.createElement('canvas');
		const maskCanvas = document.createElement('canvas');
		const backupCanvas = document.createElement('canvas');

		imgCanvas.id = "imageCanvas";
		maskCanvas.id = "maskCanvas";
		backupCanvas.id = "backupCanvas";

		this.setlayout(imgCanvas, maskCanvas);

		// prepare content

		this.maskCanvas = maskCanvas;
		this.backupCanvas = backupCanvas;
		this.maskCtx = maskCanvas.getContext('2d');
		this.backupCtx = backupCanvas.getContext('2d');

		// separate original_imgs and imgs
		if(ComfyApp.clipspace.imgs[0] === ComfyApp.clipspace.original_imgs[0]) {
			console.log(ComfyApp.clipspace.imgs[0]);
			var copiedImage = new Image();
			copiedImage.src = ComfyApp.clipspace.original_imgs[0].src;
			ComfyApp.clipspace.imgs = [copiedImage];
		}

		this.setImages(imgCanvas, backupCanvas);
		this.setEventHandler(maskCanvas);
	}

	setImages(imgCanvas, backupCanvas) {
		const imgCtx = imgCanvas.getContext('2d');
		const backupCtx = backupCanvas.getContext('2d');
		const maskCtx = this.maskCtx;
		const maskCanvas = this.maskCanvas;

		// image load
		const orig_image = new Image();
		window.addEventListener("resize", () => {
			// repositioning
			imgCanvas.width = window.innerWidth - 250;
			imgCanvas.height = window.innerHeight - 300;

			// redraw image
			let drawWidth = orig_image.width;
			let drawHeight = orig_image.height;
			if (orig_image.width > imgCanvas.width) {
				drawWidth = imgCanvas.width;
				drawHeight = (drawWidth / orig_image.width) * orig_image.height;
			}

			if (drawHeight > imgCanvas.height) {
				drawHeight = imgCanvas.height;
				drawWidth = (drawHeight / orig_image.height) * orig_image.width;
			}

			imgCtx.drawImage(orig_image, 0, 0, drawWidth, drawHeight);

			// update mask
			backupCtx.drawImage(maskCanvas, 0, 0, maskCanvas.width, maskCanvas.height, 0, 0, backupCanvas.width, backupCanvas.height);
			maskCanvas.width = drawWidth;
			maskCanvas.height = drawHeight;
			maskCanvas.style.top = imgCanvas.offsetTop + "px";
			maskCanvas.style.left = imgCanvas.offsetLeft + "px";
			maskCtx.drawImage(backupCanvas, 0, 0, backupCanvas.width, backupCanvas.height, 0, 0, maskCanvas.width, maskCanvas.height);
		});

		const filepath = ComfyApp.clipspace.images;

		const touched_image = new Image();

		touched_image.onload = function() {
			backupCanvas.width = touched_image.width;
			backupCanvas.height = touched_image.height;

			removeRGB(touched_image, backupCanvas, backupCtx, maskCtx);
		};

		touched_image.src = ComfyApp.clipspace.imgs[0].src;

		// original image load
		orig_image.onload = function() {
			window.dispatchEvent(new Event('resize'));
		};

		orig_image.src = ComfyApp.clipspace.original_imgs[0].src;
		this.image = orig_image;
	}

	setEventHandler(maskCanvas) {
		let brush_size = 10;
		const maskCtx = maskCanvas.getContext('2d');

		function draw_point(event) {
			console.log(event.button);
			if (event.button == 0) {
				const maskRect = maskCanvas.getBoundingClientRect();
				const x = event.offsetX || event.targetTouches[0].clientX - maskRect.left;
				const y = event.offsetY || event.targetTouches[0].clientY - maskRect.top;

				maskCtx.beginPath();
				maskCtx.fillStyle = "rgb(0,0,0)";
				maskCtx.globalCompositeOperation = "source-over";
				maskCtx.arc(x, y, brush_size, 0, Math.PI * 2, false);
				maskCtx.fill();
			}
		}

		function draw_move(event) {
			if (event.buttons === 1) {
				event.preventDefault();
				const maskRect = maskCanvas.getBoundingClientRect();
				const x = event.offsetX || event.targetTouches[0].clientX - maskRect.left;
				const y = event.offsetY || event.targetTouches[0].clientY - maskRect.top;

				maskCtx.beginPath();
				maskCtx.fillStyle = "rgb(0,0,0)";
				maskCtx.globalCompositeOperation = "source-over";
				maskCtx.arc(x, y, brush_size, 0, Math.PI * 2, false);
				maskCtx.fill();
			}
			else if(event.buttons === 2) {
				event.preventDefault();
				const maskRect = maskCanvas.getBoundingClientRect();
				const x = event.offsetX || event.targetTouches[0].clientX - maskRect.left;
				const y = event.offsetY || event.targetTouches[0].clientY - maskRect.top;

				maskCtx.beginPath();
				maskCtx.globalCompositeOperation = "destination-out";
				maskCtx.arc(x, y, brush_size, 0, Math.PI * 2, false);
				maskCtx.fill();
			}
		}

		function handleWheelEvent(event) {
			if(event.deltaY < 0)
				brush_size = Math.min(brush_size+2, 100);
			else
				brush_size = Math.max(brush_size-2, 1);
		}

		maskCanvas.addEventListener("contextmenu", (event) => {
			event.preventDefault();
		});

		maskCanvas.addEventListener('wheel', handleWheelEvent);
		maskCanvas.addEventListener('mousedown', draw_point);
		maskCanvas.addEventListener('mousemove', draw_move);
		maskCanvas.addEventListener('touchmove', draw_move);

	}

	save() {
		const backupCtx = this.backupCanvas.getContext('2d', {willReadFrequently:true});

		backupCtx.clearRect(0,0,this.backupCanvas.width,this.backupCanvas.height);
		backupCtx.drawImage(this.maskCanvas,
			0, 0, this.maskCanvas.width, this.maskCanvas.height,
			0, 0, this.backupCanvas.width, this.backupCanvas.height);

		// paste mask data into alpha channel
		const backupData = backupCtx.getImageData(0, 0, this.backupCanvas.width, this.backupCanvas.height);

		// refine mask image
		for (let i = 0; i < backupData.data.length; i += 4) {
			if(backupData.data[i+3] == 255)
				backupData.data[i+3] = 0;
			else
				backupData.data[i+3] = 255;

			backupData.data[i] = 0;
			backupData.data[i+1] = 0;
			backupData.data[i+2] = 0;
		}

		backupCtx.globalCompositeOperation = 'source-over';
		backupCtx.putImageData(backupData, 0, 0);

		const formData = new FormData();
		const filename = "clipspace-mask-" + performance.now() + ".png";

		const item =
			{
				"filename": filename,
				"subfolder": "",
				"type": "temp",
			};

		if(ComfyApp.clipspace.images)
			ComfyApp.clipspace.images[0] = item;

		if(ComfyApp.clipspace.widgets) {
			const index = ComfyApp.clipspace.widgets.findIndex(obj => obj.name === 'image');

			if(index >= 0)
				ComfyApp.clipspace.widgets[index].value = item;
		}

		const dataURL = this.backupCanvas.toDataURL();
		const blob = dataURLToBlob(dataURL);

		const original_blob = loadedImageToBlob(ComfyApp.clipspace.original_imgs[0]);

		formData.append('image', blob, filename);
		formData.append('original_image', original_blob);
		formData.append('type', "temp");

		uploadMask(item, formData);
		this.close();
	}
}

app.registerExtension({
	name: "Comfy.MaskEditor",
	init(app) {
		const callback =
			function () {
				let dlg = new MaskEditorDialog(app);
				dlg.show();
			};

		ClipspaceDialog.registerButton("MaskEditor", callback);
	}
});