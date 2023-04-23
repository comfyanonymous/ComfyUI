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

	createButton(name,callback) {
		var button = document.createElement("button");
		button.innerText = name;
		button.style.position = "absolute";
		button.style.top = "5px";
		button.addEventListener("click", callback);
		return button;
	}

	createLeftButton(name,left,callback) {
		var button = this.createButton(name,callback);
		button.style.left = left;
		return button;
	}

	createRightButton(name,right,callback) {
		var button = this.createButton(name,callback);
		button.style.right = right;
		return button;
	}

	setlayout(imgCanvas, maskCanvas) {
		const self = this;
		var bottom_panel = document.createElement("div");
		bottom_panel.style.position = "relative";
		bottom_panel.style.bottom = "0";
		bottom_panel.style.left = "0";
		bottom_panel.style.right = "0";
		bottom_panel.style.height = "50px";

		var clearButton = this.createLeftButton("Clear", "0px", 
			() => {
				self.maskCtx.clearRect(0, 0, self.maskCanvas.width, self.maskCanvas.height);
				self.backupCtx.clearRect(0, 0, self.backupCanvas.width, self.backupCanvas.height);
			});

		var plusButton = this.createLeftButton("Brush +", "70px", () => { self.brush_size = Math.min(self.brush_size+5, 100); });
		var minusButton = this.createLeftButton("Brush -", "160px", () => { self.brush_size = Math.max(self.brush_size-5, 1); });

		var saveButton = this.createRightButton("Save", "90px", () => {
				document.addEventListener('mouseup', this.handleMouseUp);
				self.save(); 
			});
		var cancelButton = this.createRightButton("Cancel", "0px", () => {			
				document.addEventListener('mouseup', this.handleMouseUp);
				self.close(); 
			});
		
		this.element.appendChild(imgCanvas);
		this.element.appendChild(maskCanvas);
		this.element.appendChild(bottom_panel);

		bottom_panel.appendChild(clearButton);
		bottom_panel.appendChild(plusButton);
		bottom_panel.appendChild(minusButton);
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
			imgCanvas.height = window.innerHeight - 200;

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
		maskCanvas.addEventListener("contextmenu", (event) => {
			event.preventDefault();
		});

		const self = this;
		maskCanvas.addEventListener('wheel', (event) => this.handleWheelEvent(self,event));
		maskCanvas.addEventListener('mousedown', (event) => this.handleMouseDown(self,event));
		document.addEventListener('mouseup', (event) => this.handleMouseUp(self,event));
		maskCanvas.addEventListener('mousemove', (event) => this.draw_move(self,event));
		maskCanvas.addEventListener('touchmove', (event) => this.draw_move(self,event));

	}

	brush_size = 10;
	drawing_mode = false;
	lastx = -1;
	lasty = -1;
	lasttime = 0;

	handleWheelEvent(self, event) {
		if(event.deltaY < 0)
			self.brush_size = Math.min(self.brush_size+2, 100);
		else
			self.brush_size = Math.max(self.brush_size-2, 1);
	}

	draw_move(self, event) {
		if (event instanceof TouchEvent || event.buttons === 1) {
			event.preventDefault();

			var diff = performance.now() - self.lasttime;

			const maskRect = self.maskCanvas.getBoundingClientRect();
			const x = event.offsetX || event.targetTouches[0].clientX - maskRect.left;
			const y = event.offsetY || event.targetTouches[0].clientY - maskRect.top;

			if(diff > 20)
				requestAnimationFrame(() => {
					self.maskCtx.beginPath();
					self.maskCtx.fillStyle = "rgb(0,0,0)";
					self.maskCtx.globalCompositeOperation = "source-over";
					self.maskCtx.arc(x, y, this.brush_size, 0, Math.PI * 2, false);
					self.maskCtx.fill();
					self.lastx = x;
					self.lasty = y;
				});
			else
				requestAnimationFrame(() => {					
					self.maskCtx.beginPath();
					self.maskCtx.fillStyle = "rgb(0,0,0)";
					self.maskCtx.globalCompositeOperation = "source-over";
					
					var dx = x - self.lastx;
					var dy = y - self.lasty;

					var distance = Math.sqrt(dx * dx + dy * dy);
					var directionX = dx / distance;
					var directionY = dy / distance;

					for (var i = 0; i < distance; i++) {
						var px = self.lastx + (directionX * i);
						var py = self.lasty + (directionY * i);
						self.maskCtx.arc(px, py, this.brush_size, 0, Math.PI * 2, false);
						self.maskCtx.fill();
					}
					self.lastx = x;
					self.lasty = y;
				});

			self.lasttime = performance.now();
		}
		else if(event.buttons === 2) {
			event.preventDefault();
			const maskRect = maskCanvas.getBoundingClientRect();
			const x = event.offsetX || event.targetTouches[0].clientX - maskRect.left;
			const y = event.offsetY || event.targetTouches[0].clientY - maskRect.top;

			if(diff > 16 && !drawing_mode) // cannot tracking drawing_mode for touch event
				requestAnimationFrame(() => {
					self.maskCtx.beginPath();
					self.maskCtx.globalCompositeOperation = "destination-out";
					self.maskCtx.arc(x, y, this.brush_size, 0, Math.PI * 2, false);
					self.maskCtx.fill();
					self.lastx = x;
					self.lasty = y;
				});
			else
				requestAnimationFrame(() => {					
					self.maskCtx.beginPath();
					self.maskCtx.globalCompositeOperation = "destination-out";
					
					var dx = x - self.lastx;
					var dy = y - self.lasty;

					var distance = Math.sqrt(dx * dx + dy * dy);
					var directionX = dx / distance;
					var directionY = dy / distance;

					for (var i = 0; i < distance; i++) {
						var px = self.lastx + (directionX * i);
						var py = self.lasty + (directionY * i);
						self.maskCtx.arc(px, py, this.brush_size, 0, Math.PI * 2, false);
						self.maskCtx.fill();
					}
					self.lastx = x;
					self.lasty = y;
				});

				self.lasttime = performance.now();
		}
	}

	handleMouseDown(self, event) {
		self.drawing_mode = true;
		if (event.button == 0) {
			const maskRect = maskCanvas.getBoundingClientRect();
			const x = event.offsetX || event.targetTouches[0].clientX - maskRect.left;
			const y = event.offsetY || event.targetTouches[0].clientY - maskRect.top;

			self.maskCtx.beginPath();
			self.maskCtx.fillStyle = "rgb(0,0,0)";
			self.maskCtx.globalCompositeOperation = "source-over";
			self.maskCtx.arc(x, y, this.brush_size, 0, Math.PI * 2, false);
			self.maskCtx.fill();
			self.lastx = x;
			self.lasty = y;
			self.lasttime = performance.now();
		}
		else if(event.button == 2) {
			const maskRect = maskCanvas.getBoundingClientRect();
			const x = event.offsetX || event.targetTouches[0].clientX - maskRect.left;
			const y = event.offsetY || event.targetTouches[0].clientY - maskRect.top;

			self.maskCtx.beginPath();
			self.maskCtx.globalCompositeOperation = "destination-out";						
			self.maskCtx.arc(x, y, this.brush_size, 0, Math.PI * 2, false);
			self.maskCtx.fill();
			self.lastx = x;
			self.lasty = y;
			self.lasttime = performance.now();
		}
	}

	handleMouseUp(self, event) {
		self.drawing_mode = false;
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