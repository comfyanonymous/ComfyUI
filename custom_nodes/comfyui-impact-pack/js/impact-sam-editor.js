import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { ComfyDialog, $el } from "../../scripts/ui.js";
import { ComfyApp } from "../../scripts/app.js";
import { ClipspaceDialog } from "../../extensions/core/clipspace.js";

function addMenuHandler(nodeType, cb) {
	const getOpts = nodeType.prototype.getExtraMenuOptions;
	nodeType.prototype.getExtraMenuOptions = function () {
		const r = getOpts.apply(this, arguments);
		cb.apply(this, arguments);
		return r;
	};
}

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
	await api.fetchApi('/upload/mask', {
		method: 'POST',
		body: formData
	}).then(response => {}).catch(error => {
		console.error('Error:', error);
	});

	ComfyApp.clipspace.imgs[ComfyApp.clipspace['selectedIndex']] = new Image();
	ComfyApp.clipspace.imgs[ComfyApp.clipspace['selectedIndex']].src = `view?filename=${filepath.filename}&type=${filepath.type}`;

	if(ComfyApp.clipspace.images)
		ComfyApp.clipspace.images[ComfyApp.clipspace['selectedIndex']] = filepath;

	ClipspaceDialog.invalidatePreview();
}

class ImpactSamEditorDialog extends ComfyDialog {
	static instance = null;

	static getInstance() {
		if(!ImpactSamEditorDialog.instance) {
			ImpactSamEditorDialog.instance = new ImpactSamEditorDialog();
		}

		return ImpactSamEditorDialog.instance;
	}

	constructor() {
		super();
		this.element = $el("div.comfy-modal", { parent: document.body }, 
			[ $el("div.comfy-modal-content", 
				[...this.createButtons()]),
			]);
	}

	createButtons() {
		return [];
	}

	createButton(name, callback) {
		var button = document.createElement("button");
		button.innerText = name;
		button.addEventListener("click", callback);
		return button;
	}

	createLeftButton(name, callback) {
		var button = this.createButton(name, callback);
		button.style.cssFloat = "left";
		button.style.marginRight = "4px";
		return button;
	}

	createRightButton(name, callback) {
		var button = this.createButton(name, callback);
		button.style.cssFloat = "right";
		button.style.marginLeft = "4px";
		return button;
	}

	createLeftSlider(self, name, callback) {
		const divElement = document.createElement('div');
		divElement.id = "sam-confidence-slider";
		divElement.style.cssFloat = "left";
		divElement.style.fontFamily = "sans-serif";
		divElement.style.marginRight = "4px";
		divElement.style.color = "var(--input-text)";
		divElement.style.backgroundColor = "var(--comfy-input-bg)";
		divElement.style.borderRadius = "8px";
		divElement.style.borderColor = "var(--border-color)";
		divElement.style.borderStyle = "solid";
		divElement.style.fontSize = "15px";
		divElement.style.height = "21px";
		divElement.style.padding = "1px 6px";
		divElement.style.display = "flex";
		divElement.style.position = "relative";
		divElement.style.top = "2px";
		self.confidence_slider_input = document.createElement('input');
		self.confidence_slider_input.setAttribute('type', 'range');
		self.confidence_slider_input.setAttribute('min', '0');
		self.confidence_slider_input.setAttribute('max', '100');
		self.confidence_slider_input.setAttribute('value', '70');
		const labelElement = document.createElement("label");
		labelElement.textContent = name;

		divElement.appendChild(labelElement);
		divElement.appendChild(self.confidence_slider_input);

		self.confidence_slider_input.addEventListener("change", callback);

		return divElement;
	}

	async detect_and_invalidate_mask_canvas(self) {
		const mask_img = await self.detect(self);

		const canvas = self.maskCtx.canvas;
		const ctx = self.maskCtx;

		ctx.clearRect(0, 0, canvas.width, canvas.height);

		await new Promise((resolve, reject) => {
						self.mask_image = new Image();
						self.mask_image.onload = function() {
							ctx.drawImage(self.mask_image, 0, 0, canvas.width, canvas.height);
							resolve();
						};
						self.mask_image.onerror = reject;
						self.mask_image.src = mask_img.src;
				});
	}

	setlayout(imgCanvas, maskCanvas, pointsCanvas) {
		const self = this;

		// If it is specified as relative, using it only as a hidden placeholder for padding is recommended
		// to prevent anomalies where it exceeds a certain size and goes outside of the window.
		var placeholder = document.createElement("div");
		placeholder.style.position = "relative";
		placeholder.style.height = "50px";

		var bottom_panel = document.createElement("div");
		bottom_panel.style.position = "absolute";
		bottom_panel.style.bottom = "0px";
		bottom_panel.style.left = "20px";
		bottom_panel.style.right = "20px";
		bottom_panel.style.height = "50px";

		var brush = document.createElement("div");
		brush.id = "sam-brush";
		brush.style.backgroundColor = "blue";
		brush.style.outline = "2px solid pink";
		brush.style.borderRadius = "50%";
		brush.style.MozBorderRadius = "50%";
		brush.style.WebkitBorderRadius = "50%";
		brush.style.position = "absolute";
		brush.style.zIndex = 100;
		brush.style.pointerEvents = "none";
		this.brush = brush;
		this.element.appendChild(imgCanvas);
		this.element.appendChild(maskCanvas);
		this.element.appendChild(pointsCanvas);
		this.element.appendChild(placeholder); // must below z-index than bottom_panel to avoid covering button
		this.element.appendChild(bottom_panel);
		document.body.appendChild(brush);
		this.brush_size = 5;

		var confidence_slider = this.createLeftSlider(self, "Confidence", (event) => {
			self.confidence = event.target.value;
		});

		var clearButton = this.createLeftButton("Clear", () => {
				self.maskCtx.clearRect(0, 0, self.maskCanvas.width, self.maskCanvas.height);
				self.pointsCtx.clearRect(0, 0, self.pointsCanvas.width, self.pointsCanvas.height);

				self.prompt_points = [];

				self.invalidatePointsCanvas(self);
			});

		var detectButton = this.createLeftButton("Detect", () => self.detect_and_invalidate_mask_canvas(self));
		
		var cancelButton = this.createRightButton("Cancel", () => {
				document.removeEventListener("mouseup", ImpactSamEditorDialog.handleMouseUp);
				document.removeEventListener("keydown", ImpactSamEditorDialog.handleKeyDown);
				self.close();
			});

		self.saveButton = this.createRightButton("Save", () => {
				document.removeEventListener("mouseup", ImpactSamEditorDialog.handleMouseUp);
				document.removeEventListener("keydown", ImpactSamEditorDialog.handleKeyDown);
				self.save(self);
			});

		var undoButton = this.createLeftButton("Undo", () => {
				if(self.prompt_points.length > 0) {
					self.prompt_points.pop();
					self.pointsCtx.clearRect(0, 0, self.pointsCanvas.width, self.pointsCanvas.height);
					self.invalidatePointsCanvas(self);
				}
			});

		bottom_panel.appendChild(clearButton);
		bottom_panel.appendChild(detectButton);
		bottom_panel.appendChild(self.saveButton);
		bottom_panel.appendChild(cancelButton);
		bottom_panel.appendChild(confidence_slider);
		bottom_panel.appendChild(undoButton);

		imgCanvas.style.position = "relative";
		imgCanvas.style.top = "200";
		imgCanvas.style.left = "0";

		maskCanvas.style.position = "absolute";
		maskCanvas.style.opacity = 0.5;
		pointsCanvas.style.position = "absolute";
	}

	show() {
		this.mask_image = null;
		self.prompt_points = [];

		this.message_box = $el("p", ["Please wait a moment while the SAM model and the image are being loaded."]);
		this.element.appendChild(this.message_box);

		if(self.imgCtx) {
			self.imgCtx.clearRect(0, 0, self.imageCanvas.width, self.imageCanvas.height);
		}

		const target_image_path = ComfyApp.clipspace.imgs[ComfyApp.clipspace['selectedIndex']].src;
		this.load_sam(target_image_path);

		if(!this.is_layout_created) {
			// layout
			const imgCanvas = document.createElement('canvas');
			const maskCanvas = document.createElement('canvas');
			const pointsCanvas = document.createElement('canvas');

			imgCanvas.id = "imageCanvas";
			maskCanvas.id = "samEditorMaskCanvas";
			pointsCanvas.id = "pointsCanvas";

			this.setlayout(imgCanvas, maskCanvas, pointsCanvas);

			// prepare content
			this.imgCanvas = imgCanvas;
			this.maskCanvas = maskCanvas;
			this.pointsCanvas = pointsCanvas;
			this.maskCtx = maskCanvas.getContext('2d');
			this.pointsCtx = pointsCanvas.getContext('2d');

			this.is_layout_created = true;

			// replacement of onClose hook since close is not real close
			const self = this;
			const observer = new MutationObserver(function(mutations) {
			mutations.forEach(function(mutation) {
					if (mutation.type === 'attributes' && mutation.attributeName === 'style') {
						if(self.last_display_style && self.last_display_style != 'none' && self.element.style.display == 'none') {
							ComfyApp.onClipspaceEditorClosed();
						}

						self.last_display_style = self.element.style.display;
					}
				});
			});

			const config = { attributes: true };
			observer.observe(this.element, config);
		}

		this.setImages(target_image_path, this.imgCanvas, this.pointsCanvas);

		if(ComfyApp.clipspace_return_node) {
			this.saveButton.innerText = "Save to node";
		}
		else {
			this.saveButton.innerText = "Save";
		}
		this.saveButton.disabled = true;

		this.element.style.display = "block";
		this.element.style.zIndex = 8888; // NOTE: alert dialog must be high priority.
	}

	updateBrushPreview(self, event) {
		event.preventDefault();

		const centerX = event.pageX;
		const centerY = event.pageY;

		const brush = self.brush;

		brush.style.width = self.brush_size * 2 + "px";
		brush.style.height = self.brush_size * 2 + "px";
		brush.style.left = (centerX - self.brush_size) + "px";
		brush.style.top = (centerY - self.brush_size) + "px";
	}

	setImages(target_image_path, imgCanvas, pointsCanvas) {
		const imgCtx = imgCanvas.getContext('2d');
		const maskCtx = this.maskCtx;
		const maskCanvas = this.maskCanvas;

		const self = this;

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
			let w = (drawWidth * imgCanvas.clientWidth/imgCanvas.width) + "px";
			let h = (drawHeight * imgCanvas.clientHeight/imgCanvas.height) + "px";

			pointsCanvas.width = drawWidth * imgCanvas.clientWidth/imgCanvas.width;
			pointsCanvas.height = drawHeight * imgCanvas.clientHeight/imgCanvas.height;
			pointsCanvas.style.top = imgCanvas.offsetTop + "px";
			pointsCanvas.style.left = imgCanvas.offsetLeft + "px";

			maskCanvas.width = pointsCanvas.width;
			maskCanvas.height = pointsCanvas.height;
			maskCanvas.style.top = imgCanvas.offsetTop + "px";
			maskCanvas.style.left = imgCanvas.offsetLeft + "px";

			self.invalidateMaskCanvas(self);
			self.invalidatePointsCanvas(self);
		});

		// original image load
		orig_image.onload = () => self.onLoaded(self);
		const rgb_url = new URL(target_image_path);
		rgb_url.searchParams.delete('channel');
		rgb_url.searchParams.set('channel', 'rgb');
		orig_image.src = rgb_url;
		self.image = orig_image;
	}

	onLoaded(self) {
		if(self.message_box) {
			self.element.removeChild(self.message_box);
			self.message_box = null;
		}

		window.dispatchEvent(new Event('resize'));

		self.setEventHandler(pointsCanvas);
		self.saveButton.disabled = false;
	}

	setEventHandler(targetCanvas) {
		targetCanvas.addEventListener("contextmenu", (event) => {
			event.preventDefault();
		});

		const self = this;
		targetCanvas.addEventListener('pointermove', (event) => this.updateBrushPreview(self,event));
		targetCanvas.addEventListener('pointerdown', (event) => this.handlePointerDown(self,event));
		targetCanvas.addEventListener('pointerover', (event) => { this.brush.style.display = "block"; });
		targetCanvas.addEventListener('pointerleave', (event) => { this.brush.style.display = "none"; });
		document.addEventListener('keydown', ImpactSamEditorDialog.handleKeyDown);
	}

	static handleKeyDown(event) {
		const self = ImpactSamEditorDialog.instance;
		if (event.key === '=') { // positive
			brush.style.backgroundColor = "blue";
			brush.style.outline = "2px solid pink";
			self.is_positive_mode = true;
		} else if (event.key === '-') { // negative
			brush.style.backgroundColor = "red";
			brush.style.outline = "2px solid skyblue";
			self.is_positive_mode = false;
		}
	}

	is_positive_mode = true;
	prompt_points = [];
	confidence = 70;

	invalidatePointsCanvas(self) {
		const ctx = self.pointsCtx;

		for (const i in self.prompt_points) {
			const [is_positive, x, y] = self.prompt_points[i];

			const scaledX = x * ctx.canvas.width / self.image.width;
			const scaledY = y * ctx.canvas.height / self.image.height;

			if(is_positive)
				ctx.fillStyle = "blue";
			else
				ctx.fillStyle = "red";
			ctx.beginPath();
			ctx.arc(scaledX, scaledY, 3, 0, 3 * Math.PI);
			ctx.fill();
		}
	}

	invalidateMaskCanvas(self) {
		if(self.mask_image) {
			self.maskCtx.clearRect(0, 0, self.maskCanvas.width, self.maskCanvas.height);
			self.maskCtx.drawImage(self.mask_image, 0, 0, self.maskCanvas.width, self.maskCanvas.height);
		}
	}

	async load_sam(url) {
		const parsedUrl = new URL(url);
		const searchParams = new URLSearchParams(parsedUrl.search);

		const filename = searchParams.get("filename") || "";
		const fileType = searchParams.get("type") || "";
		const subfolder = searchParams.get("subfolder") || "";

		const data = {
				sam_model_name: "auto",
				filename: filename,
				type: fileType,
				subfolder: subfolder
			};

		api.fetchApi('/sam/prepare', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(data)
		});
	}

	async detect(self) {
		const positive_points = [];
		const negative_points = [];

		for(const i in self.prompt_points) {
			const [is_positive, x, y] = self.prompt_points[i];
			const point = [x,y];
			if(is_positive) {
				positive_points.push(point);
			}
			else
				negative_points.push(point);
		}

		const data = {
			positive_points: positive_points,
			negative_points: negative_points,
			threshold: self.confidence/100
		};
		
		const response = await api.fetchApi('/sam/detect', {
			method: 'POST',
			headers: { 'Content-Type': 'image/png' },
			body: JSON.stringify(data)
		});
		
		const blob = await response.blob();
		const url = URL.createObjectURL(blob);

		return new Promise((resolve, reject) => {
			const image = new Image();
			image.onload = () => resolve(image);
			image.onerror = reject;
			image.src = url;
		});
	}

	handlePointerDown(self, event) {
		if ([0, 2, 5].includes(event.button)) {
			event.preventDefault();
			const x = event.offsetX || event.targetTouches[0].clientX - maskRect.left;
			const y = event.offsetY || event.targetTouches[0].clientY - maskRect.top;

			const originalX = x * self.image.width / self.pointsCanvas.clientWidth;
			const originalY = y * self.image.height / self.pointsCanvas.clientHeight;

			var point = null;
			if (event.button == 0) {
				// positive
				point = [true, originalX, originalY];
			} else {
				// negative
				point = [false, originalX, originalY];
			}

			self.prompt_points.push(point);

			self.invalidatePointsCanvas(self);
		}
	}

	async save(self) {
		if(!self.mask_image) {
			this.close();
			return;
		}

		const save_canvas = document.createElement('canvas');

		const save_ctx = save_canvas.getContext('2d', {willReadFrequently:true});
		save_canvas.width = self.mask_image.width;
		save_canvas.height = self.mask_image.height;

		save_ctx.drawImage(self.mask_image, 0, 0, save_canvas.width, save_canvas.height);

		const save_data = save_ctx.getImageData(0, 0, save_canvas.width, save_canvas.height);

		// refine mask image
		for (let i = 0; i < save_data.data.length; i += 4) {
			if(save_data.data[i]) {
				save_data.data[i+3] = 0;
			}
			else {
				save_data.data[i+3] = 255;
			}

			save_data.data[i] = 0;
			save_data.data[i+1] = 0;
			save_data.data[i+2] = 0;
		}

		save_ctx.globalCompositeOperation = 'source-over';
		save_ctx.putImageData(save_data, 0, 0);

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
				ComfyApp.clipspace.widgets[index].value = `${filename} [temp]`;
		}

		const dataURL = save_canvas.toDataURL();
		const blob = dataURLToBlob(dataURL);

		let original_url = new URL(this.image.src);

		const original_ref = { filename: original_url.searchParams.get('filename') };

		let original_subfolder = original_url.searchParams.get("subfolder");
		if(original_subfolder)
			original_ref.subfolder = original_subfolder;

		let original_type = original_url.searchParams.get("type");
		if(original_type)
			original_ref.type = original_type;

		formData.append('image', blob, filename);
		formData.append('original_ref', JSON.stringify(original_ref));
		formData.append('type', "temp");

		await uploadMask(item, formData);
		ComfyApp.onClipspaceEditorSave();
		this.close();
	}
}

app.registerExtension({
	name: "Comfy.Impact.SAMEditor",
	init(app) {
		const callback =
			function () {
				let dlg = ImpactSamEditorDialog.getInstance();
				dlg.show();
			};

		const context_predicate = () => ComfyApp.clipspace && ComfyApp.clipspace.imgs && ComfyApp.clipspace.imgs.length > 0
		ClipspaceDialog.registerButton("Impact SAM Detector", context_predicate, callback);
	},

	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (Array.isArray(nodeData.output) && (nodeData.output.includes("MASK") || nodeData.output.includes("IMAGE"))) {
			addMenuHandler(nodeType, function (_, options) {
				options.unshift({
					content: "Open in SAM Detector",
					callback: () => {
						ComfyApp.copyToClipspace(this);
						ComfyApp.clipspace_return_node = this;

						let dlg = ImpactSamEditorDialog.getInstance();
						dlg.show();
					},
				});
			});
		}
	}
});

