function getNumberDefaults(inputData, defaultStep) {
	let defaultVal = inputData[1]["default"];
	let { min, max, step } = inputData[1];

	if (defaultVal == undefined) defaultVal = 0;
	if (min == undefined) min = 0;
	if (max == undefined) max = 2048;
	if (step == undefined) step = defaultStep;

	return { val: defaultVal, config: { min, max, step: 10.0 * step } };
}

export function addValueControlWidget(node, targetWidget, defaultValue = "randomize", values) {
    const valueControl = node.addWidget("combo", "control_after_generate", defaultValue, function (v) { }, {
        values: ["fixed", "increment", "decrement", "randomize"],
        serialize: false, // Don't include this in prompt.
    });
    valueControl.afterQueued = () => {

		var v = valueControl.value;

		if (targetWidget.type == "combo" && v !== "fixed") {
			let current_index = targetWidget.options.values.indexOf(targetWidget.value);
			let current_length = targetWidget.options.values.length;

			switch (v) {
				case "increment":
					current_index += 1;
					break;
				case "decrement":
					current_index -= 1;
					break;
				case "randomize":
					current_index = Math.floor(Math.random() * current_length);
				default:
					break;
			}
			current_index = Math.max(0, current_index);
			current_index = Math.min(current_length - 1, current_index);
			if (current_index >= 0) {
				let value = targetWidget.options.values[current_index];
				targetWidget.value = value;
				targetWidget.callback(value);
			}
		} else { //number
			let min = targetWidget.options.min;
			let max = targetWidget.options.max;
			// limit to something that javascript can handle
			max = Math.min(1125899906842624, max);
			min = Math.max(-1125899906842624, min);
			let range = (max - min) / (targetWidget.options.step / 10);

			//adjust values based on valueControl Behaviour
			switch (v) {
				case "fixed":
					break;
				case "increment":
					targetWidget.value += targetWidget.options.step / 10;
					break;
				case "decrement":
					targetWidget.value -= targetWidget.options.step / 10;
					break;
				case "randomize":
					targetWidget.value = Math.floor(Math.random() * range) * (targetWidget.options.step / 10) + min;
				default:
					break;
			}
		/*check if values are over or under their respective
		* ranges and set them to min or max.*/
			if (targetWidget.value < min)
				targetWidget.value = min;

			if (targetWidget.value > max)
				targetWidget.value = max;
		}
	}
	return valueControl;	
};

function seedWidget(node, inputName, inputData) {
	const seed = ComfyWidgets.INT(node, inputName, inputData);
	const seedControl = addValueControlWidget(node, seed.widget, "randomize");

	seed.widget.linkedWidgets = [seedControl];
	return seed;
}

const MultilineSymbol = Symbol();
const MultilineResizeSymbol = Symbol();

function addMultilineWidget(node, name, opts, app) {
	const MIN_SIZE = 50;

	function computeSize(size) {
		if (node.widgets[0].last_y == null) return;

		let y = node.widgets[0].last_y;
		let freeSpace = size[1] - y;

		// Compute the height of all non customtext widgets
		let widgetHeight = 0;
		const multi = [];
		for (let i = 0; i < node.widgets.length; i++) {
			const w = node.widgets[i];
			if (w.type === "customtext") {
				multi.push(w);
			} else {
				if (w.computeSize) {
					widgetHeight += w.computeSize()[1] + 4;
				} else {
					widgetHeight += LiteGraph.NODE_WIDGET_HEIGHT + 4;
				}
			}
		}

		// See how large each text input can be
		freeSpace -= widgetHeight;
		freeSpace /= multi.length + (!!node.imgs?.length);

		if (freeSpace < MIN_SIZE) {
			// There isnt enough space for all the widgets, increase the size of the node
			freeSpace = MIN_SIZE;
			node.size[1] = y + widgetHeight + freeSpace * (multi.length + (!!node.imgs?.length));
			node.graph.setDirtyCanvas(true);
		}

		// Position each of the widgets
		for (const w of node.widgets) {
			w.y = y;
			if (w.type === "customtext") {
				y += freeSpace;
			} else if (w.computeSize) {
				y += w.computeSize()[1] + 4;
			} else {
				y += LiteGraph.NODE_WIDGET_HEIGHT + 4;
			}
		}

		node.inputHeight = freeSpace;
	}

	const widget = {
		type: "customtext",
		name,
		get value() {
			return this.inputEl.value;
		},
		set value(x) {
			this.inputEl.value = x;
		},
		draw: function (ctx, _, widgetWidth, y, widgetHeight) {
			if (!this.parent.inputHeight) {
				// If we are initially offscreen when created we wont have received a resize event
				// Calculate it here instead
				computeSize(node.size);
			}
			const visible = app.canvas.ds.scale > 0.5 && this.type === "customtext";
			const margin = 10;
			const elRect = ctx.canvas.getBoundingClientRect();
			const transform = new DOMMatrix()
				.scaleSelf(elRect.width / ctx.canvas.width, elRect.height / ctx.canvas.height)
				.multiplySelf(ctx.getTransform())
				.translateSelf(margin, margin + y);

			Object.assign(this.inputEl.style, {
				transformOrigin: "0 0",
				transform: transform,
				left: "0px",
				top: "0px",
				width: `${widgetWidth - (margin * 2)}px`,
				height: `${this.parent.inputHeight - (margin * 2)}px`,
				position: "absolute",
				background: (!node.color)?'':node.color,
				color: (!node.color)?'':'white',
				zIndex: app.graph._nodes.indexOf(node),
			});
			this.inputEl.hidden = !visible;
		},
	};
	widget.inputEl = document.createElement("textarea");
	widget.inputEl.className = "comfy-multiline-input";
	widget.inputEl.value = opts.defaultVal;
	widget.inputEl.placeholder = opts.placeholder || "";
	document.addEventListener("mousedown", function (event) {
		if (!widget.inputEl.contains(event.target)) {
			widget.inputEl.blur();
		}
	});
	widget.parent = node;
	document.body.appendChild(widget.inputEl);

	node.addCustomWidget(widget);

	app.canvas.onDrawBackground = function () {
		// Draw node isnt fired once the node is off the screen
		// if it goes off screen quickly, the input may not be removed
		// this shifts it off screen so it can be moved back if the node is visible.
		for (let n in app.graph._nodes) {
			n = graph._nodes[n];
			for (let w in n.widgets) {
				let wid = n.widgets[w];
				if (Object.hasOwn(wid, "inputEl")) {
					wid.inputEl.style.left = -8000 + "px";
					wid.inputEl.style.position = "absolute";
				}
			}
		}
	};

	node.onRemoved = function () {
		// When removing this node we need to remove the input from the DOM
		for (let y in this.widgets) {
			if (this.widgets[y].inputEl) {
				this.widgets[y].inputEl.remove();
			}
		}
	};

	widget.onRemove = () => {
		widget.inputEl?.remove();

		// Restore original size handler if we are the last
		if (!--node[MultilineSymbol]) {
			node.onResize = node[MultilineResizeSymbol];
			delete node[MultilineSymbol];
			delete node[MultilineResizeSymbol];
		}
	};

	if (node[MultilineSymbol]) {
		node[MultilineSymbol]++;
	} else {
		node[MultilineSymbol] = 1;
		const onResize = (node[MultilineResizeSymbol] = node.onResize);

		node.onResize = function (size) {
			computeSize(size);

			// Call original resizer handler
			if (onResize) {
				onResize.apply(this, arguments);
			}
		};
	}

	return { minWidth: 400, minHeight: 200, widget };
}

const FLOAT = (node, inputName, inputData) => {
	const { val, config } = getNumberDefaults(inputData, 0.5);
	return { widget: node.addWidget("number", inputName, val, () => {}, config) };
}

const INT = (node, inputName, inputData) => {
	const { val, config } = getNumberDefaults(inputData, 1);
	Object.assign(config, { precision: 0 });
	return {
		widget: node.addWidget(
			"number",
			inputName,
			val,
			function (v) {
				const s = this.options.step / 10;
				this.value = Math.round(v / s) * s;
			},
			config
		),
	};
}

const STRING = (node, inputName, inputData, app) => {
	const defaultVal = inputData[1].default || "";
	const multiline = !!inputData[1].multiline;

	if (multiline) {
		return addMultilineWidget(node, inputName, { defaultVal, ...inputData[1] }, app);
	} else {
		return { widget: node.addWidget("text", inputName, defaultVal, () => {}, {}) };
	}
}

const COMBO = (node, inputName, inputData) => {
	const type = inputData[0];
	let defaultValue = type[0];
    let options = inputData[1] || {}
	if (options.default) {
		defaultValue = options.default
	}

	if (options.is_list) {
		defaultValue = [defaultValue]
		const widget = node.addWidget("text", inputName, defaultValue, () => {}, { values: type })
		widget.disabled = true;
		return { widget };
	}
	else {
		return { widget: node.addWidget("combo", inputName, defaultValue, () => {}, { values: type }) };
	}
}

const IMAGEUPLOAD = (node, inputName, inputData, app) => {
	const imageWidget = node.widgets.find((w) => w.name === "image");
	let uploadWidget;

	function showImage(name) {
		const img = new Image();
		img.onload = () => {
			node.imgs = [img];
			app.graph.setDirtyCanvas(true);
		};
		let folder_separator = name.lastIndexOf("/");
		let subfolder = "";
		if (folder_separator > -1) {
			subfolder = name.substring(0, folder_separator);
			name = name.substring(folder_separator + 1);
		}
		img.src = `/view?filename=${name}&type=input&subfolder=${subfolder}`;
		node.setSizeForImage?.();
	}

	var default_value = imageWidget.value;
	Object.defineProperty(imageWidget, "value", {
		set : function(value) {
			this._real_value = value;
		},

		get : function() {
			let value = "";
			if (this._real_value) {
				value = this._real_value;
			} else {
				return default_value;
			}

			if (value.filename) {
				let real_value = value;
				value = "";
				if (real_value.subfolder) {
					value = real_value.subfolder + "/";
				}

				value += real_value.filename;

				if(real_value.type && real_value.type !== "input")
					value += ` [${real_value.type}]`;
			}
			return value;
		}
	});

	// Add our own callback to the combo widget to render an image when it changes
	const cb = node.callback;
	imageWidget.callback = function () {
		showImage(imageWidget.value);
		if (cb) {
			return cb.apply(this, arguments);
		}
	};

	// On load if we have a value then render the image
	// The value isnt set immediately so we need to wait a moment
	// No change callbacks seem to be fired on initial setting of the value
	requestAnimationFrame(() => {
		if (imageWidget.value) {
			showImage(imageWidget.value);
		}
	});

	async function uploadFile(file, updateNode) {
		try {
			// Wrap file in formdata so it includes filename
			const body = new FormData();
			body.append("image", file);
			const resp = await fetch("/upload/image", {
				method: "POST",
				body,
			});

			if (resp.status === 200) {
				const data = await resp.json();
				// Add the file as an option and update the widget value
				if (!imageWidget.options.values.includes(data.name)) {
					imageWidget.options.values.push(data.name);
				}

				if (updateNode) {
					showImage(data.name);

					imageWidget.value = data.name;
				}
			} else {
				alert(resp.status + " - " + resp.statusText);
			}
		} catch (error) {
			alert(error);
		}
	}

	const fileInput = document.createElement("input");
	Object.assign(fileInput, {
		type: "file",
		accept: "image/jpeg,image/png,image/webp",
		style: "display: none",
		onchange: async () => {
			if (fileInput.files.length) {
				await uploadFile(fileInput.files[0], true);
			}
		},
	});
	document.body.append(fileInput);

	// Create the button widget for selecting the files
	uploadWidget = node.addWidget("button", "choose file to upload", "image", () => {
		fileInput.value = null;
		fileInput.click();
	}, { serialize: false });

	// Add handler to check if an image is being dragged over our node
	node.onDragOver = function (e) {
		if (e.dataTransfer && e.dataTransfer.items) {
			const image = [...e.dataTransfer.items].find((f) => f.kind === "file" && f.type.startsWith("image/"));
			return !!image;
		}

		return false;
	};

	// On drop upload files
	node.onDragDrop = function (e) {
		console.log("onDragDrop called");
		let handled = false;
		for (const file of e.dataTransfer.files) {
			if (file.type.startsWith("image/")) {
				uploadFile(file, !handled); // Dont await these, any order is fine, only update on first one
				handled = true;
			}
		}

		return handled;
	};

	return { widget: uploadWidget };
}

async function loadImageAsync(imageURL) {
    return new Promise((resolve) => {
        const e = new Image();
        e.setAttribute('crossorigin', 'anonymous');
        e.addEventListener("load", () => { resolve(e); });
        e.src = imageURL;
        return e;
    });
}

const MULTIIMAGEUPLOAD = (node, inputName, inputData, app) => {
	let filepaths = { input: [], output: [] }

	if (inputData[1] && inputData[1].filepaths) {
		filepaths = inputData[1].filepaths
	}

	const update = function(v) {
		this.value = v
	}

	const imagesWidget = node.addWidget("combo", inputName, inputData, update, { values: filepaths["input"] })
	imagesWidget._filepaths = filepaths
	imagesWidget._entries = filepaths["input"]

	async function showImages(names) {
		node.imgs = []

		for (const name of names) {
			let folder_separator = name.lastIndexOf("/");
			let subfolder = "";
			if (folder_separator > -1) {
				subfolder = name.substring(0, folder_separator);
				name = name.substring(folder_separator + 1);
			}
			const src = `/view?filename=${name}&type=input&subfolder=${subfolder}${app.getPreviewFormatParam()}`;
			const img = await loadImageAsync(src);
			node.imgs.push(img)
			node.imageIndex = null;
			node.setSizeForImage?.();
			app.graph.setDirtyCanvas(true);
		}
	}

	var default_value = imagesWidget.value;
	Object.defineProperty(imagesWidget, "value", {
		set : function(value) {
			if (typeof value === "string") {
				value = [value]
			}
			this._real_value = value;
		},

		get : function() {
			this._real_value ||= []

			const result = []

			for (const value of this._real_value) {
				if (value.filename) {
					let real_value = value;
					value = "";
					if (real_value.subfolder) {
						value = real_value.subfolder + "/";
					}

					value += real_value.filename;

					if(real_value.type && real_value.type !== "input")
						value += ` [${real_value.type}]`;
				}

				result.push(value)
			}

			this._real_value = result
			return this._real_value;
		}
	});

	// Add our own callback to the combo widget to render an image when it changes
	const cb = node.callback;
	imagesWidget.callback = () => {
		showImages(imagesWidget.value).then(() => {
			if (cb) {
				return cb.apply(this, arguments);
			}
		})
	};

	// On load if we have a value then render the image
	// The value isnt set immediately so we need to wait a moment
	// No change callbacks seem to be fired on initial setting of the value
	requestAnimationFrame(async () => {
		if (Array.isArray(imagesWidget.value) && imagesWidget.value.length > 0) {
			await showImages(imagesWidget.value);
		}
	});

	async function uploadFiles(files, updateNode) {
		for (const file of files) {
			try {
				// Wrap file in formdata so it includes filename
				const body = new FormData();
				body.append("image", file);
				const resp = await fetch("/upload/image", {
					method: "POST",
					body,
				});

				if (resp.status === 200) {
					const data = await resp.json();
					if (updateNode) {
						imagesWidget.value.push(data.name)
					}
				} else {
					alert(resp.status + " - " + resp.statusText);
				}
			} catch (error) {
				alert(error);
			}
		}

		if (updateNode) {
			await showImages(imagesWidget.value);
		}
	}

	const fileInput = document.createElement("input");
	Object.assign(fileInput, {
		type: "file",
		multiple: "multiple",
		accept: "image/jpeg,image/png,image/webp",
		style: "display: none",
		onchange: async () => {
			if (fileInput.files.length) {
				await uploadFiles(fileInput.files, true);
			}
		},
	});
	document.body.append(fileInput);

	// Create the button widget for selecting the files
	const pickWidget = node.addWidget("button", "pick files from ComfyUI folders", "images", () => {
		const graphCanvas = LiteGraph.LGraphCanvas.active_canvas
		if (graphCanvas == null)
			return;

		if (imagesWidget.panel != null)
			return

		imagesWidget.panel = graphCanvas.createPanel("Pick Images", { closable: true });
		imagesWidget.panel.onClose = () => {
			imagesWidget.panel = null;
		}
        imagesWidget.panel.node = node;
        imagesWidget.panel.classList.add("multiimageupload_dialog");
        const swap = (arr, i, j) => {
            const temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }

		const rootHtml = `
<div class="left">
</div>
<div class="right">
</div>
`;
		const rootElem = imagesWidget.panel.addHTML(rootHtml, "root");
		const left = rootElem.querySelector('.left')
		const right = rootElem.querySelector('.right')

		const previewHtml = `
<img class="image-preview" src="" />
<div class="bar">
	<select class='folder-type'>
       <option value="output">Output</option>
       <option value="input">Input</option>
	</select>
	<button class='image-path'><span class="image-path-text"></span></button>
</div>
<div class="bar actions">
    <button class="add-image">Add</button>
    <button class="replace-image">Replace</button>
</div>`;
		const previewElem = document.createElement("div");
		previewElem.innerHTML = previewHtml;
		previewElem.className = "multiimageupload_preview";
		right.appendChild(previewElem);

		const folderTypeSel = previewElem.querySelector('.folder-type');
		const imagePathSel = previewElem.querySelector('.image-path');
		const imagePathText = previewElem.querySelector('.image-path-text');
		const imagePreview = previewElem.querySelector('.image-preview');

		folderTypeSel.addEventListener("change", (event) => {
			const filepaths = imagesWidget._filepaths[event.target.value];
			imagesWidget._entries = filepaths
			imagePathText.innerHTML = filepaths[0];
			imagePreview.src = `/view?filename=${filepaths[0]}&type=${event.target.value}`
		});

		imagePathSel.addEventListener("click", (event) => {
			const type = folderTypeSel.value;
			const filepaths = imagesWidget._filepaths[folderTypeSel.value];
			const entries = imagesWidget._entries

			const innerClicked = (v, _options, e, prev) => {
				const filename = v;
				imagePathText.innerHTML = filename;
				imagePreview.src = `/view?filename=${filename}&type=${type}`
			}

			new LiteGraph.ContextMenu(entries, {
				event,
				callback: innerClicked,
				node,
				className: "dark" // required for contextMenuFilter.js to kick in
			});
		});

		folderTypeSel.value = "input";
		folderTypeSel.dispatchEvent(new Event('change'));

		const addButton = previewElem.querySelector('.add-image');
		addButton.addEventListener("click", async (event) => {
			const filename = imagePathText.innerHTML;
			const type = folderTypeSel.value;
			let value = filename;
			if (type !== "input")
				value += ` [${type}]`
			imagesWidget._real_value.push(value)
			imagesWidget.value = imagesWidget._real_value
			await showImages(imagesWidget.value);
			inner_refresh();
		})

		const replaceButton = previewElem.querySelector('.replace-image');
		replaceButton.addEventListener("click", async (event) => {
			const filename = imagePathText.innerHTML;
			const type = folderTypeSel.value;
			let value = filename;
			if (type !== "input")
				value += ` [${type}]`
			imagesWidget._real_value = [value]
			imagesWidget.value = imagesWidget._real_value
			await showImages(imagesWidget.value);
			inner_refresh();
		})

		imagesWidget.panel.footer.style.display = "flex";

		const clearButton = imagesWidget.panel.addButton("Clear", () => {
			imagesWidget.value = []
			showImages(imagesWidget.value);
			inner_refresh();
		})
		clearButton.style.display = "block";
		clearButton.style.marginLeft = "initial";
		clearButton.style.height = "28px";

		const okButton = imagesWidget.panel.addButton("OK", () => {
			imagesWidget.panel.close();
		})
		okButton.style.display = "block";
		okButton.style.height = "28px";
		okButton.style.marginLeft = "auto";

		const inner_refresh = () => {
			left.innerHTML = ""
            graphCanvas.draw(true);

			if (node.imgs) {
				for (let i = 0; i < imagesWidget.value.length; i++) {
                    const imagePath = imagesWidget.value[i];
                    const img = node.imgs[i];
                    if (!imagePath || !img)
                        continue;
					const html = `
<img src="${img.src}" />
<div class="bar">
    <button class="delete">&#10005;</button>
    <button class="move_up">↑</button>
    <button class="move_down">↓</button>
    <span class='image-path'></span>
    <span class='type'></span>
</div>`;
                    const elem = document.createElement("div");
                    elem.innerHTML = html;
                    elem.className = "multiimageupload_image";
                    left.appendChild(elem);

                    elem.dataset["imagePath"] = imagePath
                    elem.dataset["imageIndex"] = "" + i;
                    elem.querySelector(".image-path").innerText = imagePath
                    elem.querySelector(".type").innerText = ""
                    elem.querySelector(".delete").addEventListener("click", function(e) {
                        const imageIndex = +this.parentNode.parentNode.dataset["imageIndex"]
                        imagesWidget._real_value.splice(imageIndex, 1)
                        imagesWidget.value = imagesWidget._real_value
                        node.imgs.splice(imageIndex, 1);
                        node.imageIndex = null;
                        node.setSizeForImage?.();
                        inner_refresh();
                    });
                    const move_up = elem.querySelector(".move_up");
                    move_up.disabled = i <= 0;
                    move_up.addEventListener("click", function(e) {
                        const imageIndex = +this.parentNode.parentNode.dataset["imageIndex"]
                        if (imageIndex < 0)
                            return;
                        swap(imagesWidget.value, imageIndex, imageIndex - 1);
                        swap(node.imgs, imageIndex, imageIndex - 1);
                        inner_refresh();
                    });
                    const move_down = elem.querySelector(".move_down")
                    move_down.disabled = i >= imagesWidget.value.length - 1;
                    move_down.addEventListener("click", function(e) {
                        const imageIndex = +this.parentNode.parentNode.dataset["imageIndex"]
                        if (imageIndex > imagesWidget.value.length - 1)
                            return;
                        swap(imagesWidget.value, imageIndex, imageIndex + 1);
                        swap(node.imgs, imageIndex, imageIndex + 1);
                        inner_refresh();
                    });
                }
            }
        }

        inner_refresh();
        document.body.appendChild(imagesWidget.panel);
	}, { serialize: false });

	const uploadWidget = node.addWidget("button", "choose files to upload", "images", () => {
		fileInput.value = null;
		fileInput.click();
	}, { serialize: false });

	const clearWidget = node.addWidget("button", "clear all uploads", "images", () => {
		imagesWidget.value = []
		showImages(imagesWidget.value);
	}, { serialize: false });

	// Add handler to check if an image is being dragged over our node
	node.onDragOver = function (e) {
		if (e.dataTransfer && e.dataTransfer.items) {
			const image = [...e.dataTransfer.items].find((f) => f.kind === "file" && f.type.startsWith("image/"));
			return !!image;
		}

		return false;
	};

	// On drop upload files
	node.onDragDrop = async (e) => {
		console.log("onDragDrop called");
		let handled = false;
		const files = Array.from(e.dataTransfer.files).filter(file => file.type.startsWith("image/"))
		if (files) {
			await uploadFiles(files, true);
			handled = true;
		}

		return handled;
	};

	return { widget: uploadWidget };
}

export const ComfyWidgets = {
	"INT:seed": seedWidget,
	"INT:noise_seed": seedWidget,
	FLOAT,
	INT,
	STRING,
	COMBO,
	IMAGEUPLOAD,
	MULTIIMAGEUPLOAD,
};
