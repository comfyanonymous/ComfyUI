function getNumberDefaults(inputData, defaultStep) {
	let defaultVal = inputData[1]["default"];
	let { min, max, step } = inputData[1];

	if (defaultVal == undefined) defaultVal = 0;
	if (min == undefined) min = 0;
	if (max == undefined) max = 2048;
	if (step == undefined) step = defaultStep;

	return { val: defaultVal, config: { min, max, step: 10.0 * step } };
}

export function addRandomizeWidget(node, targetWidget, name, defaultValue = false) {
	const randomize = node.addWidget("toggle", name, defaultValue, function (v) {}, {
		on: "enabled",
		off: "disabled",
		serialize: false, // Don't include this in prompt.
	});

	randomize.afterQueued = () => {
		if (randomize.value) {
			const min = targetWidget.options?.min;
			let max = targetWidget.options?.max;
			if (min != null || max != null) {
				if (max) {
					// limit max to something that javascript can handle
					max = Math.min(1125899906842624, max);
				}
				targetWidget.value = Math.floor(Math.random() * ((max ?? 9999999999) - (min ?? 0) + 1) + (min ?? 0));
			} else {
				targetWidget.value = Math.floor(Math.random() * 1125899906842624);
			}
		}
	};
	return randomize;
}

function seedWidget(node, inputName, inputData) {
	const seed = ComfyWidgets.INT(node, inputName, inputData);
	const randomize = addRandomizeWidget(node, seed.widget, "Random seed after every gen", true);

	seed.widget.linkedWidgets = [randomize];
	return { widget: seed, randomize };
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
		freeSpace /= multi.length;

		if (freeSpace < MIN_SIZE) {
			// There isnt enough space for all the widgets, increase the size of the node
			freeSpace = MIN_SIZE;
			node.size[1] = y + widgetHeight + freeSpace * multi.length;
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
			const t = ctx.getTransform();
			const margin = 10;
			Object.assign(this.inputEl.style, {
				left: `${t.a * margin + t.e}px`,
				top: `${t.d * (y + widgetHeight - margin - 3) + t.f}px`,
				width: `${(widgetWidth - margin * 2 - 3) * t.a}px`,
				height: `${(this.parent.inputHeight - margin * 2 - 4) * t.d}px`,
				position: "absolute",
				zIndex: 1,
				fontSize: `${t.d * 10.0}px`,
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

export const ComfyWidgets = {
	"INT:seed": seedWidget,
	"INT:noise_seed": seedWidget,
	FLOAT(node, inputName, inputData) {
		const { val, config } = getNumberDefaults(inputData, 0.5);
		return { widget: node.addWidget("number", inputName, val, () => {}, config) };
	},
	INT(node, inputName, inputData) {
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
	},
	STRING(node, inputName, inputData, app) {
		const defaultVal = inputData[1].default || "";
		const multiline = !!inputData[1].multiline;

		if (multiline) {
			return addMultilineWidget(node, inputName, { defaultVal, ...inputData[1] }, app);
		} else {
			return { widget: node.addWidget("text", inputName, defaultVal, () => {}, {}) };
		}
	},
	COMBO(node, inputName, inputData) {
		const type = inputData[0];
		let defaultValue = type[0];
		if (inputData[1] && inputData[1].default) {
			defaultValue = inputData[1].default;
		}
		return { widget: node.addWidget("combo", inputName, defaultValue, () => {}, { values: type }) };
	},
	IMAGEUPLOAD(node, inputName, inputData, app) {
		const imageWidget = node.widgets.find((w) => w.name === "image");
		let uploadWidget;

		function showImage(name) {
			// Position the image somewhere sensible
			if (!node.imageOffset) {
				node.imageOffset = uploadWidget.last_y ? uploadWidget.last_y + 25 : 75;
			}

			const img = new Image();
			img.onload = () => {
				node.imgs = [img];
				app.graph.setDirtyCanvas(true);
			};
			img.src = `/view?filename=${name}&type=input`;
		}

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
			accept: "image/jpeg,image/png",
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
			fileInput.click();
		});
		uploadWidget.serialize = false;

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
	},
};
