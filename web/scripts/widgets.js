function getNumberDefaults(inputData, defaultStep) {
	let defaultVal = inputData[1]["default"];
	let { min, max, step } = inputData[1];

	if (defaultVal == undefined) defaultVal = 0;
	if (min == undefined) min = 0;
	if (max == undefined) max = 2048;
	if (step == undefined) step = defaultStep;

	return { val: defaultVal, config: { min, max, step: 10.0 * step } };
}

function seedWidget(node, inputName, inputData) {
	const seed = ComfyWidgets.INT(node, inputName, inputData);
	const randomize = node.addWidget("toggle", "Random seed after every gen", true, function (v) {}, {
		on: "enabled",
		off: "disabled",
		serialize: false, // Don't include this in prompt.
	});

	randomize.afterQueued = () => {
		if (randomize.value) {
			seed.widget.value = Math.floor(Math.random() * 1125899906842624);
		}
	};

	return { widget: seed, randomize };
}

function addMultilineWidget(node, name, defaultVal, dynamicPrompt, app) {
	const widget = {
		type: "customtext",
		name,
		get value() {
			return this.inputEl.value;
		},
		set value(x) {
			this.inputEl.value = x;
		},
		options: {
			dynamicPrompt,
		},
		draw: function (ctx, _, widgetWidth, y, widgetHeight) {
			const visible = app.canvas.ds.scale > 0.5;
			const t = ctx.getTransform();
			const margin = 10;
			Object.assign(this.inputEl.style, {
				left: `${t.a * margin + t.e}px`,
				top: `${t.d * (y + widgetHeight - margin) + t.f}px`,
				width: `${(widgetWidth - margin * 2 - 3) * t.a}px`,
				height: `${(this.parent.size[1] - (y + widgetHeight) - 3) * t.d}px`,
				position: "absolute",
				zIndex: 1,
				fontSize: `${t.d * 10.0}px`,
			});
			this.inputEl.hidden = !visible;
		},
	};
	widget.inputEl = document.createElement("textarea");
	widget.inputEl.className = "comfy-multiline-input";
	widget.inputEl.value = defaultVal;
	document.addEventListener("click", function (event) {
		if (!widget.inputEl.contains(event.target)) {
			widget.inputEl.blur();
		}
	});
	widget.parent = node;
	document.body.appendChild(widget.inputEl);

	node.addCustomWidget(widget);

	node.onRemoved = function () {
		// When removing this node we need to remove the input from the DOM
		for (let y in this.widgets) {
			if (this.widgets[y].inputEl) {
				this.widgets[y].inputEl.remove();
			}
		}
	};

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
		const dynamicPrompt = !!inputData[1].dynamic_prompt;

		if (multiline) {
			return addMultilineWidget(node, inputName, defaultVal, dynamicPrompt, app);
		} else {
			return { widget: node.addWidget("text", inputName, defaultVal, () => {}, { dynamicPrompt }) };
		}
	},
};
