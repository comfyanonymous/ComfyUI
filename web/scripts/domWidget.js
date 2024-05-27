import { app, ANIM_PREVIEW_WIDGET } from "./app.js";

const SIZE = Symbol();

function intersect(a, b) {
	const x = Math.max(a.x, b.x);
	const num1 = Math.min(a.x + a.width, b.x + b.width);
	const y = Math.max(a.y, b.y);
	const num2 = Math.min(a.y + a.height, b.y + b.height);
	if (num1 >= x && num2 >= y) return [x, y, num1 - x, num2 - y];
	else return null;
}

function getClipPath(node, element) {
	const selectedNode = Object.values(app.canvas.selected_nodes)[0];
	if (selectedNode && selectedNode !== node) {
		const elRect = element.getBoundingClientRect();
		const MARGIN = 7;
		const scale = app.canvas.ds.scale;

		const bounding = selectedNode.getBounding();
		const intersection = intersect(
			{ x: elRect.x / scale, y: elRect.y / scale, width: elRect.width / scale, height: elRect.height / scale },
			{
				x: selectedNode.pos[0] + app.canvas.ds.offset[0] - MARGIN,
				y: selectedNode.pos[1] + app.canvas.ds.offset[1] - LiteGraph.NODE_TITLE_HEIGHT - MARGIN,
				width: bounding[2] + MARGIN + MARGIN,
				height: bounding[3] + MARGIN + MARGIN,
			}
		);

		if (!intersection) {
			return "";
		}

		const widgetRect = element.getBoundingClientRect();
		const clipX = intersection[0] - widgetRect.x / scale + "px";
		const clipY = intersection[1] - widgetRect.y / scale + "px";
		const clipWidth = intersection[2] + "px";
		const clipHeight = intersection[3] + "px";
		const path = `polygon(0% 0%, 0% 100%, ${clipX} 100%, ${clipX} ${clipY}, calc(${clipX} + ${clipWidth}) ${clipY}, calc(${clipX} + ${clipWidth}) calc(${clipY} + ${clipHeight}), ${clipX} calc(${clipY} + ${clipHeight}), ${clipX} 100%, 100% 100%, 100% 0%)`;
		return path;
	}
	return "";
}

function computeSize(size) {
	if (this.widgets?.[0]?.last_y == null) return;

	let y = this.widgets[0].last_y;
	let freeSpace = size[1] - y;

	let widgetHeight = 0;
	let dom = [];
	for (const w of this.widgets) {
		if (w.type === "converted-widget") {
			// Ignore
			delete w.computedHeight;
		} else if (w.computeSize) {
			widgetHeight += w.computeSize()[1] + 4;
		} else if (w.element) {
			// Extract DOM widget size info
			const styles = getComputedStyle(w.element);
			let minHeight = w.options.getMinHeight?.() ?? parseInt(styles.getPropertyValue("--comfy-widget-min-height"));
			let maxHeight = w.options.getMaxHeight?.() ?? parseInt(styles.getPropertyValue("--comfy-widget-max-height"));

			let prefHeight = w.options.getHeight?.() ?? styles.getPropertyValue("--comfy-widget-height");
			if (prefHeight.endsWith?.("%")) {
				prefHeight = size[1] * (parseFloat(prefHeight.substring(0, prefHeight.length - 1)) / 100);
			} else {
				prefHeight = parseInt(prefHeight);
				if (isNaN(minHeight)) {
					minHeight = prefHeight;
				}
			}
			if (isNaN(minHeight)) {
				minHeight = 50;
			}
			if (!isNaN(maxHeight)) {
				if (!isNaN(prefHeight)) {
					prefHeight = Math.min(prefHeight, maxHeight);
				} else {
					prefHeight = maxHeight;
				}
			}
			dom.push({
				minHeight,
				prefHeight,
				w,
			});
		} else {
			widgetHeight += LiteGraph.NODE_WIDGET_HEIGHT + 4;
		}
	}

	freeSpace -= widgetHeight;

	// Calculate sizes with all widgets at their min height
	const prefGrow = []; // Nodes that want to grow to their prefd size
	const canGrow = []; // Nodes that can grow to auto size
	let growBy = 0;
	for (const d of dom) {
		freeSpace -= d.minHeight;
		if (isNaN(d.prefHeight)) {
			canGrow.push(d);
			d.w.computedHeight = d.minHeight;
		} else {
			const diff = d.prefHeight - d.minHeight;
			if (diff > 0) {
				prefGrow.push(d);
				growBy += diff;
				d.diff = diff;
			} else {
				d.w.computedHeight = d.minHeight;
			}
		}
	}

	if (this.imgs && !this.widgets.find((w) => w.name === ANIM_PREVIEW_WIDGET)) {
		// Allocate space for image
		freeSpace -= 220;
	}

	this.freeWidgetSpace = freeSpace;

	if (freeSpace < 0) {
		// Not enough space for all widgets so we need to grow
		size[1] -= freeSpace;
		this.graph.setDirtyCanvas(true);
	} else {
		// Share the space between each
		const growDiff = freeSpace - growBy;
		if (growDiff > 0) {
			// All pref sizes can be fulfilled
			freeSpace = growDiff;
			for (const d of prefGrow) {
				d.w.computedHeight = d.prefHeight;
			}
		} else {
			// We need to grow evenly
			const shared = -growDiff / prefGrow.length;
			for (const d of prefGrow) {
				d.w.computedHeight = d.prefHeight - shared;
			}
			freeSpace = 0;
		}

		if (freeSpace > 0 && canGrow.length) {
			// Grow any that are auto height
			const shared = freeSpace / canGrow.length;
			for (const d of canGrow) {
				d.w.computedHeight += shared;
			}
		}
	}

	// Position each of the widgets
	for (const w of this.widgets) {
		w.y = y;
		if (w.computedHeight) {
			y += w.computedHeight;
		} else if (w.computeSize) {
			y += w.computeSize()[1] + 4;
		} else {
			y += LiteGraph.NODE_WIDGET_HEIGHT + 4;
		}
	}
}

// Override the compute visible nodes function to allow us to hide/show DOM elements when the node goes offscreen
const elementWidgets = new Set();
const computeVisibleNodes = LGraphCanvas.prototype.computeVisibleNodes;
LGraphCanvas.prototype.computeVisibleNodes = function () {
	const visibleNodes = computeVisibleNodes.apply(this, arguments);
	for (const node of app.graph._nodes) {
		if (elementWidgets.has(node)) {
			const hidden = visibleNodes.indexOf(node) === -1;
			for (const w of node.widgets) {
				if (w.element) {
					w.element.hidden = hidden;
					w.element.style.display = hidden ? "none" : undefined;
					if (hidden) {
						w.options.onHide?.(w);
					}
				}
			}
		}
	}

	return visibleNodes;
};

let enableDomClipping = true;

export function addDomClippingSetting() {
	app.ui.settings.addSetting({
		id: "Comfy.DOMClippingEnabled",
		name: "Enable DOM element clipping (enabling may reduce performance)",
		type: "boolean",
		defaultValue: enableDomClipping,
		onChange(value) {
			enableDomClipping = !!value;
		},
	});
}

LGraphNode.prototype.addDOMWidget = function (name, type, element, options) {
	options = { hideOnZoom: true, selectOn: ["focus", "click"], ...options };

	if (!element.parentElement) {
		document.body.append(element);
	}

	let mouseDownHandler;
	if (element.blur) {
		mouseDownHandler = (event) => {
			if (!element.contains(event.target)) {
				element.blur();
			}
		};
		document.addEventListener("mousedown", mouseDownHandler);
	}

	const widget = {
		type,
		name,
		get value() {
			return options.getValue?.() ?? undefined;
		},
		set value(v) {
			options.setValue?.(v);
			widget.callback?.(widget.value);
		},
		draw: function (ctx, node, widgetWidth, y, widgetHeight) {
			if (widget.computedHeight == null) {
				computeSize.call(node, node.size);
			}

			const hidden =
				node.flags?.collapsed ||
				(!!options.hideOnZoom && app.canvas.ds.scale < 0.5) ||
				widget.computedHeight <= 0 ||
				widget.type === "converted-widget"||
				widget.type === "hidden";
			element.hidden = hidden;
			element.style.display = hidden ? "none" : null;
			if (hidden) {
				widget.options.onHide?.(widget);
				return;
			}

			const margin = 10;
			const elRect = ctx.canvas.getBoundingClientRect();
			const transform = new DOMMatrix()
				.scaleSelf(elRect.width / ctx.canvas.width, elRect.height / ctx.canvas.height)
				.multiplySelf(ctx.getTransform())
				.translateSelf(margin, margin + y);

			const scale = new DOMMatrix().scaleSelf(transform.a, transform.d);

			Object.assign(element.style, {
				transformOrigin: "0 0",
				transform: scale,
				left: `${transform.a + transform.e}px`,
				top: `${transform.d + transform.f}px`,
				width: `${widgetWidth - margin * 2}px`,
				height: `${(widget.computedHeight ?? 50) - margin * 2}px`,
				position: "absolute",
				zIndex: app.graph._nodes.indexOf(node),
			});

			if (enableDomClipping) {
				element.style.clipPath = getClipPath(node, element);
				element.style.willChange = "clip-path";
			}

			this.options.onDraw?.(widget);
		},
		element,
		options,
		onRemove() {
			if (mouseDownHandler) {
				document.removeEventListener("mousedown", mouseDownHandler);
			}
			element.remove();
		},
	};

	for (const evt of options.selectOn) {
		element.addEventListener(evt, () => {
			app.canvas.selectNode(this);
			app.canvas.bringToFront(this);
		});
	}

	this.addCustomWidget(widget);
	elementWidgets.add(this);

	const collapse = this.collapse;
	this.collapse = function() {
		collapse.apply(this, arguments);
		if(this.flags?.collapsed) {
			element.hidden = true;
			element.style.display = "none";
		}
	}

	const onRemoved = this.onRemoved;
	this.onRemoved = function () {
		element.remove();
		elementWidgets.delete(this);
		onRemoved?.apply(this, arguments);
	};

	if (!this[SIZE]) {
		this[SIZE] = true;
		const onResize = this.onResize;
		this.onResize = function (size) {
			options.beforeResize?.call(widget, this);
			computeSize.call(this, size);
			onResize?.apply(this, arguments);
			options.afterResize?.call(widget, this);
		};
	}

	return widget;
};
