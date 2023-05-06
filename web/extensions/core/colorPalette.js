import { app } from "/scripts/app.js";
import { $el } from "/scripts/ui.js";
import { api } from "/scripts/api.js";

// Manage color palettes

const colorPalettes = {
	"dark": {
		"id": "dark",
		"name": "Dark (Default)",
		"colors": {
			"node_slot": {
				"CLIP": "#FFD500", // bright yellow
				"CLIP_VISION": "#A8DADC", // light blue-gray
				"CLIP_VISION_OUTPUT": "#ad7452", // rusty brown-orange
				"CONDITIONING": "#FFA931", // vibrant orange-yellow
				"CONTROL_NET": "#6EE7B7", // soft mint green
				"IMAGE": "#64B5F6", // bright sky blue
				"LATENT": "#FF9CF9", // light pink-purple
				"MASK": "#81C784", // muted green
				"MODEL": "#B39DDB", // light lavender-purple
				"STYLE_MODEL": "#C2FFAE", // light green-yellow
				"VAE": "#FF6E6E", // bright red
			},
			"litegraph_base": {
				"NODE_TITLE_COLOR": "#999",
				"NODE_SELECTED_TITLE_COLOR": "#FFF",
				"NODE_TEXT_SIZE": 14,
				"NODE_TEXT_COLOR": "#AAA",
				"NODE_SUBTEXT_SIZE": 12,
				"NODE_DEFAULT_COLOR": "#333",
				"NODE_DEFAULT_BGCOLOR": "#353535",
				"NODE_DEFAULT_BOXCOLOR": "#666",
				"NODE_DEFAULT_SHAPE": "box",
				"NODE_BOX_OUTLINE_COLOR": "#FFF",
				"DEFAULT_SHADOW_COLOR": "rgba(0,0,0,0.5)",
				"DEFAULT_GROUP_FONT": 24,

				"WIDGET_BGCOLOR": "#222",
				"WIDGET_OUTLINE_COLOR": "#666",
				"WIDGET_TEXT_COLOR": "#DDD",
				"WIDGET_SECONDARY_TEXT_COLOR": "#999",

				"LINK_COLOR": "#9A9",
				"EVENT_LINK_COLOR": "#A86",
				"CONNECTING_LINK_COLOR": "#AFA",
			},
			"comfy_base": {
				"fg-color": "#fff",
				"bg-color": "#202020",
				"comfy-menu-bg": "#353535",
				"comfy-input-bg": "#222",
				"input-text": "#ddd",
				"descrip-text": "#999",
				"drag-text": "#ccc",
				"error-text": "#ff4444",
				"border-color": "#4e4e4e"
			}
		},
	},
	"light": {
		"id": "light",
		"name": "Light",
		"colors": {
			"node_slot": {
				"CLIP": "#FFA726", // orange
				"CLIP_VISION": "#5C6BC0", // indigo
				"CLIP_VISION_OUTPUT": "#8D6E63", // brown
				"CONDITIONING": "#EF5350", // red
				"CONTROL_NET": "#66BB6A", // green
				"IMAGE": "#42A5F5", // blue
				"LATENT": "#AB47BC", // purple
				"MASK": "#9CCC65", // light green
				"MODEL": "#7E57C2", // deep purple
				"STYLE_MODEL": "#D4E157", // lime
				"VAE": "#FF7043", // deep orange
			},
			"litegraph_base": {
				"NODE_TITLE_COLOR": "#222",
				"NODE_SELECTED_TITLE_COLOR": "#000",
				"NODE_TEXT_SIZE": 14,
				"NODE_TEXT_COLOR": "#444",
				"NODE_SUBTEXT_SIZE": 12,
				"NODE_DEFAULT_COLOR": "#F7F7F7",
				"NODE_DEFAULT_BGCOLOR": "#F5F5F5",
				"NODE_DEFAULT_BOXCOLOR": "#CCC",
				"NODE_DEFAULT_SHAPE": "box",
				"NODE_BOX_OUTLINE_COLOR": "#000",
				"DEFAULT_SHADOW_COLOR": "rgba(0,0,0,0.1)",
				"DEFAULT_GROUP_FONT": 24,

				"WIDGET_BGCOLOR": "#D4D4D4",
				"WIDGET_OUTLINE_COLOR": "#999",
				"WIDGET_TEXT_COLOR": "#222",
				"WIDGET_SECONDARY_TEXT_COLOR": "#555",

				"LINK_COLOR": "#4CAF50",
				"EVENT_LINK_COLOR": "#FF9800",
				"CONNECTING_LINK_COLOR": "#2196F3",
			},
			"comfy_base": {
				"fg-color": "#222",
				"bg-color": "#DDD",
				"comfy-menu-bg": "#F5F5F5",
				"comfy-input-bg": "#C9C9C9",
				"input-text": "#222",
				"descrip-text": "#444",
				"drag-text": "#555",
				"error-text": "#F44336",
				"border-color": "#888"
			}
		},
	},
	"solarized": {
		"id": "solarized",
		"name": "Solarized",
		"colors": {
			"node_slot": {
				"CLIP": "#2AB7CA", // light blue
				"CLIP_VISION": "#6c71c4", // blue violet
				"CLIP_VISION_OUTPUT": "#859900", // olive green
				"CONDITIONING": "#d33682", // magenta
				"CONTROL_NET": "#d1ffd7", // light mint green
				"IMAGE": "#5940bb", // deep blue violet
				"LATENT": "#268bd2", // blue
				"MASK": "#CCC9E7", // light purple-gray
				"MODEL": "#dc322f", // red
				"STYLE_MODEL": "#1a998a", // teal
				"UPSCALE_MODEL": "#054A29", // dark green
				"VAE": "#facfad", // light pink-orange
			},
			"litegraph_base": {
				"NODE_TITLE_COLOR": "#fdf6e3", // Base3
				"NODE_SELECTED_TITLE_COLOR": "#A9D400",
				"NODE_TEXT_SIZE": 14,
				"NODE_TEXT_COLOR": "#657b83", // Base00
				"NODE_SUBTEXT_SIZE": 12,
				"NODE_DEFAULT_COLOR": "#094656",
				"NODE_DEFAULT_BGCOLOR": "#073642", // Base02
				"NODE_DEFAULT_BOXCOLOR": "#839496", // Base0
				"NODE_DEFAULT_SHAPE": "box",
				"NODE_BOX_OUTLINE_COLOR": "#fdf6e3", // Base3
				"DEFAULT_SHADOW_COLOR": "rgba(0,0,0,0.5)",
				"DEFAULT_GROUP_FONT": 24,

				"WIDGET_BGCOLOR": "#002b36", // Base03
				"WIDGET_OUTLINE_COLOR": "#839496", // Base0
				"WIDGET_TEXT_COLOR": "#fdf6e3", // Base3
				"WIDGET_SECONDARY_TEXT_COLOR": "#93a1a1", // Base1

				"LINK_COLOR": "#2aa198", // Solarized Cyan
				"EVENT_LINK_COLOR": "#268bd2", // Solarized Blue
				"CONNECTING_LINK_COLOR": "#859900", // Solarized Green
			},
			"comfy_base": {
				"fg-color": "#fdf6e3", // Base3
				"bg-color": "#002b36", // Base03
				"comfy-menu-bg": "#073642", // Base02
				"comfy-input-bg": "#002b36", // Base03
				"input-text": "#93a1a1", // Base1
				"descrip-text": "#586e75", // Base01
				"drag-text": "#839496", // Base0
				"error-text": "#dc322f", // Solarized Red
				"border-color": "#657b83" // Base00
			}
		},
	}
};

const id = "Comfy.ColorPalette";
const idCustomColorPalettes = "Comfy.CustomColorPalettes";
const defaultColorPaletteId = "dark";
const els = {}
// const ctxMenu = LiteGraph.ContextMenu;
app.registerExtension({
	name: id,
	init() {
		const sortObjectKeys = (unordered) => {
			return Object.keys(unordered).sort().reduce((obj, key) => {
				obj[key] = unordered[key];
				return obj;
			}, {});
		};

		const getSlotTypes = async () => {
			var types = [];

			const defs = await api.getNodeDefs();
			for (const nodeId in defs) {
				const nodeData = defs[nodeId];

				var inputs = nodeData["input"]["required"];
				if (nodeData["input"]["optional"] != undefined){
					inputs = Object.assign({}, nodeData["input"]["required"], nodeData["input"]["optional"])
				}

				for (const inputName in inputs) {
					const inputData = inputs[inputName];
					const type = inputData[0];

					if (!Array.isArray(type)) {
						types.push(type);
					}
				}

				for (const o in nodeData["output"]) {
					const output = nodeData["output"][o];
					types.push(output);
				}
			}

			return types;
		};

		const completeColorPalette = async (colorPalette) => {
			var types = await getSlotTypes();

			for (const type of types) {
				if (!colorPalette.colors.node_slot[type]) {
					colorPalette.colors.node_slot[type] = "";
				}
			}

			colorPalette.colors.node_slot = sortObjectKeys(colorPalette.colors.node_slot);

			return colorPalette;
		};

		const getColorPaletteTemplate = async () => {
			let colorPalette = {
				"id": "my_color_palette_unique_id",
				"name": "My Color Palette",
				"colors": {
					"node_slot": {
					},
					"litegraph_base": {
					},
					"comfy_base": {
					}
				}
			};

			// Copy over missing keys from default color palette
			const defaultColorPalette = colorPalettes[defaultColorPaletteId];
			for (const key in defaultColorPalette.colors.litegraph_base) {
				if (!colorPalette.colors.litegraph_base[key]) {
					colorPalette.colors.litegraph_base[key] = "";
				}
			}
			for (const key in defaultColorPalette.colors.comfy_base) {
				if (!colorPalette.colors.comfy_base[key]) {
					colorPalette.colors.comfy_base[key] = "";
				}
			}

			return completeColorPalette(colorPalette);
		};

		const getCustomColorPalettes = () => {
			return app.ui.settings.getSettingValue(idCustomColorPalettes, {});
		};

		const setCustomColorPalettes = (customColorPalettes) => {
			return app.ui.settings.setSettingValue(idCustomColorPalettes, customColorPalettes);
		};

		const addCustomColorPalette = async (colorPalette) => {
			if (typeof(colorPalette) !== "object") {
				app.ui.dialog.show("Invalid color palette");
				return;
			}

			if (!colorPalette.id) {
				app.ui.dialog.show("Color palette missing id");
				return;
			}

			if (!colorPalette.name) {
				app.ui.dialog.show("Color palette missing name");
				return;
			}

			if (!colorPalette.colors) {
				app.ui.dialog.show("Color palette missing colors");
				return;
			}

			if (colorPalette.colors.node_slot && typeof(colorPalette.colors.node_slot) !== "object") {
				app.ui.dialog.show("Invalid color palette colors.node_slot");
				return;
			}

			let customColorPalettes = getCustomColorPalettes();
			customColorPalettes[colorPalette.id] = colorPalette;
			setCustomColorPalettes(customColorPalettes);

			for (const option of els.select.childNodes) {
				if (option.value === "custom_" + colorPalette.id) {
					els.select.removeChild(option);
				}
			}

			els.select.append($el("option", { textContent: colorPalette.name + " (custom)", value: "custom_" + colorPalette.id, selected: true }));

			setColorPalette("custom_" + colorPalette.id);
			await loadColorPalette(colorPalette);
		};

		const deleteCustomColorPalette = async (colorPaletteId) => {
			let customColorPalettes = getCustomColorPalettes();
			delete customColorPalettes[colorPaletteId];
			setCustomColorPalettes(customColorPalettes);

			for (const option of els.select.childNodes) {
				if (option.value === defaultColorPaletteId) {
					option.selected = true;
				}

				if (option.value === "custom_" + colorPaletteId) {
					els.select.removeChild(option);
				}
			}

			setColorPalette(defaultColorPaletteId);
			await loadColorPalette(getColorPalette());
		};

		const loadColorPalette = async (colorPalette) => {
			colorPalette = await completeColorPalette(colorPalette);
			if (colorPalette.colors) {
				// Sets the colors of node slots and links
				if (colorPalette.colors.node_slot) {
					Object.assign(app.canvas.default_connection_color_byType, colorPalette.colors.node_slot);
					Object.assign(LGraphCanvas.link_type_colors, colorPalette.colors.node_slot);
				}
				// Sets the colors of the LiteGraph objects
				if (colorPalette.colors.litegraph_base) {
					// Everything updates correctly in the loop, except the Node Title and Link Color for some reason
					app.canvas.node_title_color = colorPalette.colors.litegraph_base.NODE_TITLE_COLOR;
					app.canvas.default_link_color = colorPalette.colors.litegraph_base.LINK_COLOR;

					for (const key in colorPalette.colors.litegraph_base) {
						if (colorPalette.colors.litegraph_base.hasOwnProperty(key) && LiteGraph.hasOwnProperty(key)) {
							LiteGraph[key] = colorPalette.colors.litegraph_base[key];
						}
					}
				}
				// Sets the color of ComfyUI elements
				if (colorPalette.colors.comfy_base) {
					const rootStyle = document.documentElement.style;
					for (const key in colorPalette.colors.comfy_base) {
					  	rootStyle.setProperty('--' + key, colorPalette.colors.comfy_base[key]);
					}
				}
				app.canvas.draw(true, true);
			}
		};

		const getColorPalette = (colorPaletteId) => {
			if (!colorPaletteId) {
				colorPaletteId = app.ui.settings.getSettingValue(id, defaultColorPaletteId);
			}

			if (colorPaletteId.startsWith("custom_")) {
				colorPaletteId = colorPaletteId.substr(7);
				let customColorPalettes = getCustomColorPalettes();
				if (customColorPalettes[colorPaletteId]) {
					return customColorPalettes[colorPaletteId];
				}
			}

			return colorPalettes[colorPaletteId];
		};

		const setColorPalette = (colorPaletteId) => {
			app.ui.settings.setSettingValue(id, colorPaletteId);
		};

		const fileInput = $el("input", {
			type: "file",
			accept: ".json",
			style: { display: "none" },
			parent: document.body,
			onchange: () => {
				let file = fileInput.files[0];

				if (file.type === "application/json" || file.name.endsWith(".json")) {
					const reader = new FileReader();
					reader.onload = async () => {
						await addCustomColorPalette(JSON.parse(reader.result));
					};
					reader.readAsText(file);
				}
			},
		});

		app.ui.settings.addSetting({
			id,
			name: "Color Palette",
			type: (name, setter, value) => {
				let options = [];

				for (const c in colorPalettes) {
					const colorPalette = colorPalettes[c];
					options.push($el("option", { textContent: colorPalette.name, value: colorPalette.id, selected: colorPalette.id === value }));
				}

				let customColorPalettes = getCustomColorPalettes();
				for (const c in customColorPalettes) {
					const colorPalette = customColorPalettes[c];
					options.push($el("option", { textContent: colorPalette.name + " (custom)", value: "custom_" + colorPalette.id, selected: "custom_" + colorPalette.id === value }));
				}

				return $el("div", [
					$el("label", { textContent: name || id }, [
						els.select = $el("select", {
							onchange: (e) => {
								setter(e.target.value);
							}
						}, options)
					]),
					$el("input", {
						type: "button",
						value: "Export",
						onclick: async () => {
							const colorPaletteId = app.ui.settings.getSettingValue(id, defaultColorPaletteId);
							const colorPalette = await completeColorPalette(getColorPalette(colorPaletteId));
							const json = JSON.stringify(colorPalette, null, 2); // convert the data to a JSON string
							const blob = new Blob([json], { type: "application/json" });
							const url = URL.createObjectURL(blob);
							const a = $el("a", {
								href: url,
								download: colorPaletteId + ".json",
								style: { display: "none" },
								parent: document.body,
							});
							a.click();
							setTimeout(function () {
								a.remove();
								window.URL.revokeObjectURL(url);
							}, 0);
						},
					}),
					$el("input", {
						type: "button",
						value: "Import",
						onclick: () => {
							fileInput.click();
						}
					}),
					$el("input", {
						type: "button",
						value: "Template",
						onclick: async () => {
							const colorPalette = await getColorPaletteTemplate();
							const json = JSON.stringify(colorPalette, null, 2); // convert the data to a JSON string
							const blob = new Blob([json], { type: "application/json" });
							const url = URL.createObjectURL(blob);
							const a = $el("a", {
								href: url,
								download: "color_palette.json",
								style: { display: "none" },
								parent: document.body,
							});
							a.click();
							setTimeout(function () {
								a.remove();
								window.URL.revokeObjectURL(url);
							}, 0);
						}
					}),
					$el("input", {
						type: "button",
						value: "Delete",
						onclick: async () => {
							let colorPaletteId = app.ui.settings.getSettingValue(id, defaultColorPaletteId);

							if (colorPalettes[colorPaletteId]) {
								app.ui.dialog.show("You cannot delete built-in color palette");
								return;
							}

							if (colorPaletteId.startsWith("custom_")) {
								colorPaletteId = colorPaletteId.substr(7);
							}

							await deleteCustomColorPalette(colorPaletteId);
						}
					}),
				]);
			},
			defaultValue: defaultColorPaletteId,
			async onChange(value) {
				if (!value) {
					return;
				}

				if (colorPalettes[value]) {
					await loadColorPalette(colorPalettes[value]);
				} else if (value.startsWith("custom_")) {
					value = value.substr(7);
					let customColorPalettes = getCustomColorPalettes();
					if (customColorPalettes[value]) {
						await loadColorPalette(customColorPalettes[value]);
					}
				}
			},
		});
	},
});
