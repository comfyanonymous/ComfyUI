import { app } from "/scripts/app.js";
import { $el } from "/scripts/ui.js";
import { api } from "/scripts/api.js";

// Manage color palettes

const colorPalettes = {
	"palette_1": {
		"id": "palette_1",
		"name": "Palette 1",
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
		},
	},
	"solarized": {
		"id": "solarized",
		"name": "Solarized",
		"colors": {
			"node_slot": {
				"CLIP": "#859900", // Green
				"CLIP_VISION": "#6c71c4", // Indigo
				"CLIP_VISION_OUTPUT": "#859900", // Green
				"CONDITIONING": "#d33682", // Magenta
				"CONTROL_NET": "#cb4b16", // Orange
				"IMAGE": "#dc322f", // Red
				"LATENT": "#268bd2", // Blue
				"MASK": "#073642", // Base02
				"MODEL": "#cb4b16", // Orange
				"STYLE_MODEL": "#073642", // Base02
				"UPSCALE_MODEL": "#6c71c4", // Indigo
				"VAE": "#586e75", // Base1
			},
			"litegraph_base": {
				"NODE_TITLE_COLOR": "#fdf6e3",
				"NODE_SELECTED_TITLE_COLOR": "#b58900",
				"NODE_TEXT_SIZE": 14,
				"NODE_TEXT_COLOR": "#657b83",
				"NODE_SUBTEXT_SIZE": 12,
				"NODE_DEFAULT_COLOR": "#586e75",
				"NODE_DEFAULT_BGCOLOR": "#073642",
				"NODE_DEFAULT_BOXCOLOR": "#839496",
				"NODE_DEFAULT_SHAPE": "box",
				"NODE_BOX_OUTLINE_COLOR": "#fdf6e3",
				"DEFAULT_SHADOW_COLOR": "rgba(0,0,0,0.5)",
				"DEFAULT_GROUP_FONT": 24,

				"WIDGET_BGCOLOR": "#002b36",
				"WIDGET_OUTLINE_COLOR": "#839496",
				"WIDGET_TEXT_COLOR": "#fdf6e3",
				"WIDGET_SECONDARY_TEXT_COLOR": "#93a1a1",

				"LINK_COLOR": "#2aa198",
				"EVENT_LINK_COLOR": "#268bd2",
				"CONNECTING_LINK_COLOR": "#859900",
			},
		},
	}
};

const id = "Comfy.ColorPalette";
const idCustomColorPalettes = "Comfy.CustomColorPalettes";
const defaultColorPaletteId = "palette_1";
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
					}
				}
			};

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
				if (colorPalette.colors.node_slot) {
					Object.assign(app.canvas.default_connection_color_byType, colorPalette.colors.node_slot);
				}
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
				customizeRenderLink(colorPalette);
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

function customizeRenderLink(colorPalette) {
    var LGraphCanvas = LiteGraph.LGraphCanvas;

    function getLinkColor(link, inputNode, outputNode, colorPalette) {
        let color = null;
        if (link && link.color) {
            color = link.color;
        } else if (link) {
            const matchingEntry = inputNode.outputs.find((output) => {
                return outputNode.inputs.some((input) => input.type === output.type);
            });

            if (matchingEntry) {
                let nodeType = matchingEntry.type;
                color = colorPalette.colors.node_slot[nodeType];
            }
        }
        return color;
    }

    var originalRenderLink = LGraphCanvas.prototype.renderLink;

    LGraphCanvas.prototype.renderLink = function(
        ctx,
        a,
        b,
        link,
        skip_border,
        flow,
        color,
        start_dir,
        end_dir,
        num_sublines
    ) {
        if (link) {
            const inputNode = this.graph.getNodeById(link.origin_id);
            const outputNode = this.graph.getNodeById(link.target_id);
            color = getLinkColor(link, inputNode, outputNode, colorPalette);
        }

        originalRenderLink.call(
            this,
            ctx,
            a,
            b,
            link,
            skip_border,
            flow,
            color,
            start_dir,
            end_dir,
            num_sublines
        );
    };
}