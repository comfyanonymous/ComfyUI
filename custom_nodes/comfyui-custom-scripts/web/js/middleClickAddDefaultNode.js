import { app } from "../../../scripts/app.js";

const id = "pysssss.MiddleClickAddDefaultNode";
const ext = {
	name: id,
	async setup(app) {
		app.ui.settings.addSetting({
			id,
			name: "🐍 Middle click slot to add",
			defaultValue: "Reroute",
			type: "combo",
			options: (value) =>
				[
					...Object.keys(LiteGraph.registered_node_types)
						.filter((k) => k.includes("Reroute"))
						.sort((a, b) => {
							if (a === "Reroute") return -1;
							if (b === "Reroute") return 1;
							return a.localeCompare(b);
						}),
					"[None]",
				].map((m) => ({
					value: m,
					text: m,
					selected: !value ? m === "[None]" : m === value,
				})),
			onChange(value) {
				const enable = value && value !== "[None]";
				if (value === true) {
					value = "Reroute";
				}
				LiteGraph.middle_click_slot_add_default_node = enable;
				if (enable) {
					for (const arr of Object.values(LiteGraph.slot_types_default_in).concat(
						Object.values(LiteGraph.slot_types_default_out)
					)) {
						const idx = arr.indexOf(value);
						if (idx !== 0) {
							arr.splice(idx, 1);
						}
						arr.unshift(value);
					}
				}
			},
		});
	},
};

app.registerExtension(ext);
