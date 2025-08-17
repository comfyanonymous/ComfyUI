import { app } from "../../../scripts/app.js";

// Adds context menu entries, code partly from pyssssscustom-scripts

function addMenuHandler(nodeType, cb) {
	const getOpts = nodeType.prototype.getExtraMenuOptions;
	nodeType.prototype.getExtraMenuOptions = function () {
		const r = getOpts.apply(this, arguments);
		cb.apply(this, arguments);
		return r;
	};
}

function addNode(name, nextTo, options) {
	console.log("name:", name);
	console.log("nextTo:", nextTo);
	options = { side: "left", select: true, shiftY: 0, shiftX: 0, ...(options || {}) };
	const node = LiteGraph.createNode(name);
	app.graph.add(node);
	
	node.pos = [
		options.side === "left" ? nextTo.pos[0] - (node.size[0] + options.offset): nextTo.pos[0] + nextTo.size[0] + options.offset,
		
		nextTo.pos[1] + options.shiftY,
	];
	if (options.select) {
		app.canvas.selectNode(node, false);
	}
	return node;
}

app.registerExtension({
	name: "KJNodesContextmenu",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.input && nodeData.input.required) {
			addMenuHandler(nodeType, function (_, options) {
				options.unshift(
					{
					content: "Add GetNode",
					callback: () => {addNode("GetNode", this, { side:"left", offset: 30});}
					},
					{
					content: "Add SetNode",
					callback: () => {addNode("SetNode", this, { side:"right", offset: 30 });
					},
				});
			});
		}
	},
		async setup(app) {
			const updateSlots = (value) => {
				const valuesToAddToIn = ["GetNode"];
				const valuesToAddToOut = ["SetNode"];
				// Remove entries if they exist
				for (const arr of Object.values(LiteGraph.slot_types_default_in)) {
					for (const valueToAdd of valuesToAddToIn) {
						const idx = arr.indexOf(valueToAdd);
						if (idx !== -1) {
							arr.splice(idx, 1);
						}
					}
				}
			
				for (const arr of Object.values(LiteGraph.slot_types_default_out)) {
					for (const valueToAdd of valuesToAddToOut) {
						const idx = arr.indexOf(valueToAdd);
						if (idx !== -1) {
							arr.splice(idx, 1);
						}
					}
				}
				if (value!="disabled") {
					for (const arr of Object.values(LiteGraph.slot_types_default_in)) {
						for (const valueToAdd of valuesToAddToIn) {
							const idx = arr.indexOf(valueToAdd);
							if (idx !== -1) {
								arr.splice(idx, 1);
							}
							if (value === "top") {
								arr.unshift(valueToAdd);
							} else {
								arr.push(valueToAdd);
							}
						}
					}
			
					for (const arr of Object.values(LiteGraph.slot_types_default_out)) {
						for (const valueToAdd of valuesToAddToOut) {
							const idx = arr.indexOf(valueToAdd);
							if (idx !== -1) {
								arr.splice(idx, 1);
							}
							if (value === "top") {
								arr.unshift(valueToAdd);
							} else {
								arr.push(valueToAdd);
							}
						}
					}
				}
			};
			
			app.ui.settings.addSetting({
				id: "KJNodes.SetGetMenu",
				name: "KJNodes: Make Set/Get -nodes defaults",
				tooltip: 'Adds Set/Get nodes to the top or bottom of the list of available node suggestions.',
				options: ['disabled', 'top', 'bottom'],
				defaultValue: 'disabled',
				type: "combo",
				onChange: updateSlots,
				
			});
			app.ui.settings.addSetting({
				id: "KJNodes.MiddleClickDefault",
				name: "KJNodes: Middle click default node adding",
				defaultValue: false,
				type: "boolean",
				onChange: (value) => {
					LiteGraph.middle_click_slot_add_default_node = value;
				},
			});
			app.ui.settings.addSetting({
				id: "KJNodes.nodeAutoColor",
				name: "KJNodes: Automatically set node colors",
				type: "boolean",
				defaultValue: true,
			});
			app.ui.settings.addSetting({
				id: "KJNodes.helpPopup",
				name: "KJNodes: Help popups",
				defaultValue: true,
				type: "boolean",
			});
			app.ui.settings.addSetting({
				id: "KJNodes.disablePrefix",
				name: "KJNodes: Disable automatic Set_ and Get_ prefix",
				defaultValue: true,
				type: "boolean",
			});
			app.ui.settings.addSetting({
				id: "KJNodes.browserStatus",
				name: "KJNodes: 🟢 Stoplight browser status icon 🔴",
				defaultValue: false,
				type: "boolean",
			});
}
});
