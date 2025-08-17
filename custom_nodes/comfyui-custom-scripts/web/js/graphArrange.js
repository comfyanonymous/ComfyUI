import { app } from "../../../scripts/app.js";

app.registerExtension({
	name: "pysssss.GraphArrange",
	setup(app) {
		const orig = LGraphCanvas.prototype.getCanvasMenuOptions;
		LGraphCanvas.prototype.getCanvasMenuOptions = function () {
			const options = orig.apply(this, arguments);
			options.push({ content: "Arrange (float left)", callback: () => graph.arrange() });
			options.push({
				content: "Arrange (float right)",
				callback: () => {
					(function () {
						var margin = 50;
						var layout;

						const nodes = this.computeExecutionOrder(false, true);
						const columns = [];

						// Find node first use
						for (let i = nodes.length - 1; i >= 0; i--) {
							const node = nodes[i];
							let max = null;
							for (const out of node.outputs || []) {
								if (out.links) {
									for (const link of out.links) {
										const outNode = app.graph.getNodeById(app.graph.links[link].target_id);
										if (!outNode) continue;
										var l = outNode._level - 1;
										if (max === null) max = l;
										else if (l < max) max = l;
									}
								}
							}
							if (max != null) node._level = max;
						}

						for (let i = 0; i < nodes.length; ++i) {
							const node = nodes[i];
							const col = node._level || 1;
							if (!columns[col]) {
								columns[col] = [];
							}
							columns[col].push(node);
						}

						let x = margin;

						for (let i = 0; i < columns.length; ++i) {
							const column = columns[i];
							if (!column) {
								continue;
							}
							column.sort((a, b) => {
								var as = !(a.type === "SaveImage" || a.type === "PreviewImage");
								var bs = !(b.type === "SaveImage" || b.type === "PreviewImage");
								var r = as - bs;
								if (r === 0) r = (a.inputs?.length || 0) - (b.inputs?.length || 0);
								if (r === 0) r = (a.outputs?.length || 0) - (b.outputs?.length || 0);
								return r;
							});
							let max_size = 100;
							let y = margin + LiteGraph.NODE_TITLE_HEIGHT;
							for (let j = 0; j < column.length; ++j) {
								const node = column[j];
								node.pos[0] = layout == LiteGraph.VERTICAL_LAYOUT ? y : x;
								node.pos[1] = layout == LiteGraph.VERTICAL_LAYOUT ? x : y;
								const max_size_index = layout == LiteGraph.VERTICAL_LAYOUT ? 1 : 0;
								if (node.size[max_size_index] > max_size) {
									max_size = node.size[max_size_index];
								}
								const node_size_index = layout == LiteGraph.VERTICAL_LAYOUT ? 0 : 1;
								y += node.size[node_size_index] + margin + LiteGraph.NODE_TITLE_HEIGHT + j;
							}

							// Right align in column
							for (let j = 0; j < column.length; ++j) {
								const node = column[j];
								node.pos[0] += max_size - node.size[0];
							}
							x += max_size + margin;
						}

						this.setDirtyCanvas(true, true);
					}).apply(app.graph);
				},
			});
			return options;
		};
	},
});
