import { app } from "../../../scripts/app.js";
import { $el } from "../../../scripts/ui.js";

const colorShade = (col, amt) => {
	col = col.replace(/^#/, "");
	if (col.length === 3) col = col[0] + col[0] + col[1] + col[1] + col[2] + col[2];

	let [r, g, b] = col.match(/.{2}/g);
	[r, g, b] = [parseInt(r, 16) + amt, parseInt(g, 16) + amt, parseInt(b, 16) + amt];

	r = Math.max(Math.min(255, r), 0).toString(16);
	g = Math.max(Math.min(255, g), 0).toString(16);
	b = Math.max(Math.min(255, b), 0).toString(16);

	const rr = (r.length < 2 ? "0" : "") + r;
	const gg = (g.length < 2 ? "0" : "") + g;
	const bb = (b.length < 2 ? "0" : "") + b;

	return `#${rr}${gg}${bb}`;
};

app.registerExtension({
	name: "pysssss.CustomColors",
	setup() {
		let picker;
		let activeNode;
		const onMenuNodeColors = LGraphCanvas.onMenuNodeColors;
		LGraphCanvas.onMenuNodeColors = function (value, options, e, menu, node) {
			const r = onMenuNodeColors.apply(this, arguments);
			requestAnimationFrame(() => {
				const menus = document.querySelectorAll(".litecontextmenu");
				for (let i = menus.length - 1; i >= 0; i--) {
					if (menus[i].firstElementChild.textContent.includes("No color") || menus[i].firstElementChild.value?.content?.includes("No color")) {
						$el(
							"div.litemenu-entry.submenu",
							{
								parent: menus[i],
								$: (el) => {
									el.onclick = () => {
										LiteGraph.closeAllContextMenus();
										if (!picker) {
											picker = $el("input", {
												type: "color",
												parent: document.body,
												style: {
													display: "none",
												},
											});
											picker.onchange = () => {
												if (activeNode) {
													const fApplyColor = function(node){
														if (picker.value) {
															if (node.constructor === LiteGraph.LGraphGroup) {
																node.color = picker.value;
															} else {
																node.color = colorShade(picker.value, 20);
																node.bgcolor = picker.value;
															}
														}
													}
													const graphcanvas = LGraphCanvas.active_canvas;
													if (!graphcanvas.selected_nodes || Object.keys(graphcanvas.selected_nodes).length <= 1){
														fApplyColor(activeNode);
													} else {
														for (let i in graphcanvas.selected_nodes) {
															fApplyColor(graphcanvas.selected_nodes[i]);
														}
													}

													activeNode.setDirtyCanvas(true, true);
												}
											};
										}
										activeNode = null;
										picker.value = node.bgcolor;
										activeNode = node;
										picker.click();
									};
								},
							},
							[
								$el("span", {
									style: {
										paddingLeft: "4px",
										display: "block",
									},
									textContent: "🎨 Custom",
								}),
							]
						);
						break;
					}
				}
			});
			return r;
		};
	},
});
