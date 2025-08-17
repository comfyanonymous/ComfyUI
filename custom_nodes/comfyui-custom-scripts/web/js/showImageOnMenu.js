import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { $el } from "../../../scripts/ui.js";

const id = "pysssss.ShowImageOnMenu";
const ext = {
	name: id,
	async setup(app) {
		let enabled = true;
		let nodeId = null;
		const img = $el("img", {
			style: {
				width: "100%",
				height: "150px",
				objectFit: "contain",
			},
		});
		const link = $el(
			"a",
			{
				style: {
					width: "100%",
					height: "150px",
					marginTop: "10px",
					order: 100, // Place this item last (until someone else has a higher order)
					display: "none",
				},
				href: "#",
				onclick: (e) => {
					e.stopPropagation();
					e.preventDefault();
					const node = app.graph.getNodeById(nodeId);
					if (!node) return;
					app.canvas.centerOnNode(node);
					app.canvas.setZoom(1);
				},
			},
			[img]
		);

		app.ui.menuContainer.append(link);

		const show = (src, node) => {
			img.src = src;
			nodeId = Number(node);
			link.style.display = "unset";
		};

		api.addEventListener("executed", ({ detail }) => {
			if (!enabled) return;
			const images = detail?.output?.images;
			if (!images || !images.length) return;
			const format = app.getPreviewFormatParam();
			const src = [
				`./view?filename=${encodeURIComponent(images[0].filename)}`,
				`type=${images[0].type}`,
				`subfolder=${encodeURIComponent(images[0].subfolder)}`,
				`t=${+new Date()}${format}`,].join('&');
			show(src, detail.node);
		});

		api.addEventListener("b_preview", ({ detail }) => {
			if (!enabled) return;
			show(URL.createObjectURL(detail), app.runningNodeId);
		});

		app.ui.settings.addSetting({
			id,
			name: "🐍 Show Image On Menu",
			defaultValue: true,
			type: "boolean",
			onChange(value) {
				enabled = value;

				if (!enabled) link.style.display = "none";
			},
		});
	},
};

app.registerExtension(ext);
