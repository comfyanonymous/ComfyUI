import { app } from "../scripts/app.js";

const ext = {
	// Unique name for the extension
	name: "Example.LoggingExtension",
	async init(app) {

	},
	async setup(app) {
		app.canvasEl.addEventListener("click", (e)=> {
			var node = app.graph.getNodeOnPos( e.clientX, e.clientY, app.graph._nodes, 5 );
			console.log("clicked node", node.type);
			window.parent.postMessage({ type: "onClickNodeEvent", nodeType: node.type }, window.location.origin);
		});
		app.canvasEl.addEventListener("mousemove", (e)=> {
			var node = app.graph.getNodeOnPos( e.clientX, e.clientY, app.graph._nodes, 5 );		
			if(node) {
				app.canvasEl.style.cursor = "pointer";
			} else {
				app.canvasEl.style.cursor = "default";
			}
		});

	},
};

app.registerExtension(ext);
