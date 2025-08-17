import { api } from "../../scripts/api.js";
import { app } from "../../scripts/app.js";

let original_show = app.ui.dialog.show;

export function customAlert(message) {
	try {
		app.extensionManager.toast.addAlert(message);
	}
	catch {
		alert(message);
	}
}

export function isBeforeFrontendVersion(compareVersion) {
    try {
        const frontendVersion = window['__COMFYUI_FRONTEND_VERSION__'];
        if (typeof frontendVersion !== 'string') {
            return false;
        }

        function parseVersion(versionString) {
            const parts = versionString.split('.').map(Number);
            return parts.length === 3 && parts.every(part => !isNaN(part)) ? parts : null;
        }

        const currentVersion = parseVersion(frontendVersion);
        const comparisonVersion = parseVersion(compareVersion);

        if (!currentVersion || !comparisonVersion) {
            return false;
        }

        for (let i = 0; i < 3; i++) {
            if (currentVersion[i] > comparisonVersion[i]) {
                return false;
            } else if (currentVersion[i] < comparisonVersion[i]) {
                return true;
            }
        }

        return false;
    } catch {
        return true;
    }
}

function dialog_show_wrapper(html) {
	if (typeof html === "string") {
		if(html.includes("IMPACT-PACK-SIGNAL: STOP CONTROL BRIDGE")) {
			return;
		}

		this.textElement.innerHTML = html;
	} else {
		this.textElement.replaceChildren(html);
	}
	this.element.style.display = "flex";
}

app.ui.dialog.show = dialog_show_wrapper;


function nodeFeedbackHandler(event) {
	let nodes = app.graph._nodes_by_id;
	let node = nodes[event.detail.node_id];
	if(node) {
		const w = node.widgets.find((w) => event.detail.widget_name === w.name);
		if(w) {
			w.value = event.detail.value;
		}
	}
}

api.addEventListener("impact-node-feedback", nodeFeedbackHandler);


function setMuteState(event) {
	let nodes = app.graph._nodes_by_id;
	let node = nodes[event.detail.node_id];
	if(node) {
		if(event.detail.is_active)
			node.mode = 0;
		else
			node.mode = 2;
	}
}

api.addEventListener("impact-node-mute-state", setMuteState);


async function bridgeContinue(event) {
	let nodes = app.graph._nodes_by_id;
	let node = nodes[event.detail.node_id];
	if(node) {
		const mutes = new Set(event.detail.mutes);
		const actives = new Set(event.detail.actives);
		const bypasses = new Set(event.detail.bypasses);

		for(let i in app.graph._nodes_by_id) {
			let this_node = app.graph._nodes_by_id[i];
			if(mutes.has(i)) {
				this_node.mode = 2;
			}
			else if(actives.has(i)) {
				this_node.mode = 0;
			}
			else if(bypasses.has(i)) {
				this_node.mode = 4;
			}
		}

		await app.queuePrompt(0, 1);
	}
}

api.addEventListener("impact-bridge-continue", bridgeContinue);


function addQueue(event) {
	app.queuePrompt(0, 1);
}

api.addEventListener("impact-add-queue", addQueue);


function refreshPreview(event) {
	let node_id = event.detail.node_id;
	let item = event.detail.item;
	let img = new Image();
	img.src = `/view?filename=${item.filename}&subfolder=${item.subfolder}&type=${item.type}&no-cache=${Date.now()}`;
	let node = app.graph._nodes_by_id[node_id];
	if(node)
		node.imgs = [img];
}

api.addEventListener("impact-preview", refreshPreview);
