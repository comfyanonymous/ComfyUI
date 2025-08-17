import { app } from "../../../scripts/app.js";

const notificationSetup = () => {
	if (!("Notification" in window)) {
		console.log("This browser does not support notifications.");
		alert("This browser does not support notifications.");
		return;
	}
	if (Notification.permission === "denied") {
		console.log("Notifications are blocked. Please enable them in your browser settings.");
		alert("Notifications are blocked. Please enable them in your browser settings.");
		return;
	}
	if (Notification.permission !== "granted") {
		Notification.requestPermission();
	}
	return true;
};

app.registerExtension({
	name: "pysssss.SystemNotification",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "SystemNotification|pysssss") {
			const onExecuted = nodeType.prototype.onExecuted;
			nodeType.prototype.onExecuted = async function ({ message, mode }) {
				onExecuted?.apply(this, arguments);

				if (mode === "on empty queue") {
					if (app.ui.lastQueueSize !== 0) {
						await new Promise((r) => setTimeout(r, 500));
					}
					if (app.ui.lastQueueSize !== 0) {
						return;
					}
				}
				if (!notificationSetup()) return;
				const notification = new Notification("ComfyUI", { body: message ?? "Your notification has triggered." });
			};

			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				onNodeCreated?.apply(this, arguments);
				notificationSetup();
			};
		}
	},
});
