import { api } from "../../../scripts/api.js";
import { app } from "../../../scripts/app.js";

// Simple script that adds the current queue size to the window title
// Adds a favicon that changes color while active

app.registerExtension({
	name: "pysssss.FaviconStatus",
	async setup() {
		let link = document.querySelector("link[rel~='icon']");
		if (!link) {
			link = document.createElement("link");
			link.rel = "icon";
			document.head.appendChild(link);
		}

		const getUrl = (active, user) => new URL(`assets/favicon${active ? "-active" : ""}${user ? ".user" : ""}.ico`, import.meta.url);
		const testUrl = async (active) => {
			const url = getUrl(active, true);
			const r = await fetch(url, {
				method: "HEAD",
			});
			if (r.status === 200) {
				return url;
			}
			return getUrl(active, false);
		};
		const activeUrl = await testUrl(true);
		const idleUrl = await testUrl(false);

		let executing = false;
		const update = () => (link.href = executing ? activeUrl : idleUrl);

		for (const e of ["execution_start", "progress"]) {
			api.addEventListener(e, () => {
				executing = true;
				update();
			});
		}

		api.addEventListener("executing", ({ detail }) => {
			// null will be sent when it's finished
			executing = !!detail;
			update();
		});

		api.addEventListener("status", ({ detail }) => {
			let title = "ComfyUI";
			if (detail && detail.exec_info.queue_remaining) {
				title = `(${detail.exec_info.queue_remaining}) ${title}`;
			}
			document.title = title;
			update();
			executing = false;
		});
		update();
	},
});
