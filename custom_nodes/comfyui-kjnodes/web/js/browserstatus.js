import { api } from "../../../scripts/api.js";
import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "KJNodes.browserstatus",
    setup() {
        if (!app.ui.settings.getSettingValue("KJNodes.browserStatus")) {
            return;
        }
        api.addEventListener("status", ({ detail }) => {
            let title = "ComfyUI";
            let favicon = "green";
            let queueRemaining = detail && detail.exec_info.queue_remaining;

            if (queueRemaining) {
                favicon = "red";
                title = `00% - ${queueRemaining} | ${title}`;
            } 
            let link = document.querySelector("link[rel~='icon']");
            if (!link) {
                link = document.createElement("link");
                link.rel = "icon";
                document.head.appendChild(link);
            }
            link.href = new URL(`../${favicon}.png`, import.meta.url);
            document.title = title;
        });
        //add progress to the title
        api.addEventListener("progress", ({ detail }) => {
			const { value, max } = detail;
			const progress = Math.floor((value / max) * 100);
			let title = document.title;
		
			if (!isNaN(progress) && progress >= 0 && progress <= 100) {
				const paddedProgress = String(progress).padStart(2, '0');
				title = `${paddedProgress}% ${title.replace(/^\d+%\s/, '')}`;
			}
			document.title = title;
		});
    },
    init() {
        if (!app.ui.settings.getSettingValue("KJNodes.browserStatus")) {
            return;
        }
        const pythongossFeed = app.extensions.find(
            (e) => e.name === 'pysssss.FaviconStatus',
          )
          if (pythongossFeed) {
            console.warn("KJNodes - Overriding pysssss.FaviconStatus")
            pythongossFeed.setup = function() {
                console.warn("Disabled by KJNodes")
            };
          }
    },
});