import { $el } from "./ui.js";

export async function addStylesheet(urlOrFile, relativeTo) {
	return new Promise((res, rej) => {
		let url;
		if (urlOrFile.endsWith(".js")) {
			url = urlOrFile.substr(0, urlOrFile.length - 2) + "css";
		} else {
			url = new URL(urlOrFile, relativeTo ?? `${window.location.protocol}//${window.location.host}`).toString();
		}
		$el("link", {
			parent: document.head,
			rel: "stylesheet",
			type: "text/css",
			href: url,
			onload: res,
			onerror: rej,
		});
	});
}
