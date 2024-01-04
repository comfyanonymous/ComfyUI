import { $el } from "./ui.js";

// Simple date formatter
const parts = {
	d: (d) => d.getDate(),
	M: (d) => d.getMonth() + 1,
	h: (d) => d.getHours(),
	m: (d) => d.getMinutes(),
	s: (d) => d.getSeconds(),
};
const format =
	Object.keys(parts)
		.map((k) => k + k + "?")
		.join("|") + "|yyy?y?";

function formatDate(text, date) {
	return text.replace(new RegExp(format, "g"), function (text) {
		if (text === "yy") return (date.getFullYear() + "").substring(2);
		if (text === "yyyy") return date.getFullYear();
		if (text[0] in parts) {
			const p = parts[text[0]](date);
			return (p + "").padStart(text.length, "0");
		}
		return text;
	});
}

export function applyTextReplacements(app, value) {
	return value.replace(/%([^%]+)%/g, function (match, text) {
		const split = text.split(".");
		if (split.length !== 2) {
			// Special handling for dates
			if (split[0].startsWith("date:")) {
				return formatDate(split[0].substring(5), new Date());
			}

			if (text !== "width" && text !== "height") {
				// Dont warn on standard replacements
				console.warn("Invalid replacement pattern", text);
			}
			return match;
		}

		// Find node with matching S&R property name
		let nodes = app.graph._nodes.filter((n) => n.properties?.["Node name for S&R"] === split[0]);
		// If we cant, see if there is a node with that title
		if (!nodes.length) {
			nodes = app.graph._nodes.filter((n) => n.title === split[0]);
		}
		if (!nodes.length) {
			console.warn("Unable to find node", split[0]);
			return match;
		}

		if (nodes.length > 1) {
			console.warn("Multiple nodes matched", split[0], "using first match");
		}

		const node = nodes[0];

		const widget = node.widgets?.find((w) => w.name === split[1]);
		if (!widget) {
			console.warn("Unable to find widget", split[1], "on node", split[0], node);
			return match;
		}

		return ((widget.value ?? "") + "").replaceAll(/\/|\\/g, "_");
	});
}

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
