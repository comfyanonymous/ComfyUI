import { $el } from "./ui.js";

export function needLoadPrebuiltWorkflow(workflowId) {
	var loaded = localStorage.getItem('PrebuiltWorkflowId' + workflowId);
	if (loaded) {
		return false
	} else {
		localStorage.setItem('PrebuiltWorkflowId' + workflowId, true);
		return true
	}
}

export async function getWorkflow() {
	let flow_json = null;
	const queryString = window.location.search;
	const urlParams = new URLSearchParams(queryString);
	const workflowId = urlParams.get('workflow');
	if (workflowId && needLoadPrebuiltWorkflow(workflowId)) {
		await fetch('../workflows/' + workflowId + '/' + workflowId + '.json').then(
			response => {
				flow_json = response.json()
			}
		)
	}
	return flow_json;
}

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


function setCookie(name, value, days) {
	var expires = "";
	if (days) {
		var date = new Date();
		date.setTime(date.getTime() + (days * 24 * 60 * 60 * 1000));
		expires = "; expires=" + date.toUTCString();
	}
	document.cookie = name + "=" + (value || "") + expires + "; path=/";
}

function getCookie(name) {
	var nameEQ = name + "=";
	var ca = document.cookie.split(';');
	for (var i = 0; i < ca.length; i++) {
		var c = ca[i];
		while (c.charAt(0) == ' ') c = c.substring(1, c.length);
		if (c.indexOf(nameEQ) == 0) return c.substring(nameEQ.length, c.length);
	}
	return null;
}

export async function getWorkflow() {
	let flow_json = null;
	const queryString = window.location.search;
	const urlParams = new URLSearchParams(queryString);
	const workflowId = urlParams.get('workflow');
	if (workflowId){
		await fetch('../workflows/' + workflowId + '/' + workflowId + '.json').then(
			response => {
				flow_json = response.json()
			}
		)
	} 
	return flow_json;
}

export function getUserId() {
	var uid = getCookie('uid');
	if (uid == null) {
		const queryString = window.location.search;
		const urlParams = new URLSearchParams(queryString);
		const email = urlParams.get('email');
		uid = prompt("Please enter your nickname \n(less than ten letters)", email ? email.split("@")[0] : "anonymous");
		setCookie('uid', uid, 999);
	}
	return uid ? uid : "anonymous";
}
