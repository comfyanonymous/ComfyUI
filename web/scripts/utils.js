import { $el } from "./ui.js";
import { api } from "./api.js";

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


export function clone(obj) {
	try {
		if (typeof structuredClone !== "undefined") {
			return structuredClone(obj);
		}
	} catch (error) {
		// structuredClone is stricter than using JSON.parse/stringify so fallback to that
	}

	return JSON.parse(JSON.stringify(obj));
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

/**
 * @param { string } filename
 * @param { Blob } blob
 */
export function downloadBlob(filename, blob) {
	const url = URL.createObjectURL(blob);
	const a = $el("a", {
		href: url,
		download: filename,
		style: { display: "none" },
		parent: document.body,
	});
	a.click();
	setTimeout(function () {
		a.remove();
		window.URL.revokeObjectURL(url);
	}, 0);
}

/**
 * @template T
 * @param {string} name
 * @param {T} [defaultValue]
 * @param {(currentValue: any, previousValue: any)=>void} [onChanged]
 * @returns {T}
 */
export function prop(target, name, defaultValue, onChanged) {
	let currentValue;
	Object.defineProperty(target, name, {
		get() {
			return currentValue;
		},
		set(newValue) {
			const prevValue = currentValue;
			currentValue = newValue;
			onChanged?.(currentValue, prevValue, target, name);
		},
	});
	return defaultValue;
}

export function getStorageValue(id) {
	const clientId = api.clientId ?? api.initialClientId;
	return (clientId && sessionStorage.getItem(`${id}:${clientId}`)) ?? localStorage.getItem(id);
}

export function setStorageValue(id, value) {
	const clientId = api.clientId ?? api.initialClientId;
	if (clientId) {
		sessionStorage.setItem(`${id}:${clientId}`, value);
	}
	localStorage.setItem(id, value);
}

export class tmpl {
	static #subs = Symbol();

	constructor(template, components) {
		this.template = template;
		this.components = components;
	}

	build = (context) => {
		const wrapper = document.createElement("div");
		wrapper.innerHTML = this.template;

		for (const el of wrapper.children) {
			this.#build(el, context);
		}

		return wrapper.children;
	};

	#build = (el, ctx) => {
		if (this.#loop(el, ctx)) return;

		tmpl.#if(el, ctx);
		tmpl.#text(el, ctx);

		for (const child of el.children) {
			this.#build(child, ctx);
		}
	};

	#loop = (el, ctx) => {
		const attr = el.getAttribute(":each");
		if (attr) {
			const id = el.getAttribute(":tmpl");
			el.removeAttribute(":each");
			el.removeAttribute(":tmpl");

			if (!id || !this.components[id]) {
				console.warn("Missing template: " + id, el);
				return false;
			}

			tmpl.#bind(ctx[attr], (v) => {
				if (v == null) {
					v = [];
				} else if (!(v instanceof Array)) {
					if (typeof v === "object") {
						v = Object.entries(v);
					} else {
						v = [v];
					}
				}

				const res = v.flatMap((item, index, array) => {
					const parent = el.cloneNode(true);
					const children = this.components[id]({ item, index, array }, ctx, this);
					parent.replaceChildren(...children);
					return parent;
				});
				el.replaceWith(...res);
			});
			return true;
		}
		return false;
	};

	static rx(value, onChange, immediate = true) {
		let v = value;
		const subs = [];
		if (onChange) {
			subs.push(onChange);
			if (immediate) {
				onChange(v);
			}
		}
		return {
			get value() {
				return v;
			},
			set value(value) {
				let old = v;
				v = value;
				for (const sub of subs) {
					sub(v, old);
				}
			},
			[tmpl.#subs]: subs,
		};
	}

	static watch(rx, cb) {
		const subs = rx[tmpl.#subs];
		if (!subs) throw new Error("Attempt to watch non-reactive object");
		subs.push(cb);
		return () => subs.splice(subs.indexOf(cb), 1);
	}

	static #isReactive(value) {
		return !!value?.[tmpl.#subs];
	}

	static #unwrap(value) {
		return tmpl.#isReactive(value) ? value.value : value;
	}

	static #bind(value, check) {
		if (tmpl.#isReactive(value)) {
			tmpl.watch(value, check);
			value = tmpl.#unwrap(value);
		}
		check(value);
	}

	static #if(el, ctx) {
		let visibilityAttr = el.getAttribute(":if");
		let negate = false;

		if (visibilityAttr) {
			el.removeAttribute(":if");
		} else {
			visibilityAttr = el.getAttribute(":if!");
			if (visibilityAttr) {
				el.removeAttribute(":if!");
				negate = true;
			}
		}

		if (visibilityAttr) {
			el.removeAttribute(visibilityAttr);
			tmpl.#bind(ctx[visibilityAttr], (v) => {
				if (negate) {
					v = !v;
				}
				if (v) {
					el.style.display = "none";
				} else {
					el.style.removeProperty("display");
				}
			});
		}
	}

	static #text(el, ctx) {
		const attr = el.getAttribute(":text");
		if (attr) {
			el.removeAttribute(":text");
			tmpl.#bind(ctx[attr], (v) => (el.textContent = v));
		}
	}
}