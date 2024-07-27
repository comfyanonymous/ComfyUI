/**
 * @typedef {  string | string[] | Record<string, boolean> } ClassList
 */

/**
 * @param { HTMLElement } element
 * @param { ClassList } classList
 * @param { string[] } requiredClasses
 */
export function applyClasses(element, classList, ...requiredClasses) {
	classList ??= "";

	let str;
	if (typeof classList === "string") {
		str = classList;
	} else if (classList instanceof Array) {
		str = classList.join(" ");
	} else {
		str = Object.entries(classList).reduce((p, c) => {
			if (c[1]) {
				p += (p.length ? " " : "") + c[0];
			}
			return p;
		}, "");
	}
	element.className = str;
	if (requiredClasses) {
		element.classList.add(...requiredClasses);
	}
}

/**
 * @param { HTMLElement } element
 * @param { { onHide?: (el: HTMLElement) => void, onShow?: (el: HTMLElement, value) => void } } [param1]
 * @returns
 */
export function toggleElement(element, { onHide, onShow } = {}) {
	let placeholder;
	let hidden;
	return (value) => {
		if (value) {
			if (hidden) {
				hidden = false;
				placeholder.replaceWith(element);
			}
			onShow?.(element, value);
		} else {
			if (!placeholder) {
				placeholder = document.createComment("");
			}
			hidden = true;
			element.replaceWith(placeholder);
			onHide?.(element);
		}
	};
}
