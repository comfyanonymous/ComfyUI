import { addStylesheet } from "./utils.js";

addStylesheet(import.meta.url);

export function createSpinner() {
	const div = document.createElement("div");
	div.innerHTML = `<div class="pysssss-lds-ring"><div></div><div></div><div></div><div></div></div>`;
	return div.firstElementChild;
}
