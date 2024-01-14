import { $el } from "../ui.js";

/**
 * @typedef { { text: string, value?: string, tooltip?: string } } ToggleSwitchItem
 */
/**
 * Creates a toggle switch element
 * @param { string } name
 * @param { Array<string | ToggleSwitchItem } items
 * @param { Object } [opts]
 * @param { (e: { item: ToggleSwitchItem, prev?: ToggleSwitchItem }) => void } [opts.onChange]
 */
export function toggleSwitch(name, items, { onChange } = {}) {
	let selectedIndex;
	let elements;
	
	function updateSelected(index) {
		if (selectedIndex != null) {
			elements[selectedIndex].classList.remove("comfy-toggle-selected");
		}
		onChange?.({ item: items[index], prev: selectedIndex == null ? undefined : items[selectedIndex] });
		selectedIndex = index;
		elements[selectedIndex].classList.add("comfy-toggle-selected");
	}

	elements = items.map((item, i) => {
		if (typeof item === "string") item = { text: item };
		if (!item.value) item.value = item.text;

		const toggle = $el(
			"label",
			{
				textContent: item.text,
				title: item.tooltip ?? "",
			},
			$el("input", {
				name,
				type: "radio",
				value: item.value ?? item.text,
				checked: item.selected,
				onchange: () => {
					updateSelected(i);
				},
			})
		);
		if (item.selected) {
			updateSelected(i);
		}
		return toggle;
	});

	const container = $el("div.comfy-toggle-switch", elements);

	if (selectedIndex == null) {
		elements[0].children[0].checked = true;
		updateSelected(0);
	}

	return container;
}
