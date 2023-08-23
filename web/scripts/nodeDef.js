import { ComfyWidgets } from "./widgets.js"
import { range } from "./utils.js"

export function isBackendNodeDefInputType(inputName, type) {
	const widgets = Object.assign({}, ComfyWidgets, ComfyWidgets.customWidgets);
    return !Array.isArray(type) && !(type in ComfyWidgets) && !(`${type}:${inputName}` in ComfyWidgets);
}

export function iterateNodeDefInputs(def) {
    var inputs = def.input.required || {}
    if (def.input.optional != null) {
        inputs = Object.assign({}, def.input.required, def.input.optional)
    }
    return Object.entries(inputs);
}

export function iterateNodeDefOutputs(def) {
    const outputCount = def.output ? def.output.length : 0;
    return range(outputCount).map(i => {
        return {
            type: def.output[i],
            name: def.output_name[i] || def.output[i],
            is_list: def.output_is_list[i],
        }
    })
}
