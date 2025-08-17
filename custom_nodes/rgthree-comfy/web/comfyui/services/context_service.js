import { getConnectedOutputNodesAndFilterPassThroughs } from "../utils.js";
export let SERVICE;
const OWNED_PREFIX = "+";
const REGEX_PREFIX = /^[\+⚠️]\s*/;
const REGEX_EMPTY_INPUT = /^\+\s*$/;
export function stripContextInputPrefixes(name) {
    return name.replace(REGEX_PREFIX, "");
}
export function getContextOutputName(inputName) {
    if (inputName === "base_ctx")
        return "CONTEXT";
    return stripContextInputPrefixes(inputName).toUpperCase();
}
export var InputMutationOperation;
(function (InputMutationOperation) {
    InputMutationOperation[InputMutationOperation["UNKNOWN"] = 0] = "UNKNOWN";
    InputMutationOperation[InputMutationOperation["ADDED"] = 1] = "ADDED";
    InputMutationOperation[InputMutationOperation["REMOVED"] = 2] = "REMOVED";
    InputMutationOperation[InputMutationOperation["RENAMED"] = 3] = "RENAMED";
})(InputMutationOperation || (InputMutationOperation = {}));
export class ContextService {
    constructor() {
        if (SERVICE) {
            throw new Error("ContextService was already instantiated.");
        }
    }
    onInputChanges(node, mutation) {
        const childCtxs = getConnectedOutputNodesAndFilterPassThroughs(node, node, 0);
        for (const childCtx of childCtxs) {
            childCtx.handleUpstreamMutation(mutation);
        }
    }
    getDynamicContextInputsData(node) {
        return node
            .getContextInputsList()
            .map((input, index) => ({
            name: stripContextInputPrefixes(input.name),
            type: String(input.type),
            index,
        }))
            .filter((i) => i.type !== "*");
    }
    getDynamicContextOutputsData(node) {
        return node.outputs.map((output, index) => ({
            name: stripContextInputPrefixes(output.name),
            type: String(output.type),
            index,
        }));
    }
}
SERVICE = new ContextService();
