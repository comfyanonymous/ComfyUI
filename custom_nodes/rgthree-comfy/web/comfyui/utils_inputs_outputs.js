export function removeUnusedInputsFromEnd(node, minNumber = 1, nameMatch) {
    var _a;
    if (node.removed)
        return;
    for (let i = node.inputs.length - 1; i >= minNumber; i--) {
        if (!((_a = node.inputs[i]) === null || _a === void 0 ? void 0 : _a.link)) {
            if (!nameMatch || nameMatch.test(node.inputs[i].name)) {
                node.removeInput(i);
            }
            continue;
        }
        break;
    }
}
