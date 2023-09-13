export function registerUiOutputListener(nodeType, nodeData, message_type, func) {
    if (nodeData?.ui_output?.includes(message_type)) {
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            if (message[message_type]) func.apply(this, [message[message_type]]);
        }
    }
};