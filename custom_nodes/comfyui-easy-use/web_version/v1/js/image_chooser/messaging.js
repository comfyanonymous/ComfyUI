import { api } from "../../../../scripts/api.js";
import { FlowState } from "./state.js";

function send_message_from_pausing_node(message) {
    const id = app.runningNodeId;
    send_message(id, message);
}

function send_message(id, message) {
    const body = new FormData();
    body.append('message',message);
    body.append('id', id);
    api.fetchApi("/easyuse/image_chooser_message", { method: "POST", body, });
}

function send_cancel() {
    send_message(-1,'__cancel__');
    FlowState.cancelling = true;
    api.interrupt();
    FlowState.cancelling = false;
}

var skip_next = 0;
function skip_next_restart_message() { skip_next += 1; }
function send_onstart() {
    if (skip_next>0) {
        skip_next -= 1;
        return false;
    }
    send_message(-1,'__start__');
    return true;
}

export { send_message_from_pausing_node, send_cancel, send_message, send_onstart, skip_next_restart_message }