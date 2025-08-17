import { app, ComfyApp } from "../../scripts/app.js";

export function new_editor() {
    return app.ui.settings.getSettingValue('Comfy.MaskEditor.UseNewEditor')
}

function get_mask_editor_element() {
    return new_editor() ? document.getElementById('maskEditor') : document.getElementById('maskCanvas')?.parentElement
}

export function mask_editor_showing() {
    return get_mask_editor_element() && get_mask_editor_element().style.display != 'none'
}

export function hide_mask_editor() {
    if (mask_editor_showing()) document.getElementById('maskEditor').style.display = 'none'
}

function get_mask_editor_cancel_button() {
    if (document.getElementById("maskEditor_topBarCancelButton")) return document.getElementById("maskEditor_topBarCancelButton")
    return get_mask_editor_element?.parentElement?.lastChild?.childNodes[2]
}

function get_mask_editor_save_button() {
    if (document.getElementById("maskEditor_topBarSaveButton")) return document.getElementById("maskEditor_topBarSaveButton")
    return get_mask_editor_element?.parentElement?.lastChild?.childNodes[2]
}

export function mask_editor_listen_for_cancel(callback) {
    const cancel_button = get_mask_editor_cancel_button()
    if (cancel_button && !cancel_button.filter_listener_added) {
        cancel_button.addEventListener('click', callback )
        cancel_button.filter_listener_added = true
    }
}

export function press_maskeditor_save() {
    get_mask_editor_save_button()?.click()
}

export function press_maskeditor_cancel() {
    get_mask_editor_cancel_button()?.click()
}