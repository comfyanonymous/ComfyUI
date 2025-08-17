
import { any_restrictions, describe_restrictions } from "./ue_properties.js";
import { app } from "../../scripts/app.js";
import { create } from "./use_everywhere_utilities.js";
import { edit_window } from "./floating_window.js";
import { settingsCache } from "./use_everywhere_cache.js";

const HOVERTIME = 500
var hover_node_id = null
var mouse_pos     = [0,0]

const ue_tooltip_element = create('span', 'ue_tooltip', document.body, {id:'ue_tooltip'})

function show_tooltip(node) {
    if (!node) return
    ue_tooltip_element.style.display = "block"
    ue_tooltip_element.style.left = `${mouse_pos.x+10}px`
    ue_tooltip_element.style.top = `${mouse_pos.y+5}px`
    ue_tooltip_element.innerHTML = ""
    ue_tooltip_element.appendChild(describe_restrictions(node))
    ue_tooltip_element.showing = node.id
}

function show_on_hover() {
    if (mouse_pos.x==app.canvas.mouse[0] && mouse_pos.y==app.canvas.mouse[1]) {
        show_tooltip(app.canvas.node_over)
    } else if (app.canvas.node_over?.id == hover_node_id) {
        mouse_pos = { x:app.canvas.mouse[0], y:app.canvas.mouse[1] }
        setTimeout(show_on_hover, HOVERTIME)
    }
}

function hide_tooltip() {
    var ue_tooltip_element = document.getElementById('ue_tooltip')
    if (ue_tooltip_element) {
        ue_tooltip_element.style.display = "none"
        ue_tooltip_element.showing = null
    }
}

export function maybe_show_tooltip() {
    if (!settingsCache.getSettingValue('Use Everywhere.Graphics.tooltips')) return

    const node = app.canvas?.node_over
    hover_node_id = node?.id

    if (!node) return hide_tooltip()
    if (edit_window.showing) return hide_tooltip()
    if (!(node.IS_UE && any_restrictions(node))) return hide_tooltip()
    
    if (ue_tooltip_element.showing) return
    
    mouse_pos = { x:app.canvas.mouse[0], y:app.canvas.mouse[1] }
    setTimeout(show_on_hover, HOVERTIME)
}