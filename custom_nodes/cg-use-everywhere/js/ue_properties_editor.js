import { LinkRenderController } from "./use_everywhere_ui.js";
import { i18n, default_regex } from "./i18n.js";
import { app } from "../../scripts/app.js";
import { default_priority } from "./ue_properties.js";
import { GROUP_RESTRICTION_OPTIONS, COLOR_RESTRICTION_OPTIONS } from "./i18n.js";
import { edit_window } from "./floating_window.js";

const REGEXES = ['title', 'input', 'group']
const P_REGEXES = ['prompt', 'negative']

export function edit_restrictions(a,b,c,d, node) {
    edit_window.set_body(create_editor_html(node))
    edit_window.set_title(`Restrictions for node #${node.id}`)
    edit_window.maybe_move_to(app.canvas.mouse[0]+10, app.canvas.mouse[1]+10)
    edit_window.show()
}

function add_row(table, header) {
    const row = document.createElement('tr')
    table.appendChild(row)
    const header_elem = document.createElement('th')
    header_elem.innerText = header
    row.appendChild(header_elem)
    return row
}

function add_cell(row, cell) {
    const td = document.createElement('td')
    row.appendChild(td)
    td.appendChild(cell)
}

function changed(node, property, value) {
    node.properties.ue_properties[property] = value

    if (!node.properties.ue_properties.priority) {
        document.getElementById('priority_value').value = `${default_priority(node)}`
    }

    if (node.properties.ue_properties.prompt_regexes) {
        for (var i=0; i<2; i++) {
            if (!node.properties.ue_properties[`${P_REGEXES[i]}_regex`]) {
                document.getElementById(`${P_REGEXES[i]}_regex_value`).value = default_regex(`${P_REGEXES[i]}_regex`)
            }
        }
    }

    const elem = document.getElementById(`${property}_value`)
    if (elem) elem.style.opacity = (value) ? "1" : "0.5"

    LinkRenderController.instance().mark_link_list_outdated()
    app.canvas.setDirty(true,true)
}

function create_editor_html(node) {
    const table = document.createElement('table')

    for (var i=0; i<=2; i++) {
        
        if (i==1 && node.properties.ue_properties.prompt_regexes) {
            for (var j=0; j<2; j++) {
                const name = P_REGEXES[j]
                const row = add_row(table, `${i18n(name)} regex`)
                const input = document.createElement('input')
                input.value = node.properties.ue_properties[`${name}_regex`] || default_regex(`${name}_regex`)
                if (!node.properties.ue_properties[`${name}_regex`]) input.style.opacity = 0.5
                input.id = `${name}_regex_value`
                input.style.width = '250px'
                input.addEventListener('input', ()=>{ changed(node, `${name}_regex`, input.value)})
                add_cell(row,input)                
            }
        } else {
            const name = REGEXES[i]
            const row = add_row(table, `${i18n(name)} regex`)
            const input = document.createElement('input')
            input.value = node.properties.ue_properties[`${name}_regex`] || ''
            input.addEventListener('input', ()=>{ changed(node, `${name}_regex`, input.value)})
            add_cell(row,input)
        }
    }

    const gr_row    = add_row(table, i18n("Group"))
    const gr_select = document.createElement('select')
    add_cell(gr_row,gr_select)
    GROUP_RESTRICTION_OPTIONS.forEach((gro, i)=>{
        const gr_option = document.createElement('option')
        gr_option.value = `${i}`
        gr_option.innerText = gro
        gr_select.appendChild(gr_option)
    })
    gr_select.value = `${node.properties.ue_properties.group_restricted || 0}`
    gr_select.addEventListener('input', ()=>{ changed(node, `group_restricted`, parseInt(gr_select.value))})

    const col_row    = add_row(table, i18n("Color"))
    const col_select = document.createElement('select')
    add_cell(col_row,col_select)
    COLOR_RESTRICTION_OPTIONS.forEach((cro, i)=>{
        const col_option = document.createElement('option')
        col_option.value = `${i}`
        col_option.innerText = cro
        col_select.appendChild(col_option)
    })
    col_select.value = `${node.properties.ue_properties.color_restricted || 0}`
    col_select.addEventListener('input', ()=>{ changed(node, `color_restricted`, parseInt(col_select.value))})

    const priority_row = add_row(table, i18n("Priority"))
    const priority_edit = document.createElement("input")
    priority_edit.value = `${node.properties.ue_properties.priority || default_priority(node)}`
    priority_edit.addEventListener('input', ()=>{ 
        const p = parseInt(priority_edit.value)
        if (p) changed(node, `priority`, p)
        if (priority_edit.value=='') changed(node, `priority`, undefined)
    })
    priority_edit.id = 'priority_value'
    if (!node.properties.ue_properties.priority) priority_edit.style.opacity = 0.5
    add_cell(priority_row,priority_edit)
    return table
}