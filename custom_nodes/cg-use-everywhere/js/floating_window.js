import { create } from "./use_everywhere_utilities.js"
import { settingsCache } from "./use_everywhere_cache.js"


export class FloatingWindow extends HTMLDivElement {
    constructor() {
        super()
        this.classList.add('ue_editor')
        this.header = create('div', 'ue_editor_header', this)  
        this.body   = create('div', 'ue_editor_body', this)  
        this.footer = create('div', 'ue_editor_footer', this, {innerText:"Close"})  

        this.header.addEventListener('mousedown',this.start_dragging.bind(this))
        this.addEventListener('mousemove', this.mousemovedover.bind(this))
        document.addEventListener('mousemove', this.mousemoved.bind(this))
        document.addEventListener('mouseup',this.stop_dragging.bind(this))
        document.addEventListener('mouseleave',this.stop_dragging.bind(this))
        this.footer.addEventListener('click', this.hide.bind(this))

        this.dragging = false
        this.hide()
        document.body.append(this)
    }

    show() { 
        this.style.display = 'flex';
        const tt = document.getElementById('ue_tooltip')
        if (tt) tt.style.display = 'none' 
        this.showing = true
    }
    hide() { 
        this.style.display = 'none' 
        this.showing = false
    }
    set_title(title) { 
        this.header.innerText = title 
    }
    set_body(element) {
        this.body.innerHTML = ""
        this.body.appendChild(element)
    }

    maybe_move_to(x,y) {
        if (this.position && settingsCache.getSettingValue("Use Everywhere.Graphics.preserve edit window position")) return
        this.move_to(x,y)
    }

    move_to(x,y) {
        this.position = {x:x,y:y}
        this.style.left = `${this.position.x}px`
        this.style.top = `${this.position.y}px`
    }

    swallow(e) {
        e.stopPropagation()
        e.preventDefault()
    }

    start_dragging(e) {
        this.dragging = true
        this.swallow(e)
    }

    stop_dragging(e) {
        this.dragging = false
    }

    mousemovedover(e) {
        this.mousemoved(e)
        this.swallow(e)
    }

    mousemoved(e) {
        if (this.dragging) this.move_to( this.position.x + e.movementX , this.position.y + e.movementY )
    }


}


customElements.define('ue-floating',  FloatingWindow, {extends: 'div'})
export const edit_window = new FloatingWindow()