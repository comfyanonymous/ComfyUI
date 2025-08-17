
export class FloatingWindow extends HTMLDivElement {
    constructor(title, x, y, parent, movecallback) {
        super()
        this.movecallback = movecallback
        this.classList.add('cgfloat')
        this.header = document.createElement('div')
        this.header.classList.add('cgfloat_header')
        this.header.innerText = title
        this.append(this.header)
        this.body = document.createElement('div')
        this.body.classList.add('cgfloat_body')
        this.append(this.body)

        this.header.addEventListener('mousedown',this.header_mousedown.bind(this))
        document.addEventListener('mouseup',this.header_mouseup.bind(this))
        document.addEventListener('mousemove',this.header_mousemove.bind(this))
        document.addEventListener('mouseleave',this.header_mouseup.bind(this))
        
        this.dragging = false
        this.move_to(x,y)
        

        if (parent) parent.append(this)
        else document.body.append(this)
    }

    show() { this.style.display = 'block' }
    hide() { this.style.display = 'none' }
    set_title(title) { this.header.innerText = title }

    move_to(x,y,supress) {
        this.position = {x:x,y:y}
        this.style.left = `${this.position.x}px`
        this.style.top = `${this.position.y}px`
        if (!supress) this.movecallback(x,y)
    }

    swallow(e) {
        e.stopPropagation()
        e.preventDefault()
    }

    header_mousedown(e) {
        this.dragging = true
        this.swallow(e)
    }

    header_mouseup(e) {
        this.dragging = false
    }

    header_mousemove(e) {
        if (this.dragging) {
            this.move_to( this.position.x + e.movementX , this.position.y + e.movementY )
            this.swallow(e)
        }
    }


}
customElements.define('cg-floater',  FloatingWindow, {extends: 'div'})