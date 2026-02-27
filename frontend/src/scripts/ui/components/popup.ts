import { $el } from '../../ui'
import { prop } from '../../utils'
import { type ClassList, applyClasses } from '../utils'

export class ComfyPopup extends EventTarget {
  element = $el('div.comfyui-popup')
  open: boolean
  children: HTMLElement[]
  target: HTMLElement
  ignoreTarget: boolean
  container: HTMLElement
  position: string
  closeOnEscape: boolean
  horizontal: string
  classList: ClassList

  constructor(
    {
      target,
      container = document.body,
      classList = '',
      ignoreTarget = true,
      closeOnEscape = true,
      position = 'absolute',
      horizontal = 'left'
    }: {
      target: HTMLElement
      container?: HTMLElement
      classList?: ClassList
      ignoreTarget?: boolean
      closeOnEscape?: boolean
      position?: 'absolute' | 'relative'
      horizontal?: 'left' | 'right'
    },
    ...children: HTMLElement[]
  ) {
    super()
    this.target = target
    this.ignoreTarget = ignoreTarget
    this.container = container
    this.position = position
    this.closeOnEscape = closeOnEscape
    this.horizontal = horizontal

    container.append(this.element)

    this.children = prop(this, 'children', children, () => {
      this.element.replaceChildren(...this.children)
      this.update()
    })
    this.classList = prop(this, 'classList', classList, () =>
      applyClasses(this.element, this.classList, 'comfyui-popup', horizontal)
    )
    this.open = prop(this, 'open', false, (v, o) => {
      if (v === o) return
      if (v) {
        this._show()
      } else {
        this._hide()
      }
    })
  }

  toggle() {
    this.open = !this.open
  }

  private _hide() {
    this.element.classList.remove('open')
    window.removeEventListener('resize', this.update)
    window.removeEventListener('click', this._clickHandler, { capture: true })
    window.removeEventListener('keydown', this._escHandler, { capture: true })

    this.dispatchEvent(new CustomEvent('close'))
    this.dispatchEvent(new CustomEvent('change'))
  }

  private _show() {
    this.element.classList.add('open')
    this.update()

    window.addEventListener('resize', this.update)
    window.addEventListener('click', this._clickHandler, { capture: true })
    if (this.closeOnEscape) {
      window.addEventListener('keydown', this._escHandler, { capture: true })
    }

    this.dispatchEvent(new CustomEvent('open'))
    this.dispatchEvent(new CustomEvent('change'))
  }

  // @ts-expect-error fixme ts strict error
  private _escHandler = (e) => {
    if (e.key === 'Escape') {
      this.open = false
      e.preventDefault()
      e.stopImmediatePropagation()
    }
  }

  // @ts-expect-error fixme ts strict error
  private _clickHandler = (e) => {
    /** @type {any} */
    const target = e.target
    if (
      !this.element.contains(target) &&
      this.ignoreTarget &&
      !this.target.contains(target)
    ) {
      this.open = false
    }
  }

  update = () => {
    const rect = this.target.getBoundingClientRect()
    this.element.style.setProperty('--bottom', 'unset')
    if (this.position === 'absolute') {
      if (this.horizontal === 'left') {
        this.element.style.setProperty('--left', rect.left + 'px')
      } else {
        this.element.style.setProperty(
          '--left',
          rect.right - this.element.clientWidth + 'px'
        )
      }
      this.element.style.setProperty('--top', rect.bottom + 'px')
      this.element.style.setProperty('--limit', rect.bottom + 'px')
    } else {
      this.element.style.setProperty('--left', 0 + 'px')
      this.element.style.setProperty('--top', rect.height + 'px')
      this.element.style.setProperty('--limit', rect.height + 'px')
    }

    const thisRect = this.element.getBoundingClientRect()
    if (thisRect.height < 30) {
      // Move up instead
      this.element.style.setProperty('--top', 'unset')
      this.element.style.setProperty('--bottom', rect.height + 5 + 'px')
      this.element.style.setProperty('--limit', rect.height + 5 + 'px')
    }
  }
}
