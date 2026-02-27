import { $el } from '../ui'

export class ComfyDialog<
  T extends HTMLElement = HTMLElement
> extends EventTarget {
  element: T
  textElement!: HTMLElement
  private _buttons: HTMLButtonElement[] | null

  constructor(type = 'div', buttons: HTMLButtonElement[] | null = null) {
    super()
    this._buttons = buttons
    this.element = $el(type + '.comfy-modal', { parent: document.body }, [
      $el('div.comfy-modal-content', [
        $el('p', { $: (p) => (this.textElement = p) }),
        ...this.createButtons()
      ])
    ]) as T
  }

  createButtons() {
    return (
      this._buttons ?? [
        $el('button', {
          type: 'button',
          textContent: 'Close',
          onclick: () => this.close()
        })
      ]
    )
  }

  close() {
    this.element.style.display = 'none'
  }

  show(html: string | HTMLElement | HTMLElement[]): void {
    if (typeof html === 'string') {
      this.textElement.innerHTML = html
    } else {
      this.textElement.replaceChildren(
        ...(html instanceof Array ? html : [html])
      )
    }
    this.element.style.display = 'flex'
  }
}
