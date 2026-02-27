import { $el } from '../../ui'
import { prop } from '../../utils'
import { ComfyButton } from './button'

export class ComfyButtonGroup {
  element = $el('div.comfyui-button-group')
  buttons: (HTMLElement | ComfyButton)[]

  constructor(...buttons: (HTMLElement | ComfyButton)[]) {
    this.buttons = prop(this, 'buttons', buttons, () => this.update())
  }

  insert(button: ComfyButton, index: number) {
    this.buttons.splice(index, 0, button)
    this.update()
  }

  append(button: ComfyButton) {
    this.buttons.push(button)
    this.update()
  }

  remove(indexOrButton: ComfyButton | number) {
    if (typeof indexOrButton !== 'number') {
      indexOrButton = this.buttons.indexOf(indexOrButton)
    }
    if (indexOrButton > -1) {
      const r = this.buttons.splice(indexOrButton, 1)
      this.update()
      return r
    }
  }

  update() {
    // @ts-expect-error fixme ts strict error
    this.element.replaceChildren(...this.buttons.map((b) => b['element'] ?? b))
  }
}
