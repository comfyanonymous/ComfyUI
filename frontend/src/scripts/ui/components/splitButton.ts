import { $el } from '../../ui'
import { prop } from '../../utils'
import { ComfyButton } from './button'
import { ComfyPopup } from './popup'

export class ComfySplitButton {
  arrow: ComfyButton
  element: HTMLElement
  popup: ComfyPopup
  items: Array<HTMLElement | ComfyButton>

  constructor(
    {
      primary,
      mode,
      horizontal = 'left',
      position = 'relative'
    }: {
      primary: ComfyButton
      mode?: 'hover' | 'click'
      horizontal?: 'left' | 'right'
      position?: 'relative' | 'absolute'
    },
    ...items: Array<HTMLElement | ComfyButton>
  ) {
    this.arrow = new ComfyButton({
      icon: 'chevron-down'
    })
    this.element = $el(
      'div.comfyui-split-button' + (mode === 'hover' ? '.hover' : ''),
      [
        $el(
          'div.comfyui-split-primary',
          {
            ariaLabel: 'Queue current workflow'
          },
          primary.element
        ),
        $el(
          'div.comfyui-split-arrow',
          {
            ariaLabel: 'Open extra opens',
            ariaHasPopup: 'true'
          },
          this.arrow.element
        )
      ]
    )
    this.popup = new ComfyPopup({
      target: this.element,
      container: position === 'relative' ? this.element : document.body,
      classList:
        'comfyui-split-button-popup' + (mode === 'hover' ? ' hover' : ''),
      closeOnEscape: mode === 'click',
      position,
      horizontal
    })

    this.arrow.withPopup(this.popup, mode)

    this.items = prop(this, 'items', items, () => this.update())
  }

  update() {
    this.popup.element.replaceChildren(
      ...this.items.map((b) => ('element' in b ? b.element : b))
    )
  }
}
