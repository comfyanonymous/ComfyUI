import type { ComfyApp } from '@/scripts/app'

import { $el } from '../../ui'
import { ComfyButtonGroup } from '../components/buttonGroup'
import './menu.css'

// Export to make sure following components are shimmed and exported by vite
export { ComfyButton } from '../components/button'
export { ComfySplitButton } from '../components/splitButton'
export { ComfyPopup } from '../components/popup'
export { ComfyAsyncDialog } from '@/scripts/ui/components/asyncDialog'
export { DraggableList } from '@/scripts/ui/draggableList'
export { applyTextReplacements, addStylesheet } from '@/scripts/utils'

export class ComfyAppMenu {
  app: ComfyApp
  actionsGroup: ComfyButtonGroup
  settingsGroup: ComfyButtonGroup
  viewGroup: ComfyButtonGroup
  element: HTMLElement

  constructor(app: ComfyApp) {
    this.app = app

    // Keep the group as there are custom scripts attaching extra
    // elements to it.
    this.actionsGroup = new ComfyButtonGroup()
    this.settingsGroup = new ComfyButtonGroup()
    this.viewGroup = new ComfyButtonGroup()

    this.element = $el('div.flex.gap-2.mx-2', [
      this.actionsGroup.element,
      this.settingsGroup.element,
      this.viewGroup.element
    ])
  }
}
