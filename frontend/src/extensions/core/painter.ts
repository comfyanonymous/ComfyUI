import { useExtensionService } from '@/services/extensionService'

const HIDDEN_WIDGETS = new Set(['width', 'height', 'bg_color'])

useExtensionService().registerExtension({
  name: 'Comfy.Painter',

  nodeCreated(node) {
    if (node.constructor.comfyClass !== 'Painter') return

    const [oldWidth, oldHeight] = node.size
    node.setSize([Math.max(oldWidth, 450), Math.max(oldHeight, 550)])

    node.hideOutputImages = true

    for (const widget of node.widgets ?? []) {
      if (HIDDEN_WIDGETS.has(widget.name)) {
        widget.hidden = true
      }
    }
  }
})
