import type { LGraphNode } from '@/lib/litegraph/src/LGraphNode'
import { useExtensionService } from '@/services/extensionService'

useExtensionService().registerExtension({
  name: 'Comfy.ImageCrop',

  async nodeCreated(node: LGraphNode) {
    if (node.constructor.comfyClass !== 'ImageCropV2') return

    node.hideOutputImages = true
    const [oldWidth, oldHeight] = node.size
    node.setSize([Math.max(oldWidth, 300), Math.max(oldHeight, 450)])
  }
})
