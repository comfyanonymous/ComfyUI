import type { LGraphNode } from '@/lib/litegraph/src/LGraphNode'
import type { NodeOutputWith } from '@/schemas/apiSchema'
import { api } from '@/scripts/api'
import { app } from '@/scripts/app'
import { useExtensionService } from '@/services/extensionService'

type ImageCompareOutput = NodeOutputWith<{
  a_images?: Record<string, string>[]
  b_images?: Record<string, string>[]
}>

useExtensionService().registerExtension({
  name: 'Comfy.ImageCompare',

  async nodeCreated(node: LGraphNode) {
    if (node.constructor.comfyClass !== 'ImageCompare') return

    const [oldWidth, oldHeight] = node.size
    node.setSize([Math.max(oldWidth, 400), Math.max(oldHeight, 350)])

    const onExecuted = node.onExecuted

    node.onExecuted = function (output: ImageCompareOutput) {
      onExecuted?.call(this, output)

      const { a_images: aImages, b_images: bImages } = output
      const rand = app.getRandParam()

      const toUrl = (params: Record<string, string>) =>
        api.apiURL(`/view?${new URLSearchParams(params)}${rand}`)

      const beforeImages =
        aImages && aImages.length > 0 ? aImages.map(toUrl) : []
      const afterImages =
        bImages && bImages.length > 0 ? bImages.map(toUrl) : []

      const widget = node.widgets?.find((w) => w.type === 'imagecompare')

      if (widget) {
        widget.value = { beforeImages, afterImages }
        widget.callback?.(widget.value)
      }
    }
  }
})
