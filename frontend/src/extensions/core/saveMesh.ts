import { nextTick } from 'vue'

import Load3D from '@/components/load3d/Load3D.vue'
import { useLoad3d } from '@/composables/useLoad3d'
import { createExportMenuItems } from '@/extensions/core/load3d/exportMenuHelper'
import Load3DConfiguration from '@/extensions/core/load3d/Load3DConfiguration'
import Load3dUtils from '@/extensions/core/load3d/Load3dUtils'
import type { LGraphNode } from '@/lib/litegraph/src/LGraphNode'
import type { IContextMenuValue } from '@/lib/litegraph/src/interfaces'
import type { NodeOutputWith, ResultItem } from '@/schemas/apiSchema'
import type { ComfyNodeDef } from '@/schemas/nodeDefSchema'

type SaveMeshOutput = NodeOutputWith<{
  '3d'?: ResultItem[]
}>
import type { CustomInputSpec } from '@/schemas/nodeDef/nodeDefSchemaV2'
import { ComponentWidgetImpl, addWidget } from '@/scripts/domWidget'
import { useExtensionService } from '@/services/extensionService'
import { useLoad3dService } from '@/services/load3dService'

const inputSpec: CustomInputSpec = {
  name: 'image',
  type: 'Preview3D',
  isPreview: true
}

useExtensionService().registerExtension({
  name: 'Comfy.SaveGLB',

  async beforeRegisterNodeDef(
    _nodeType: typeof LGraphNode,
    nodeData: ComfyNodeDef
  ) {
    if ('SaveGLB' === nodeData.name) {
      // @ts-expect-error InputSpec is not typed correctly
      nodeData.input.required.image = ['PREVIEW_3D']
    }
  },

  getCustomWidgets() {
    return {
      PREVIEW_3D(node) {
        const widget = new ComponentWidgetImpl({
          node,
          name: inputSpec.name,
          component: Load3D,
          inputSpec,
          options: {}
        })

        widget.type = 'load3D'

        addWidget(node, widget)

        return { widget }
      }
    }
  },

  getNodeMenuItems(node: LGraphNode): (IContextMenuValue | null)[] {
    // Only show menu items for SaveGLB nodes
    if (node.constructor.comfyClass !== 'SaveGLB') return []

    const load3d = useLoad3dService().getLoad3d(node)
    if (!load3d) return []

    if (load3d.isSplatModel()) return []

    return createExportMenuItems(load3d)
  },

  async nodeCreated(node: LGraphNode) {
    if (node.constructor.comfyClass !== 'SaveGLB') return

    const [oldWidth, oldHeight] = node.size

    node.setSize([Math.max(oldWidth, 400), Math.max(oldHeight, 550)])

    await nextTick()

    const onExecuted = node.onExecuted

    node.onExecuted = function (output: SaveMeshOutput) {
      onExecuted?.call(this, output)

      const fileInfo = output['3d']?.[0]

      if (!fileInfo) return

      useLoad3d(node).waitForLoad3d((load3d) => {
        const modelWidget = node.widgets?.find((w) => w.name === 'image')

        if (load3d && modelWidget) {
          const filePath =
            (fileInfo.subfolder ?? '') + '/' + (fileInfo.filename ?? '')

          modelWidget.value = filePath

          const config = new Load3DConfiguration(load3d, node.properties)

          const loadFolder = fileInfo.type as 'input' | 'output'

          const onModelLoaded = () => {
            load3d.removeEventListener('modelLoadingEnd', onModelLoaded)
            void Load3dUtils.generateThumbnailIfNeeded(
              load3d,
              filePath,
              loadFolder
            )
          }
          load3d.addEventListener('modelLoadingEnd', onModelLoaded)

          config.configureForSaveMesh(loadFolder, filePath)
        }
      })
    }
  }
})
