import type { LGraphNode } from '@/lib/litegraph/src/LGraphNode'
import type { ComfyNodeDef } from '@/schemas/nodeDefSchema'
import { applyTextReplacements } from '@/utils/searchAndReplace'

import { app } from '../../scripts/app'

const saveNodeTypes = new Set([
  'SaveImage',
  'SaveVideo',
  'SaveAnimatedWEBP',
  'SaveWEBM',
  'SaveAudio',
  'SaveGLB',
  'SaveAnimatedPNG',
  'CLIPSave',
  'VAESave',
  'ModelSave',
  'LoraSave',
  'SaveLatent'
])

// Use widget values and dates in output filenames

app.registerExtension({
  name: 'Comfy.SaveImageExtraOutput',
  async beforeRegisterNodeDef(
    nodeType: typeof LGraphNode,
    nodeData: ComfyNodeDef
  ) {
    if (saveNodeTypes.has(nodeData.name)) {
      const onNodeCreated = nodeType.prototype.onNodeCreated
      // When the SaveImage node is created we want to override the serialization of the output name widget to run our S&R
      nodeType.prototype.onNodeCreated = function () {
        const r = onNodeCreated
          ? // @ts-expect-error fixme ts strict error
            onNodeCreated.apply(this, arguments)
          : undefined

        // @ts-expect-error fixme ts strict error
        const widget = this.widgets.find((w) => w.name === 'filename_prefix')
        // @ts-expect-error fixme ts strict error
        widget.serializeValue = () => {
          // @ts-expect-error fixme ts strict error
          return applyTextReplacements(app.graph, widget.value)
        }

        return r
      }
    } else {
      // When any other node is created add a property to alias the node
      const onNodeCreated = nodeType.prototype.onNodeCreated
      nodeType.prototype.onNodeCreated = function () {
        const r = onNodeCreated
          ? // @ts-expect-error fixme ts strict error
            onNodeCreated.apply(this, arguments)
          : undefined

        if (!this.properties || !('Node name for S&R' in this.properties)) {
          this.addProperty('Node name for S&R', this.constructor.type, 'string')
        }

        return r
      }
    }
  }
})
