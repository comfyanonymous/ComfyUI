import { LGraphCanvas, LiteGraph } from '@/lib/litegraph/src/litegraph'
import { LGraphNode } from '@/lib/litegraph/src/litegraph'

import { app } from '../../scripts/app'
import { ComfyWidgets } from '../../scripts/widgets'

// Node that add notes to your project

app.registerExtension({
  name: 'Comfy.NoteNode',
  registerCustomNodes() {
    class NoteNode extends LGraphNode {
      static override category: string
      static collapsable: boolean
      static title_mode: number

      groupcolor = LGraphCanvas.node_colors.yellow.groupcolor
      override isVirtualNode: boolean

      constructor(title: string) {
        super(title)

        this.color = LGraphCanvas.node_colors.yellow.color
        this.bgcolor = LGraphCanvas.node_colors.yellow.bgcolor

        if (!this.properties) {
          this.properties = { text: '' }
        }
        ComfyWidgets.STRING(
          this,
          'text',
          ['STRING', { default: this.properties.text, multiline: true }],
          app
        )

        this.serialize_widgets = true
        this.isVirtualNode = true
      }
    }

    // Load default visibility

    LiteGraph.registerNodeType(
      'Note',
      Object.assign(NoteNode, {
        title_mode: LiteGraph.NORMAL_TITLE,
        title: 'Note',
        collapsable: true
      })
    )

    NoteNode.category = 'utils'

    /** Markdown variant of NoteNode */
    class MarkdownNoteNode extends LGraphNode {
      static override title = 'Markdown Note'

      groupcolor = LGraphCanvas.node_colors.yellow.groupcolor

      constructor(title: string) {
        super(title)

        this.color = LGraphCanvas.node_colors.yellow.color
        this.bgcolor = LGraphCanvas.node_colors.yellow.bgcolor

        if (!this.properties) {
          this.properties = { text: '' }
        }
        ComfyWidgets.MARKDOWN(
          this,
          'text',
          ['STRING', { default: this.properties.text }],
          app
        )

        this.serialize_widgets = true
        this.isVirtualNode = true
      }
    }

    LiteGraph.registerNodeType('MarkdownNote', MarkdownNoteNode)
    MarkdownNoteNode.category = 'utils'
  }
})
