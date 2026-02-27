import { Editor as TiptapEditor } from '@tiptap/core'
import TiptapLink from '@tiptap/extension-link'
import TiptapTable from '@tiptap/extension-table'
import TiptapTableCell from '@tiptap/extension-table-cell'
import TiptapTableHeader from '@tiptap/extension-table-header'
import TiptapTableRow from '@tiptap/extension-table-row'
import TiptapStarterKit from '@tiptap/starter-kit'
import { Markdown as TiptapMarkdown } from 'tiptap-markdown'

import { resolveNodeRootGraphId } from '@/lib/litegraph/src/litegraph'
import type { LGraphNode } from '@/lib/litegraph/src/litegraph'
import type { InputSpec } from '@/schemas/nodeDef/nodeDefSchemaV2'
import { app } from '@/scripts/app'
import type { ComfyWidgetConstructorV2 } from '@/scripts/widgets'
import { useWidgetValueStore } from '@/stores/widgetValueStore'

// TODO: This widget manually syncs with widgetValueStore via getValue/setValue.
// Consolidate with useStringWidget into shared helpers (domWidgetHelpers.ts).
function addMarkdownWidget(
  node: LGraphNode,
  name: string,
  opts: { defaultVal: string }
) {
  TiptapMarkdown.configure({
    html: false,
    breaks: true,
    transformPastedText: true
  })
  const editor = new TiptapEditor({
    extensions: [
      TiptapStarterKit,
      TiptapMarkdown,
      TiptapLink,
      TiptapTable,
      TiptapTableCell,
      TiptapTableHeader,
      TiptapTableRow
    ],
    content: opts.defaultVal,
    editable: false
  })

  const widgetStore = useWidgetValueStore()

  const inputEl = editor.options.element as HTMLElement
  inputEl.classList.add('comfy-markdown')
  const textarea = document.createElement('textarea')
  inputEl.append(textarea)

  const widget = node.addDOMWidget(name, 'MARKDOWN', inputEl, {
    getValue(): string {
      const graphId = resolveNodeRootGraphId(node, app.rootGraph.id)
      const storedValue = widgetStore.getWidget(graphId, node.id, name)?.value
      return typeof storedValue === 'string' ? storedValue : textarea.value
    },
    setValue(v: string) {
      textarea.value = v
      editor.commands.setContent(v)
      const graphId = resolveNodeRootGraphId(node, app.rootGraph.id)
      const widgetState = widgetStore.getWidget(graphId, node.id, name)
      if (widgetState) widgetState.value = v
    }
  })
  widget.element = inputEl
  widget.options.minNodeSize = [400, 200]

  inputEl.addEventListener('input', (event) => {
    if (event.target instanceof HTMLTextAreaElement) {
      widget.value = event.target.value
    }
    widget.callback?.(widget.value)
  })

  inputEl.addEventListener('dblclick', () => {
    inputEl.classList.add('editing')
    setTimeout(() => {
      textarea.focus()
    }, 0)
  })

  textarea.addEventListener('blur', () => {
    inputEl.classList.remove('editing')
  })

  textarea.addEventListener('change', () => {
    editor.commands.setContent(textarea.value)
    widget.callback?.(widget.value)
  })

  inputEl.addEventListener('keydown', (event: KeyboardEvent) => {
    event.stopPropagation()
  })

  inputEl.addEventListener('pointerdown', (event: PointerEvent) => {
    if (event.button === 1) {
      app.canvas.processMouseDown(event)
    }
  })

  inputEl.addEventListener('pointermove', (event: PointerEvent) => {
    if ((event.buttons & 4) === 4) {
      app.canvas.processMouseMove(event)
    }
  })

  inputEl.addEventListener('pointerup', (event: PointerEvent) => {
    if (event.button === 1) {
      app.canvas.processMouseUp(event)
    }
  })

  return widget
}

export const useMarkdownWidget = () => {
  const widgetConstructor: ComfyWidgetConstructorV2 = (
    node: LGraphNode,
    inputSpec: InputSpec
  ) => {
    return addMarkdownWidget(node, inputSpec.name, {
      defaultVal: inputSpec.default ?? ''
    })
  }

  return widgetConstructor
}
