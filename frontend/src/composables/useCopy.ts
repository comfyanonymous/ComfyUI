import { useEventListener } from '@vueuse/core'

import { useCanvasStore } from '@/renderer/core/canvas/canvasStore'
import { shouldIgnoreCopyPaste } from '@/workbench/eventHelpers'

const clipboardHTMLWrapper = [
  '<meta charset="utf-8"><div><span data-metadata="',
  '"></span></div><span style="white-space:pre-wrap;">Text</span>'
]

/**
 * Adds a handler on copy that serializes selected nodes to JSON
 */
export const useCopy = () => {
  const canvasStore = useCanvasStore()

  useEventListener(document, 'copy', (e) => {
    if (shouldIgnoreCopyPaste(e.target)) {
      // Default system copy
      return
    }
    // copy nodes and clear clipboard
    const canvas = canvasStore.canvas
    if (canvas?.selectedItems) {
      const serializedData = canvas.copyToClipboard()
      // Use TextEncoder to handle Unicode characters properly
      const base64Data = btoa(
        String.fromCharCode(
          ...Array.from(new TextEncoder().encode(serializedData))
        )
      )
      // clearData doesn't remove images from clipboard
      e.clipboardData?.setData(
        'text/html',
        clipboardHTMLWrapper.join(base64Data)
      )
      e.preventDefault()
      e.stopImmediatePropagation()
      return false
    }
  })
}
