import type { LGraphNode } from '@/lib/litegraph/src/litegraph'
import { useCanvasInteractions } from '@/renderer/core/canvas/useCanvasInteractions'
import { ANIM_PREVIEW_WIDGET } from '@/scripts/app'
import { isDOMWidget } from '@/scripts/domWidget'

/**
 * Composable for handling animated image previews in nodes
 */
export function useNodeAnimatedImage() {
  /**
   * Shows animated image preview for a node
   * @param node The graph node to show the preview for
   */
  function showAnimatedPreview(node: LGraphNode) {
    if (!node.imgs?.length) return
    if (!node.widgets) return

    const widgetIdx = node.widgets.findIndex(
      (w) => w.name === ANIM_PREVIEW_WIDGET
    )

    node.imgs[0].classList.value = 'block size-full object-contain'
    if (widgetIdx > -1) {
      // Replace content in existing widget
      const widget = node.widgets[widgetIdx]
      if (!isDOMWidget(widget)) return
      widget.element.replaceChildren(node.imgs[0])
    } else {
      // Create new widget
      const element = document.createElement('div')
      element.appendChild(node.imgs[0])
      const widget = node.addDOMWidget(ANIM_PREVIEW_WIDGET, 'img', element, {
        hideOnZoom: false
      })
      node.overIndex = 0

      // Add event listeners for canvas interactions
      const { handleWheel, handlePointer, forwardEventToCanvas } =
        useCanvasInteractions()
      node.imgs[0].style.pointerEvents = 'none'
      element.addEventListener('wheel', handleWheel)
      element.addEventListener('pointermove', handlePointer)
      element.addEventListener('pointerup', handlePointer)
      element.addEventListener(
        'pointerdown',
        (e) => {
          return e.button !== 2 ? handlePointer(e) : forwardEventToCanvas(e)
        },
        true
      )

      widget.serialize = false
      widget.serializeValue = () => undefined
    }
  }

  /**
   * Removes animated image preview from a node
   * @param node The graph node to remove the preview from
   */
  function removeAnimatedPreview(node: LGraphNode) {
    if (!node.widgets) return

    const widgetIdx = node.widgets.findIndex(
      (w) => w.name === ANIM_PREVIEW_WIDGET
    )

    if (widgetIdx > -1) {
      node.widgets[widgetIdx].onRemove?.()
      node.widgets.splice(widgetIdx, 1)
    }
  }

  return {
    showAnimatedPreview,
    removeAnimatedPreview
  }
}
