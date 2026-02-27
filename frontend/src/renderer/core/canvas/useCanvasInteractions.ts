import { computed } from 'vue'

import { isMiddlePointerInput } from '@/base/pointerUtils'
import { useSettingStore } from '@/platform/settings/settingStore'
import { useCanvasStore } from '@/renderer/core/canvas/canvasStore'
import { app } from '@/scripts/app'

/**
 * Composable for handling canvas interactions from Vue components.
 * This provides a unified way to forward events to the LiteGraph canvas.
 */
export function useCanvasInteractions() {
  const settingStore = useSettingStore()
  const canvasStore = useCanvasStore()
  const { getCanvas } = canvasStore

  const isStandardNavMode = computed(
    () => settingStore.get('Comfy.Canvas.NavigationMode') === 'standard'
  )

  /**
   * Whether Vue node components should handle pointer events.
   * Returns false when canvas is in read-only/panning mode (e.g., space key held for panning).
   */
  const shouldHandleNodePointerEvents = computed(
    () => !(canvasStore.canvas?.read_only ?? false)
  )

  /**
   * Returns true if the wheel event target is inside an element that should
   * capture wheel events AND that element (or a descendant) currently has focus.
   *
   * This allows two-finger panning to continue over inputs until the user has
   * actively focused the widget, at which point the widget can consume scroll.
   */
  const wheelCapturedByFocusedElement = (event: WheelEvent): boolean => {
    const target = event.target as HTMLElement | null
    const captureElement = target?.closest('[data-capture-wheel="true"]')
    const active = document.activeElement as Element | null

    return !!(captureElement && active && captureElement.contains(active))
  }

  const shouldForwardWheelEvent = (event: WheelEvent): boolean =>
    !wheelCapturedByFocusedElement(event) ||
    (isStandardNavMode.value && (event.ctrlKey || event.metaKey))

  /**
   * Handles wheel events from UI components that should be forwarded to canvas
   * when appropriate (e.g., Ctrl+wheel for zoom in standard mode)
   */
  const handleWheel = (event: WheelEvent) => {
    if (!shouldForwardWheelEvent(event)) return

    // In standard mode, Ctrl+wheel should go to canvas for zoom
    if (isStandardNavMode.value && (event.ctrlKey || event.metaKey)) {
      forwardEventToCanvas(event)
      return
    }

    // In legacy mode, all wheel events go to canvas for zoom
    if (!isStandardNavMode.value) {
      forwardEventToCanvas(event)
      return
    }

    // Otherwise, let the component handle it normally
  }

  /**
   * Handles pointer events from media elements that should potentially
   * be forwarded to canvas (e.g., space+drag for panning)
   */
  const handlePointer = (event: PointerEvent) => {
    if (isMiddlePointerInput(event)) {
      forwardEventToCanvas(event)
      return
    }

    // Check if canvas exists using established pattern
    const canvas = getCanvas()
    if (!canvas) return

    // Check conditions for forwarding events to canvas
    const isSpacePanningDrag = canvas.read_only && event.buttons === 1 // Space key pressed + left mouse drag
    const isMiddleMousePanning = event.buttons === 4 // Middle mouse button for panning

    if (isSpacePanningDrag || isMiddleMousePanning) {
      event.preventDefault()
      event.stopPropagation()
      forwardEventToCanvas(event)
      return
    }
  }

  /**
   * Forwards an event to the LiteGraph canvas
   */
  const forwardEventToCanvas = (
    event: WheelEvent | PointerEvent | MouseEvent
  ) => {
    // Honor wheel capture only when the element is focused
    if (event instanceof WheelEvent && !shouldForwardWheelEvent(event)) return

    const canvasEl = app.canvas?.canvas
    if (!canvasEl) return
    event.preventDefault()
    event.stopPropagation()

    if (event instanceof WheelEvent) {
      const { clientX, clientY, deltaX, deltaY, ctrlKey, metaKey, shiftKey } =
        event
      canvasEl.dispatchEvent(
        new WheelEvent('wheel', {
          clientX,
          clientY,
          deltaX,
          deltaY,
          ctrlKey,
          metaKey,
          shiftKey
        })
      )
      return
    }

    // Create new event with same properties
    const EventConstructor = event.constructor as
      | typeof MouseEvent
      | typeof PointerEvent
    const newEvent = new EventConstructor(event.type, event)
    canvasEl.dispatchEvent(newEvent)
  }

  return {
    handleWheel,
    handlePointer,
    forwardEventToCanvas,
    shouldHandleNodePointerEvents
  }
}
