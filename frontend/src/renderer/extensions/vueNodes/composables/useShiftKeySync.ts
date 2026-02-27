import { tryOnScopeDispose, useEventListener } from '@vueuse/core'
import { shallowRef } from 'vue'

import { app } from '@/scripts/app'

/**
 * Composable for synchronizing shift key state from Vue nodes to LiteGraph canvas.
 *
 * Enables snap-to-grid preview rendering in LiteGraph during Vue node drag/resize operations
 * by dispatching synthetic keyboard events to the canvas element.
 *
 * @returns Object containing trackShiftKey function for shift state synchronization lifecycle
 *
 * @example
 * ```ts
 * const { trackShiftKey } = useShiftKeySync()
 *
 * function startDrag(event: PointerEvent) {
 *   const stopTracking = trackShiftKey(event)
 *   // ... drag logic
 *   // Call stopTracking() on pointerup to cleanup listeners
 * }
 * ```
 */
export function useShiftKeySync() {
  const shiftKeyState = shallowRef(false)
  let canvasEl: HTMLCanvasElement | null = null

  /**
   * Synchronizes shift key state to LiteGraph canvas by dispatching synthetic keyboard events.
   *
   * Only dispatches events when shift state actually changes to minimize overhead.
   * Canvas reference is lazily initialized on first sync.
   *
   * @param isShiftPressed - Current shift key state to synchronize
   */
  function syncShiftState(isShiftPressed: boolean) {
    if (isShiftPressed === shiftKeyState.value) return

    // Lazy-initialize canvas reference on first use
    if (!canvasEl) {
      canvasEl = app.canvas?.canvas ?? null
      if (!canvasEl) return // Canvas not ready yet
    }

    shiftKeyState.value = isShiftPressed
    canvasEl.dispatchEvent(
      new KeyboardEvent(isShiftPressed ? 'keydown' : 'keyup', {
        key: 'Shift',
        shiftKey: isShiftPressed,
        bubbles: true
      })
    )
  }

  /**
   * Tracks shift key state during drag/resize operations and synchronizes to canvas.
   *
   * Attaches window-level keyboard event listeners for the duration of the operation.
   * Listeners are automatically cleaned up when the returned function is called.
   *
   * @param initialEvent - Initial pointer event containing shift key state at drag/resize start
   * @returns Cleanup function that removes event listeners - must be called when operation ends
   *
   * @example
   * ```ts
   * function startDrag(event: PointerEvent) {
   *   const stopTracking = trackShiftKey(event)
   *
   *   const handlePointerUp = () => {
   *     stopTracking() // Cleanup listeners
   *   }
   * }
   * ```
   */
  function trackShiftKey(initialEvent: PointerEvent): () => void {
    // Sync initial shift state
    syncShiftState(initialEvent.shiftKey)

    // Listen for shift key press/release during the operation
    const handleKeyEvent = (e: KeyboardEvent) => {
      if (e.key !== 'Shift') return
      syncShiftState(e.shiftKey)
    }

    const stopKeydown = useEventListener(window, 'keydown', handleKeyEvent, {
      passive: true
    })
    const stopKeyup = useEventListener(window, 'keyup', handleKeyEvent, {
      passive: true
    })

    // Return cleanup function that stops both listeners
    return () => {
      stopKeydown()
      stopKeyup()
    }
  }

  // Cleanup on component unmount
  tryOnScopeDispose(() => {
    shiftKeyState.value = false
    canvasEl = null
  })

  return { trackShiftKey }
}
