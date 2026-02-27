import { useCanvasStore } from '@/renderer/core/canvas/canvasStore'
/**
 * Utility functions for handling workbench events
 */

/**
 * Check if there is selected text in the document.
 */
function hasTextSelection(): boolean {
  const selection = window.getSelection()
  return selection !== null && selection.toString().trim().length > 0
}

/**
 * Used by clipboard handlers to determine if copy/paste events should be
 * intercepted for graph operations vs. allowing default browser behavior
 * for text inputs and other UI elements.
 *
 * @param target - The event target to check
 * @returns true if copy paste events will be handled by target
 */
export function shouldIgnoreCopyPaste(target: EventTarget | null): boolean {
  const isTextInput =
    target instanceof HTMLTextAreaElement ||
    (target instanceof HTMLInputElement &&
      ![
        'button',
        'checkbox',
        'file',
        'hidden',
        'image',
        'radio',
        'range',
        'reset',
        'search',
        'submit'
      ].includes(target.type))
  return isTextInput || useCanvasStore().linearMode || hasTextSelection()
}
