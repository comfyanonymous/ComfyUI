/**
 * Utilities for pointer event handling
 */

/**
 * Checks if a pointer or mouse event is a middle button input
 * @param event - The pointer or mouse event to check
 * @returns true if the event is from the middle button/wheel
 */
export function isMiddlePointerInput(
  event: PointerEvent | MouseEvent
): boolean {
  if ('button' in event && event.button === 1) {
    return true
  }

  if ('buttons' in event && typeof event.buttons === 'number') {
    return event.buttons === 4
  }

  return false
}
