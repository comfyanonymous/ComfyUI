/**
 * Checks if a pointer/mouse event has multi-select modifier keys pressed.
 * Multi-select keys are: Ctrl (Windows/Linux), Cmd (Mac), or Shift
 *
 * @param event - The pointer or mouse event to check
 * @returns true if any multi-select modifier key is pressed
 */
export function isMultiSelectKey(event: PointerEvent | MouseEvent): boolean {
  return event.ctrlKey || event.metaKey || event.shiftKey
}
