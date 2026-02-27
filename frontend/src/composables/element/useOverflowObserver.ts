import { useMutationObserver, useResizeObserver } from '@vueuse/core'
import { debounce } from 'es-toolkit/compat'
import { readonly, ref } from 'vue'

/**
 * Observes an element for overflow changes and optionally debounces the check
 * @param element - The element to observe
 * @param options - The options for the observer
 * @param options.debounceTime - The time to debounce the check in milliseconds
 * @param options.useMutationObserver - Whether to use a mutation observer to check for overflow
 * @param options.useResizeObserver - Whether to use a resize observer to check for overflow
 * @returns An object containing the isOverflowing state and the checkOverflow function to manually trigger
 */
export const useOverflowObserver = (
  element: HTMLElement,
  options?: {
    debounceTime?: number
    useMutationObserver?: boolean
    useResizeObserver?: boolean
    onCheck?: (isOverflowing: boolean) => void
  }
) => {
  options = {
    debounceTime: 25,
    useMutationObserver: true,
    useResizeObserver: true,
    ...options
  }

  const isOverflowing = ref(false)
  const disposeFns: (() => void)[] = []
  const disposed = ref(false)

  const checkOverflowFn = () => {
    isOverflowing.value = element.scrollWidth > element.clientWidth
    options.onCheck?.(isOverflowing.value)
  }

  const checkOverflow = options.debounceTime
    ? debounce(checkOverflowFn, options.debounceTime)
    : checkOverflowFn

  if (options.useMutationObserver) {
    disposeFns.push(
      useMutationObserver(element, checkOverflow, {
        subtree: true,
        childList: true
      }).stop
    )
  }
  if (options.useResizeObserver) {
    disposeFns.push(useResizeObserver(element, checkOverflow).stop)
  }

  return {
    isOverflowing: readonly(isOverflowing),
    disposed: readonly(disposed),
    checkOverflow,
    dispose: () => {
      disposed.value = true
      disposeFns.forEach((fn) => fn())
    }
  }
}
