import { useDebounceFn, useEventListener } from '@vueuse/core'
import { ref } from 'vue'
import type { MaybeRefOrGetter } from 'vue'

interface TransformSettlingOptions {
  /**
   * Delay in ms before transform is considered "settled" after last interaction
   * @default 200
   */
  settleDelay?: number
  /**
   * Whether to use passive event listeners (better performance but can't preventDefault)
   * @default true
   */
  passive?: boolean
}

/**
 * Tracks when canvas zoom transforms are actively changing vs settled.
 *
 * This composable helps optimize rendering quality during zoom transformations.
 * When the user is actively zooming, we can reduce rendering quality
 * for better performance. Once the transform "settles" (stops changing), we can
 * trigger high-quality re-rasterization.
 *
 * The settling concept prevents constant quality switching during interactions
 * by waiting for a period of inactivity before considering the transform complete.
 *
 * Uses VueUse's useEventListener for automatic cleanup and useDebounceFn for
 * efficient settle detection.
 *
 * @example
 * ```ts
 * const { isTransforming } = useTransformSettling(canvasRef, {
 *   settleDelay: 200
 * })
 *
 * // Use in CSS classes or rendering logic
 * const cssClass = computed(() => ({
 *   'low-quality': isTransforming.value,
 *   'high-quality': !isTransforming.value
 * }))
 * ```
 */
export function useTransformSettling(
  target: MaybeRefOrGetter<HTMLElement | null | undefined>,
  options: TransformSettlingOptions = {}
) {
  const { settleDelay = 256, passive = true } = options

  const isTransforming = ref(false)

  /**
   * Mark transform as active
   */
  const markTransformActive = () => {
    isTransforming.value = true
  }

  /**
   * Mark transform as settled (debounced)
   */
  const markTransformSettled = useDebounceFn(() => {
    isTransforming.value = false
  }, settleDelay)

  /**
   * Handle zoom transform event - mark active then queue settle
   */
  const handleWheel = () => {
    markTransformActive()
    void markTransformSettled()
  }

  // Register wheel event listener with auto-cleanup
  useEventListener(target, 'wheel', handleWheel, {
    capture: true,
    passive
  })

  return {
    isTransforming
  }
}
