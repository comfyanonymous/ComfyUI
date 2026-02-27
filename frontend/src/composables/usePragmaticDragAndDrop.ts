import {
  draggable,
  dropTargetForElements
} from '@atlaskit/pragmatic-drag-and-drop/element/adapter'
import { onBeforeUnmount, onMounted, toValue } from 'vue'
import type { MaybeRefOrGetter } from 'vue'

export function usePragmaticDroppable(
  dropTargetElement: MaybeRefOrGetter<HTMLElement | null>,
  options: Omit<Parameters<typeof dropTargetForElements>[0], 'element'>
) {
  let cleanup = () => {}

  onMounted(() => {
    const element = toValue(dropTargetElement)

    if (!element) {
      return
    }

    cleanup = dropTargetForElements({
      element,
      ...options
    })
  })

  onBeforeUnmount(() => {
    cleanup()
  })
}

export function usePragmaticDraggable(
  draggableElement: MaybeRefOrGetter<HTMLElement | null>,
  options: Omit<Parameters<typeof draggable>[0], 'element'>
) {
  let cleanup = () => {}

  onMounted(() => {
    const element = toValue(draggableElement)

    if (!element) {
      return
    }

    cleanup = draggable({
      element,
      ...options
    })
    // TODO: Change to onScopeDispose
  })

  onBeforeUnmount(() => {
    cleanup()
  })
}
