import { useMouse } from '@vueuse/core'
import { defineStore } from 'pinia'
import { computed, ref, shallowRef } from 'vue'

import type NodeSearchBoxPopover from '@/components/searchbox/NodeSearchBoxPopover.vue'
import type { CanvasPointerEvent } from '@/lib/litegraph/src/litegraph'
import { useSettingStore } from '@/platform/settings/settingStore'

export const useSearchBoxStore = defineStore('searchBox', () => {
  const settingStore = useSettingStore()
  const { x, y } = useMouse()

  const useSearchBoxV2 = computed(
    () => settingStore.get('Comfy.NodeSearchBoxImpl') === 'default'
  )

  const newSearchBoxEnabled = computed(
    () => settingStore.get('Comfy.NodeSearchBoxImpl') !== 'litegraph (legacy)'
  )

  const popoverRef = shallowRef<InstanceType<
    typeof NodeSearchBoxPopover
  > | null>(null)

  function setPopoverRef(
    popover: InstanceType<typeof NodeSearchBoxPopover> | null
  ) {
    popoverRef.value = popover
  }

  const visible = ref(false)
  function toggleVisible() {
    if (newSearchBoxEnabled.value) {
      visible.value = !visible.value
      return
    }
    if (!popoverRef.value) return
    popoverRef.value.showSearchBox(
      new MouseEvent('click', {
        clientX: x.value,
        clientY: y.value,
        // @ts-expect-error layerY is a nonstandard property
        layerY: y.value
      }) as unknown as CanvasPointerEvent
    )
  }

  return {
    useSearchBoxV2,
    newSearchBoxEnabled,
    setPopoverRef,
    toggleVisible,
    visible
  }
})
