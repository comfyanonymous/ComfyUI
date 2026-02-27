import type { Ref } from 'vue'
import { computed } from 'vue'

export function usePerTabState<K extends string, V>(
  selectedTab: Ref<K>,
  stateByTab: Ref<Record<K, V>>
) {
  return computed({
    get: () => stateByTab.value[selectedTab.value],
    set: (value) => {
      stateByTab.value[selectedTab.value] = value
    }
  })
}
