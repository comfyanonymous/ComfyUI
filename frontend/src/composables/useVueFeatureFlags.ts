/**
 * Vue-related feature flags composable
 * Manages local settings-driven flags and LiteGraph integration
 */
import { createSharedComposable } from '@vueuse/core'
import { computed, watch } from 'vue'

import { useSettingStore } from '@/platform/settings/settingStore'

import { LiteGraph } from '../lib/litegraph/src/litegraph'

function useVueFeatureFlagsIndividual() {
  const settingStore = useSettingStore()

  const shouldRenderVueNodes = computed(() => {
    try {
      return settingStore.get('Comfy.VueNodes.Enabled') ?? false
    } catch {
      return false
    }
  })

  // Watch for changes and update LiteGraph immediately
  watch(
    shouldRenderVueNodes,
    () => {
      LiteGraph.vueNodesMode = shouldRenderVueNodes.value
    },
    { immediate: true }
  )

  return {
    shouldRenderVueNodes
  }
}

export const useVueFeatureFlags = createSharedComposable(
  useVueFeatureFlagsIndividual
)
