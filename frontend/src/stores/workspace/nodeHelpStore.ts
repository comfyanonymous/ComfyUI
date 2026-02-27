import { defineStore } from 'pinia'
import { computed, ref } from 'vue'

import type { ComfyNodeDefImpl } from '@/stores/nodeDefStore'

export const useNodeHelpStore = defineStore('nodeHelp', () => {
  const currentHelpNode = ref<ComfyNodeDefImpl | null>(null)
  const isHelpOpen = computed(() => currentHelpNode.value !== null)

  function openHelp(nodeDef: ComfyNodeDefImpl) {
    currentHelpNode.value = nodeDef
  }

  function closeHelp() {
    currentHelpNode.value = null
  }

  return {
    currentHelpNode,
    isHelpOpen,
    openHelp,
    closeHelp
  }
})
