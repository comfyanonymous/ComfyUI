import { defineStore } from 'pinia'
import { ref } from 'vue'

export type HelpCenterTriggerLocation = 'sidebar' | 'topbar'

export const useHelpCenterStore = defineStore('helpCenter', () => {
  const isVisible = ref(false)
  const triggerLocation = ref<HelpCenterTriggerLocation>('sidebar')

  const toggle = (location: HelpCenterTriggerLocation = 'sidebar') => {
    if (!isVisible.value) {
      triggerLocation.value = location
    }
    isVisible.value = !isVisible.value
  }

  const show = (location: HelpCenterTriggerLocation = 'sidebar') => {
    triggerLocation.value = location
    isVisible.value = true
  }

  const hide = () => {
    isVisible.value = false
  }

  return {
    isVisible,
    triggerLocation,
    toggle,
    show,
    hide
  }
})
