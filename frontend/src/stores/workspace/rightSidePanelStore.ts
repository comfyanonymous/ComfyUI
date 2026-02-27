import { defineStore } from 'pinia'
import { computed, ref, watch } from 'vue'

import { useSettingStore } from '@/platform/settings/settingStore'

export type RightSidePanelTab =
  | 'parameters'
  | 'nodes'
  | 'settings'
  | 'info'
  | 'subgraph'
  | 'errors'

type RightSidePanelSection = 'advanced-inputs' | string

/**
 * Store for managing the right side panel state.
 * This panel displays properties and settings for selected nodes.
 */
export const useRightSidePanelStore = defineStore('rightSidePanel', () => {
  const settingStore = useSettingStore()

  const isLegacyMenu = computed(
    () => settingStore.get('Comfy.UseNewMenu') === 'Disabled'
  )

  const isOpen = computed({
    get: () =>
      !isLegacyMenu.value && settingStore.get('Comfy.RightSidePanel.IsOpen'),
    set: (value: boolean) =>
      settingStore.set('Comfy.RightSidePanel.IsOpen', value)
  })
  const activeTab = ref<RightSidePanelTab>('parameters')
  const isEditingSubgraph = computed(() => activeTab.value === 'subgraph')
  const focusedSection = ref<RightSidePanelSection | null>(null)
  /**
   * Graph node ID to focus in the errors tab.
   * Set by SectionWidgets when the user clicks "See Error", consumed and
   * cleared by TabErrors after expanding the relevant error group.
   */
  const focusedErrorNodeId = ref<string | null>(null)
  const searchQuery = ref('')

  // Auto-close panel when switching to legacy menu mode
  watch(isLegacyMenu, (legacy) => {
    if (legacy) {
      void settingStore.set('Comfy.RightSidePanel.IsOpen', false)
    }
  })

  function openPanel(tab?: RightSidePanelTab) {
    if (isLegacyMenu.value) return
    isOpen.value = true
    if (tab) {
      activeTab.value = tab
    }
  }

  function closePanel() {
    isOpen.value = false
  }

  function togglePanel() {
    isOpen.value = !isOpen.value
  }

  /**
   * Focus on a specific section in the right side panel.
   * This will open the panel, switch to the parameters tab, and signal
   * the component to expand and scroll to the section.
   */
  function focusSection(section: RightSidePanelSection) {
    openPanel('parameters')
    focusedSection.value = section
  }

  /**
   * Clear the focused section after it has been handled.
   */
  function clearFocusedSection() {
    focusedSection.value = null
  }

  return {
    isOpen,
    activeTab,
    isEditingSubgraph,
    focusedSection,
    focusedErrorNodeId,
    searchQuery,
    openPanel,
    closePanel,
    togglePanel,
    focusSection,
    clearFocusedSection
  }
})
