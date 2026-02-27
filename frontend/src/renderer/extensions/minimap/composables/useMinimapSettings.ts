import { computed } from 'vue'

import { useSettingStore } from '@/platform/settings/settingStore'
import { useColorPaletteStore } from '@/stores/workspace/colorPaletteStore'

/**
 * Composable for minimap configuration options that are set by the user in the
 * settings. Provides reactive computed properties for the settings.
 */
export function useMinimapSettings() {
  const settingStore = useSettingStore()
  const colorPaletteStore = useColorPaletteStore()

  const nodeColors = computed(() =>
    settingStore.get('Comfy.Minimap.NodeColors')
  )
  const showLinks = computed(() => settingStore.get('Comfy.Minimap.ShowLinks'))
  const showGroups = computed(() =>
    settingStore.get('Comfy.Minimap.ShowGroups')
  )
  const renderBypass = computed(() =>
    settingStore.get('Comfy.Minimap.RenderBypassState')
  )
  const renderError = computed(() =>
    settingStore.get('Comfy.Minimap.RenderErrorState')
  )

  const width = 253
  const height = 200

  // Theme-aware colors
  const isLightTheme = computed(
    () => colorPaletteStore.completedActivePalette.light_theme
  )

  const containerStyles = computed(() => ({
    width: `${width}px`,
    height: `${height}px`,
    border: '1px solid var(--interface-stroke)',
    borderRadius: '8px'
  }))

  const panelStyles = computed(() => ({
    width: `210px`,
    height: `${height}px`,
    border: '1px solid var(--interface-stroke)',
    borderRadius: '8px'
  }))

  return {
    nodeColors,
    showLinks,
    showGroups,
    renderBypass,
    renderError,
    containerStyles,
    panelStyles,
    isLightTheme
  }
}
