import { storeToRefs } from 'pinia'
import { computed, readonly } from 'vue'

import { t } from '@/i18n'
import { useToastStore } from '@/platform/updates/common/toastStore'
import { api } from '@/scripts/api'
import { useSettingsDialog } from '@/platform/settings/composables/useSettingsDialog'
import { useCommandStore } from '@/stores/commandStore'
import { useSystemStatsStore } from '@/stores/systemStatsStore'
import { useManagerDialog } from '@/workbench/extensions/manager/composables/useManagerDialog'
import type { ManagerTab } from '@/workbench/extensions/manager/types/comfyManagerTypes'

export enum ManagerUIState {
  DISABLED = 'disabled',
  LEGACY_UI = 'legacy',
  NEW_UI = 'new'
}

export function useManagerState() {
  const systemStatsStore = useSystemStatsStore()
  const { systemStats, isInitialized: systemInitialized } =
    storeToRefs(systemStatsStore)
  const managerDialog = useManagerDialog()

  /**
   * The current manager UI state.
   * Computed once and cached until dependencies change (which they don't during runtime).
   * This follows Vue's conventions and provides better performance through caching.
   */
  const managerUIState = readonly(
    computed((): ManagerUIState => {
      // Wait for systemStats to be initialized
      if (!systemInitialized.value) {
        // Default to DISABLED while loading
        return ManagerUIState.DISABLED
      }

      // Get current values
      const clientSupportsV4 =
        api.getClientFeatureFlags().supports_manager_v4_ui ?? false

      const serverSupportsV4 = api.getServerFeature(
        'extension.manager.supports_v4'
      )

      // Check command line args first (highest priority)
      // --enable-manager flag enables the manager (opposite of old --disable-manager)
      const hasEnableManager =
        systemStats.value?.system?.argv?.includes('--enable-manager')

      // If --enable-manager is NOT present, manager is disabled
      if (!hasEnableManager) {
        return ManagerUIState.DISABLED
      }

      if (
        systemStats.value?.system?.argv?.includes('--enable-manager-legacy-ui')
      ) {
        return ManagerUIState.LEGACY_UI
      }

      // Both client and server support v4 = NEW_UI
      if (clientSupportsV4 && serverSupportsV4 === true) {
        return ManagerUIState.NEW_UI
      }

      // Server supports v4 but client doesn't = LEGACY_UI
      if (serverSupportsV4 === true && !clientSupportsV4) {
        return ManagerUIState.LEGACY_UI
      }

      // Server explicitly doesn't support v4 = LEGACY_UI
      if (serverSupportsV4 === false) {
        return ManagerUIState.LEGACY_UI
      }

      // If server feature flags haven't loaded yet, default to NEW_UI
      // This is a temporary state - feature flags are exchanged immediately on WebSocket connection
      // NEW_UI is the safest default since v2 API is the current standard
      // If the server doesn't support v2, API calls will fail with 404 and be handled gracefully
      if (serverSupportsV4 === undefined) {
        return ManagerUIState.NEW_UI
      }

      // Should never reach here, but if we do, disable manager
      return ManagerUIState.DISABLED
    })
  )

  /**
   * Check if manager is enabled (not DISABLED)
   */
  const isManagerEnabled = readonly(
    computed((): boolean => {
      return managerUIState.value !== ManagerUIState.DISABLED
    })
  )

  /**
   * Check if manager UI is in NEW_UI mode
   */
  const isNewManagerUI = readonly(
    computed((): boolean => {
      return managerUIState.value === ManagerUIState.NEW_UI
    })
  )

  /**
   * Check if manager UI is in LEGACY_UI mode
   */
  const isLegacyManagerUI = readonly(
    computed((): boolean => {
      return managerUIState.value === ManagerUIState.LEGACY_UI
    })
  )

  /**
   * Check if install button should be shown (only in NEW_UI mode)
   */
  const shouldShowInstallButton = readonly(
    computed((): boolean => {
      return isNewManagerUI.value
    })
  )

  /**
   * Check if manager buttons should be shown (when manager is not disabled)
   */
  const shouldShowManagerButtons = readonly(
    computed((): boolean => {
      return isManagerEnabled.value
    })
  )

  /**
   * Opens the manager UI based on current state
   * Centralizes the logic for opening manager across the app
   * @param options - Optional configuration for opening the manager
   * @param options.initialTab - Initial tab to show (for NEW_UI mode)
   * @param options.legacyCommand - Legacy command to execute (for LEGACY_UI mode)
   * @param options.showToastOnLegacyError - Whether to show toast on legacy command failure
   * @param options.isLegacyOnly - If true, shows error in NEW_UI mode instead of opening manager
   */
  const openManager = async (options?: {
    initialTab?: ManagerTab
    initialPackId?: string
    legacyCommand?: string
    showToastOnLegacyError?: boolean
    isLegacyOnly?: boolean
  }): Promise<void> => {
    const state = managerUIState.value
    const settingsDialog = useSettingsDialog()
    const commandStore = useCommandStore()

    switch (state) {
      case ManagerUIState.DISABLED:
        settingsDialog.show('extension')
        break

      case ManagerUIState.LEGACY_UI: {
        const command =
          options?.legacyCommand || 'Comfy.Manager.Menu.ToggleVisibility'
        try {
          await commandStore.execute(command)
        } catch {
          // If legacy command doesn't exist
          if (options?.showToastOnLegacyError !== false) {
            useToastStore().add({
              severity: 'error',
              summary: t('g.error'),
              detail: t('manager.legacyMenuNotAvailable'),
              life: 3000
            })
          }
          // Fallback to extensions panel if not showing toast
          if (options?.showToastOnLegacyError === false) {
            settingsDialog.show('extension')
          }
        }
        break
      }

      case ManagerUIState.NEW_UI:
        if (options?.isLegacyOnly) {
          useToastStore().add({
            severity: 'error',
            summary: t('g.error'),
            detail: t('manager.legacyMenuNotAvailable'),
            life: 3000
          })
        } else {
          managerDialog.show(options?.initialTab, options?.initialPackId)
        }
        break
    }
  }

  return {
    managerUIState,
    isManagerEnabled,
    isNewManagerUI,
    isLegacyManagerUI,
    shouldShowInstallButton,
    shouldShowManagerButtons,
    openManager
  }
}
