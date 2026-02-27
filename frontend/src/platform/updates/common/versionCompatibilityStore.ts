import { until, useStorage } from '@vueuse/core'
import { defineStore } from 'pinia'
import { gt, valid } from 'semver'
import { computed } from 'vue'

import config from '@/config'
import { useSettingStore } from '@/platform/settings/settingStore'
import { useSystemStatsStore } from '@/stores/systemStatsStore'

const DISMISSAL_DURATION_MS = 7 * 24 * 60 * 60 * 1000 // 7 days

export const useVersionCompatibilityStore = defineStore(
  'versionCompatibility',
  () => {
    const systemStatsStore = useSystemStatsStore()
    const settingStore = useSettingStore()

    const frontendVersion = computed(() => config.app_version)
    const backendVersion = computed(
      () => systemStatsStore.systemStats?.system?.comfyui_version ?? ''
    )
    const requiredFrontendVersion = computed(
      () =>
        systemStatsStore.systemStats?.system?.required_frontend_version ?? ''
    )

    const isFrontendOutdated = computed(() => {
      if (
        !frontendVersion.value ||
        !requiredFrontendVersion.value ||
        !valid(frontendVersion.value) ||
        !valid(requiredFrontendVersion.value)
      ) {
        return false
      }
      // Returns true if required version is greater than frontend version
      return gt(requiredFrontendVersion.value, frontendVersion.value)
    })

    const isFrontendNewer = computed(() => {
      // We don't warn about frontend being newer than backend
      // Only warn when frontend is outdated (behind required version)
      return false
    })

    const hasVersionMismatch = computed(() => {
      return isFrontendOutdated.value
    })

    const versionKey = computed(() => {
      if (
        !frontendVersion.value ||
        !backendVersion.value ||
        !requiredFrontendVersion.value
      ) {
        return null
      }
      return `${frontendVersion.value}-${backendVersion.value}-${requiredFrontendVersion.value}`
    })

    // Use reactive storage for dismissals - creates a reactive ref that syncs with localStorage
    // All version mismatch dismissals are stored in a single object for clean localStorage organization
    const dismissalStorage = useStorage(
      'comfy.versionMismatch.dismissals',
      {} as Record<string, number>,
      localStorage,
      {
        serializer: {
          read: (value: string) => {
            try {
              return JSON.parse(value)
            } catch {
              return {}
            }
          },
          write: (value: Record<string, number>) => JSON.stringify(value)
        }
      }
    )

    const isDismissed = computed(() => {
      if (!versionKey.value) return false

      const dismissedUntil = dismissalStorage.value[versionKey.value]
      if (!dismissedUntil) return false

      // Check if dismissal has expired
      return Date.now() < dismissedUntil
    })

    const warningsDisabled = computed(() =>
      settingStore.get('Comfy.VersionCompatibility.DisableWarnings')
    )

    const shouldShowWarning = computed(() => {
      return (
        hasVersionMismatch.value &&
        !isDismissed.value &&
        !warningsDisabled.value
      )
    })

    const warningMessage = computed(() => {
      if (isFrontendOutdated.value) {
        return {
          type: 'outdated' as const,
          frontendVersion: frontendVersion.value,
          requiredVersion: requiredFrontendVersion.value
        }
      }
      return null
    })

    async function checkVersionCompatibility() {
      if (!systemStatsStore.systemStats) {
        await until(systemStatsStore.isInitialized)
      }
    }

    function dismissWarning() {
      if (!versionKey.value) return

      const dismissUntil = Date.now() + DISMISSAL_DURATION_MS
      dismissalStorage.value = {
        ...dismissalStorage.value,
        [versionKey.value]: dismissUntil
      }
    }

    async function initialize() {
      await checkVersionCompatibility()
    }

    return {
      frontendVersion,
      backendVersion,
      requiredFrontendVersion,
      hasVersionMismatch,
      shouldShowWarning,
      warningMessage,
      isFrontendOutdated,
      isFrontendNewer,
      checkVersionCompatibility,
      dismissWarning,
      initialize
    }
  }
)
