import { until } from '@vueuse/core'
import { defineStore } from 'pinia'
import { compare, valid } from 'semver'
import { computed, ref } from 'vue'

import { isCloud, isDesktop } from '@/platform/distribution/types'
import { useSettingStore } from '@/platform/settings/settingStore'
import { useSystemStatsStore } from '@/stores/systemStatsStore'
import { stringToLocale } from '@/utils/formatUtil'

import { useReleaseService } from './releaseService'
import type { ReleaseNote } from './releaseService'

// Store for managing release notes
export const useReleaseStore = defineStore('release', () => {
  // State
  const releases = ref<ReleaseNote[]>([])
  const isLoading = ref(false)
  const error = ref<string | null>(null)

  // Services
  const releaseService = useReleaseService()
  const systemStatsStore = useSystemStatsStore()
  const settingStore = useSettingStore()

  const currentVersion = computed(() => {
    if (isCloud) {
      return systemStatsStore?.systemStats?.system?.cloud_version ?? ''
    }
    return systemStatsStore?.systemStats?.system?.comfyui_version ?? ''
  })

  // Release data from settings
  const locale = computed(() => settingStore.get('Comfy.Locale'))
  const releaseVersion = computed(() =>
    settingStore.get('Comfy.Release.Version')
  )
  const releaseStatus = computed(() => settingStore.get('Comfy.Release.Status'))
  const releaseTimestamp = computed(() =>
    settingStore.get('Comfy.Release.Timestamp')
  )
  const showVersionUpdates = computed(() =>
    settingStore.get('Comfy.Notification.ShowVersionUpdates')
  )

  // Most recent release
  const recentRelease = computed(() => {
    return releases.value[0] ?? null
  })

  // 3 most recent releases
  const recentReleases = computed(() => {
    return releases.value.slice(0, 3)
  })

  // Helper constants
  const THREE_DAYS_MS = 3 * 24 * 60 * 60 * 1000 // 3 days

  const compareVersions = (
    releaseVersion: string,
    currentVer: string
  ): number => {
    if (valid(releaseVersion) && valid(currentVer)) {
      return compare(releaseVersion, currentVer)
    }
    // Non-semver (e.g. git hash): assume different = newer
    return releaseVersion === currentVer ? 0 : 1
  }

  // New version available?
  const isNewVersionAvailable = computed(
    () =>
      !!recentRelease.value &&
      compareVersions(
        recentRelease.value.version,
        currentVersion.value || '0.0.0'
      ) > 0
  )

  const isLatestVersion = computed(
    () =>
      !!recentRelease.value &&
      compareVersions(
        recentRelease.value.version,
        currentVersion.value || '0.0.0'
      ) === 0
  )

  const hasMediumOrHighAttention = computed(() => {
    const attention = recentRelease.value?.attention
    return attention === 'medium' || attention === 'high'
  })

  // Show toast if needed
  const shouldShowToast = computed(() => {
    // Only show on desktop version
    if (!isDesktop || isCloud) {
      return false
    }

    // Skip if notifications are disabled
    if (!showVersionUpdates.value) {
      return false
    }

    if (!isNewVersionAvailable.value) {
      return false
    }

    // Skip if low attention
    if (!hasMediumOrHighAttention.value) {
      return false
    }

    // Skip if user already skipped or changelog seen
    if (
      releaseVersion.value === recentRelease.value?.version &&
      ['skipped', 'changelog seen'].includes(releaseStatus.value)
    ) {
      return false
    }

    return true
  })

  // Show red-dot indicator
  const shouldShowRedDot = computed(() => {
    // Only show on desktop version
    if (!isDesktop || isCloud) {
      return false
    }

    // Skip if notifications are disabled
    if (!showVersionUpdates.value) {
      return false
    }

    // Already latest → no dot
    if (!isNewVersionAvailable.value) {
      return false
    }

    const { version } = recentRelease.value

    // Changelog seen → clear dot
    if (
      releaseVersion.value === version &&
      releaseStatus.value === 'changelog seen'
    ) {
      return false
    }

    // Attention medium / high (levels 2 & 3)
    if (hasMediumOrHighAttention.value) {
      // Persist until changelog is opened
      return true
    }

    // Attention low (level 1) and skipped → keep up to 3 d
    if (
      releaseVersion.value === version &&
      releaseStatus.value === 'skipped' &&
      releaseTimestamp.value &&
      Date.now() - releaseTimestamp.value >= THREE_DAYS_MS
    ) {
      return false
    }

    // Not skipped → show
    return true
  })

  const shouldShowPopup = computed(() => {
    if (!isDesktop && !isCloud) {
      return false
    }

    if (!showVersionUpdates.value) {
      return false
    }

    if (!recentRelease.value) {
      return false
    }

    // Skip version check if current version isn't semver (e.g. git hash)
    const skipVersionCheck = !valid(currentVersion.value)
    if (!skipVersionCheck && !isLatestVersion.value) {
      return false
    }

    if (
      releaseVersion.value === recentRelease.value.version &&
      releaseStatus.value === "what's new seen"
    ) {
      return false
    }

    return true
  })

  // Action handlers for user interactions
  async function handleSkipRelease(version: string): Promise<void> {
    if (
      version !== recentRelease.value?.version ||
      releaseStatus.value === 'changelog seen'
    ) {
      return
    }

    await settingStore.setMany({
      'Comfy.Release.Version': version,
      'Comfy.Release.Status': 'skipped',
      'Comfy.Release.Timestamp': Date.now()
    })
  }

  async function handleShowChangelog(version: string): Promise<void> {
    if (version !== recentRelease.value?.version) {
      return
    }

    await settingStore.setMany({
      'Comfy.Release.Version': version,
      'Comfy.Release.Status': 'changelog seen',
      'Comfy.Release.Timestamp': Date.now()
    })
  }

  async function handleWhatsNewSeen(version: string): Promise<void> {
    if (version !== recentRelease.value?.version) {
      return
    }

    await settingStore.setMany({
      'Comfy.Release.Version': version,
      'Comfy.Release.Status': "what's new seen",
      'Comfy.Release.Timestamp': Date.now()
    })
  }

  // Fetch releases from API
  async function fetchReleases(): Promise<void> {
    if (isLoading.value) {
      return
    }

    if (!isCloud && !showVersionUpdates.value) {
      return
    }

    // Skip fetching if API nodes are disabled via argv
    if (
      systemStatsStore.systemStats?.system?.argv?.includes(
        '--disable-api-nodes'
      )
    ) {
      return
    }
    isLoading.value = true
    error.value = null

    try {
      // Ensure system stats are loaded
      if (!systemStatsStore.systemStats) {
        await until(systemStatsStore.isInitialized)
      }

      const fetchedReleases = await releaseService.getReleases({
        project: isCloud ? 'cloud' : 'comfyui',
        current_version: currentVersion.value,
        form_factor: systemStatsStore.getFormFactor(),
        locale: stringToLocale(locale.value)
      })

      if (fetchedReleases !== null) {
        releases.value = fetchedReleases
      } else if (releaseService.error.value) {
        error.value = releaseService.error.value
      }
    } catch (err) {
      error.value =
        err instanceof Error ? err.message : 'Unknown error occurred'
    } finally {
      isLoading.value = false
    }
  }

  // Initialize store
  async function initialize(): Promise<void> {
    await fetchReleases()
  }

  return {
    releases,
    isLoading,
    error,
    recentRelease,
    recentReleases,
    shouldShowToast,
    shouldShowRedDot,
    shouldShowPopup,
    shouldShowUpdateButton: isNewVersionAvailable,
    handleSkipRelease,
    handleShowChangelog,
    handleWhatsNewSeen,
    fetchReleases,
    initialize
  }
})
