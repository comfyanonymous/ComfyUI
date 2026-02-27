import { retry } from 'es-toolkit'
import _ from 'es-toolkit/compat'
import { until, useAsyncState } from '@vueuse/core'
import { defineStore } from 'pinia'
import { compare, valid } from 'semver'
import { ref } from 'vue'

import type { SettingParams } from '@/platform/settings/types'
import type { Settings } from '@/schemas/apiSchema'
import { api } from '@/scripts/api'
import { app } from '@/scripts/app'
import type { TreeNode } from '@/types/treeExplorerTypes'

export const getSettingInfo = (setting: SettingParams) => {
  const parts = setting.category || setting.id.split('.')
  return {
    category: parts[0] ?? 'Other',
    subCategory: parts[1] ?? 'Other'
  }
}

export interface SettingTreeNode extends TreeNode {
  data?: SettingParams
}

function tryMigrateDeprecatedValue(
  setting: SettingParams | undefined,
  value: unknown
) {
  return setting?.migrateDeprecatedValue?.(value) ?? value
}

function onChange(
  setting: SettingParams | undefined,
  newValue: unknown,
  oldValue: unknown
) {
  if (setting?.onChange) {
    setting.onChange(newValue, oldValue)
  }
  // Backward compatibility with old settings dialog.
  // Some extensions still listens event emitted by the old settings dialog.
  if (setting) {
    app.ui.settings.dispatchChange(setting.id, newValue, oldValue)
  }
}

export const useSettingStore = defineStore('setting', () => {
  const settingValues = ref<Partial<Settings>>({})
  const settingsById = ref<Record<string, SettingParams>>({})

  const {
    isReady,
    isLoading,
    error,
    execute: loadSettingValues
  } = useAsyncState(
    async () => {
      if (Object.keys(settingsById.value).length) {
        throw new Error(
          'Setting values must be loaded before any setting is registered.'
        )
      }
      settingValues.value = await retry(() => api.getSettings(), {
        retries: 3,
        delay: (attempt) => Math.min(1000 * Math.pow(2, attempt), 8000)
      })
      await migrateZoomThresholdToFontSize()
    },
    undefined,
    { immediate: false }
  )

  async function load(): Promise<void> {
    if (isReady.value) return

    if (isLoading.value) {
      await until(isLoading).toBe(false)
      return
    }

    await loadSettingValues()
  }

  /**
   * Check if a setting's value exists, i.e. if the user has set it manually.
   * @param key - The key of the setting to check.
   * @returns Whether the setting exists.
   */
  function exists<K extends keyof Settings>(key: K) {
    return settingValues.value[key] !== undefined
  }

  /**
   * Apply a setting value locally: clone, migrate, fire onChange, and
   * update the in-memory store. Returns the migrated value, or
   * `undefined` when the value is unchanged and was skipped.
   */
  function applySettingLocally<K extends keyof Settings>(
    key: K,
    value: Settings[K]
  ): Settings[K] | undefined {
    const clonedValue = _.cloneDeep(value)
    const newValue = tryMigrateDeprecatedValue(
      settingsById.value[key],
      clonedValue
    )
    const oldValue = get(key)
    if (newValue === oldValue) return undefined

    onChange(settingsById.value[key], newValue, oldValue)
    settingValues.value[key] = newValue
    return newValue as Settings[K]
  }

  /**
   * Set a setting value.
   * @param key - The key of the setting to set.
   * @param value - The value to set.
   */
  async function set<K extends keyof Settings>(key: K, value: Settings[K]) {
    const applied = applySettingLocally(key, value)
    if (applied === undefined) return
    await api.storeSetting(key, applied)
  }

  /**
   * Set multiple setting values in a single API call.
   * @param settings - A partial settings object with key-value pairs to set.
   */
  async function setMany(settings: Partial<Settings>) {
    const updatedSettings: Partial<Settings> = {}

    for (const key of Object.keys(settings) as (keyof Settings)[]) {
      const applied = applySettingLocally(
        key,
        settings[key] as Settings[typeof key]
      )
      if (applied !== undefined) {
        updatedSettings[key] = applied
      }
    }

    if (Object.keys(updatedSettings).length > 0) {
      await api.storeSettings(updatedSettings)
    }
  }

  /**
   * Get a setting value.
   * @param key - The key of the setting to get.
   * @returns The value of the setting.
   */
  function get<K extends keyof Settings>(key: K): Settings[K] {
    // Clone the value when returning to prevent external mutations
    return _.cloneDeep(settingValues.value[key] ?? getDefaultValue(key)!)
  }

  /**
   * Gets the setting params, asserting the type that is intentionally left off
   * of {@link settingsById}.
   * @param key The key of the setting to get.
   * @returns The setting.
   */
  function getSettingById<K extends keyof Settings>(
    key: K
  ): SettingParams<Settings[K]> | undefined {
    return settingsById.value[key] as SettingParams<Settings[K]> | undefined
  }

  /**
   * Get the default value of a setting.
   * @param key - The key of the setting to get.
   * @returns The default value of the setting.
   */
  function getDefaultValue<K extends keyof Settings>(
    key: K
  ): Settings[K] | undefined {
    // Assertion: settingsById is not typed.
    const param = getSettingById(key)

    if (param === undefined) return

    const versionedDefault = getVersionedDefaultValue(key, param)

    if (versionedDefault) {
      return versionedDefault
    }

    const defaultValue = param.defaultValue
    return typeof defaultValue === 'function'
      ? (defaultValue as () => Settings[K])()
      : defaultValue
  }

  function getVersionedDefaultValue<
    K extends keyof Settings,
    TValue = Settings[K]
  >(key: K, param: SettingParams<TValue> | undefined): TValue | null {
    // get default versioned value, skipping if the key is 'Comfy.InstalledVersion' to prevent infinite loop
    const defaultsByInstallVersion = param?.defaultsByInstallVersion
    if (defaultsByInstallVersion && key !== 'Comfy.InstalledVersion') {
      const installedVersion = get('Comfy.InstalledVersion')

      if (installedVersion) {
        const sortedVersions = Object.keys(defaultsByInstallVersion).sort(
          (a, b) => compare(b, a)
        )

        for (const version of sortedVersions) {
          // Ensure the version is in a valid format before comparing
          if (!valid(version)) {
            continue
          }

          if (compare(installedVersion, version) >= 0) {
            const versionedDefault =
              defaultsByInstallVersion[
                version as keyof typeof defaultsByInstallVersion
              ]
            if (versionedDefault !== undefined) {
              return typeof versionedDefault === 'function'
                ? versionedDefault()
                : versionedDefault
            }
          }
        }
      }
    }

    return null
  }

  /**
   * Register a setting.
   * @param setting - The setting to register.
   */
  function addSetting(setting: SettingParams) {
    if (!setting.id) {
      throw new Error('Settings must have an ID')
    }
    if (setting.id in settingsById.value) {
      // Setting already registered - skip to allow component remounting
      // TODO: Add store reset methods to bootstrapStore and settingStore, then
      // replace window.location.reload() with router.push() in SidebarLogoutIcon.vue
      console.warn(`Setting already registered: ${setting.id}`)
      return
    }

    settingsById.value[setting.id] = setting

    if (settingValues.value[setting.id] !== undefined) {
      settingValues.value[setting.id] = tryMigrateDeprecatedValue(
        setting,
        settingValues.value[setting.id]
      )
    }
    onChange(setting, get(setting.id), undefined)
  }

  /**
   * Migrate the old zoom threshold setting to the new font size setting.
   * Preserves the exact zoom threshold behavior by converting it to equivalent font size.
   */
  async function migrateZoomThresholdToFontSize() {
    const oldKey = 'LiteGraph.Canvas.LowQualityRenderingZoomThreshold'
    const newKey = 'LiteGraph.Canvas.MinFontSizeForLOD'

    // Only migrate if old setting exists and new setting doesn't
    if (
      settingValues.value[oldKey] !== undefined &&
      settingValues.value[newKey] === undefined
    ) {
      const oldValue = settingValues.value[oldKey] as number

      // Convert zoom threshold to equivalent font size to preserve exact behavior
      // The threshold formula is: threshold = font_size / (14 * sqrt(DPR))
      // For DPR=1: threshold = font_size / 14
      // Therefore: font_size = threshold * 14
      //
      // Examples:
      // - Old 0.6 threshold → 0.6 * 14 = 8.4px → rounds to 8px (preserves ~60% zoom threshold)
      // - Old 0.5 threshold → 0.5 * 14 = 7px (preserves 50% zoom threshold)
      // - Old 1.0 threshold → 1.0 * 14 = 14px (preserves 100% zoom threshold)
      const mappedFontSize = Math.round(oldValue * 14)
      const clampedFontSize = Math.max(1, Math.min(24, mappedFontSize))

      // Set the new value
      settingValues.value[newKey] = clampedFontSize

      // Remove the old setting to prevent confusion
      delete settingValues.value[oldKey]

      // Store the migrated setting
      await api.storeSetting(newKey, clampedFontSize)
      await api.storeSetting(oldKey, undefined)
    }
  }

  return {
    settingValues,
    settingsById,
    isReady,
    isLoading,
    error,
    load,
    addSetting,
    set,
    setMany,
    get,
    exists,
    getDefaultValue
  }
})
