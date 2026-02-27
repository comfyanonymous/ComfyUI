import { defineStore } from 'pinia'
import { computed, ref } from 'vue'

import type { ServerConfig, ServerConfigValue } from '@/constants/serverConfig'

type ServerConfigWithValue<T> = ServerConfig<T> & {
  /**
   * Current value.
   */
  value: T
  /**
   * Initial value loaded from settings.
   */
  initialValue: T
}

export const useServerConfigStore = defineStore('serverConfig', () => {
  const serverConfigById = ref<
    Record<string, ServerConfigWithValue<ServerConfigValue>>
  >({})
  const serverConfigs = computed(() => {
    return Object.values(serverConfigById.value)
  })
  const modifiedConfigs = computed<ServerConfigWithValue<ServerConfigValue>[]>(
    () => {
      return serverConfigs.value.filter((config) => {
        return config.initialValue !== config.value
      })
    }
  )
  const revertChanges = () => {
    for (const config of modifiedConfigs.value) {
      config.value = config.initialValue
    }
  }
  const serverConfigsByCategory = computed<
    Record<string, ServerConfigWithValue<ServerConfigValue>[]>
  >(() => {
    return serverConfigs.value.reduce(
      (acc, config) => {
        const category = config.category?.[0] ?? 'General'
        acc[category] = acc[category] || []
        acc[category].push(config)
        return acc
      },
      {} as Record<string, ServerConfigWithValue<ServerConfigValue>[]>
    )
  })
  const serverConfigValues = computed<Record<string, ServerConfigValue>>(() => {
    return Object.fromEntries(
      serverConfigs.value.map((config) => {
        return [
          config.id,
          config.value === config.defaultValue ||
          config.value === null ||
          config.value === undefined
            ? undefined
            : config.value
        ]
      })
    )
  })
  const launchArgs = computed<Record<string, string>>(() => {
    const args: Record<
      string,
      Omit<ServerConfigValue, 'undefined' | 'null'>
    > = Object.assign(
      {},
      ...serverConfigs.value.map((config) => {
        // Filter out configs that have the default value or undefined | null value
        if (
          config.value === config.defaultValue ||
          config.value === null ||
          config.value === undefined
        ) {
          return {}
        }
        return config.getValue
          ? config.getValue(config.value)
          : { [config.id]: config.value }
      })
    )

    // Convert true to empty string
    // Convert number to string
    return Object.fromEntries(
      Object.entries(args).map(([key, value]) => {
        if (value === true) {
          return [key, '']
        }
        return [key, value.toString()]
      })
    ) as Record<string, string>
  })
  const commandLineArgs = computed<string>(() => {
    return Object.entries(launchArgs.value)
      .map(([key, value]) => [`--${key}`, value])
      .flat()
      .filter((arg: string) => arg !== '')
      .join(' ')
  })

  function loadServerConfig(
    configs: ServerConfig<ServerConfigValue>[],
    values: Record<string, ServerConfigValue>
  ) {
    for (const config of configs) {
      const value = values[config.id] ?? config.defaultValue
      serverConfigById.value[config.id] = {
        ...config,
        value,
        initialValue: value
      }
    }
  }

  return {
    serverConfigById,
    serverConfigs,
    modifiedConfigs,
    serverConfigsByCategory,
    serverConfigValues,
    launchArgs,
    commandLineArgs,
    revertChanges,
    loadServerConfig
  }
})
