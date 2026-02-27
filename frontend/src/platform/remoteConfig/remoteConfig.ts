/**
 * Remote configuration service
 *
 * Fetches configuration from the server at runtime, enabling:
 * - Feature flags without rebuilding
 * - Server-side feature discovery
 * - Version compatibility management
 * - Avoiding vendor lock-in for native apps
 *
 * This module is tree-shaken in OSS builds.
 */

import { computed, ref } from 'vue'

import type { RemoteConfig } from './types'

/**
 * Load state for remote configuration.
 * - 'unloaded': No config loaded yet
 * - 'anonymous': Config loaded without auth (bootstrap)
 * - 'authenticated': Config loaded with auth (user-specific flags available)
 * - 'error': Failed to load config
 */
type RemoteConfigState = 'unloaded' | 'anonymous' | 'authenticated' | 'error'

/**
 * Current load state of remote configuration
 */
export const remoteConfigState = ref<RemoteConfigState>('unloaded')

/**
 * Whether the authenticated config has been loaded.
 * Use this to gate access to user-specific feature flags like teamWorkspacesEnabled.
 */
export const isAuthenticatedConfigLoaded = computed(
  () => remoteConfigState.value === 'authenticated'
)

/**
 * Reactive remote configuration
 * Updated whenever config is loaded from the server
 */
export const remoteConfig = ref<RemoteConfig>({})

export function configValueOrDefault<K extends keyof RemoteConfig>(
  remoteConfig: RemoteConfig,
  key: K,
  defaultValue: NonNullable<RemoteConfig[K]>
): NonNullable<RemoteConfig[K]> {
  const configValue = remoteConfig[key]
  return configValue || defaultValue
}
