<template>
  <slot v-if="isReady" />
  <div
    v-else
    class="fixed inset-0 z-[1100] flex items-center justify-center bg-[var(--p-mask-background)]"
  >
    <ProgressSpinner />
  </div>
</template>

<script setup lang="ts">
/**
 * WorkspaceAuthGate - Conditional auth checkpoint for workspace mode.
 *
 * This gate ensures proper initialization order for workspace-scoped auth:
 * 1. Wait for Firebase auth to resolve
 * 2. Check if teamWorkspacesEnabled feature flag is on
 * 3. If YES: Initialize workspace token and store before rendering
 * 4. If NO: Render immediately using existing Firebase auth
 *
 * This prevents race conditions where API calls use Firebase tokens
 * instead of workspace tokens when the workspace feature is enabled.
 */
import { promiseTimeout, until } from '@vueuse/core'
import { storeToRefs } from 'pinia'
import ProgressSpinner from 'primevue/progressspinner'
import { onMounted, ref } from 'vue'

import { useFeatureFlags } from '@/composables/useFeatureFlags'
import { isCloud } from '@/platform/distribution/types'
import { refreshRemoteConfig } from '@/platform/remoteConfig/refreshRemoteConfig'
import { useTeamWorkspaceStore } from '@/platform/workspace/stores/teamWorkspaceStore'
import { useFirebaseAuthStore } from '@/stores/firebaseAuthStore'

const FIREBASE_INIT_TIMEOUT_MS = 16_000
const CONFIG_REFRESH_TIMEOUT_MS = 10_000

const isReady = ref(!isCloud)

async function initialize(): Promise<void> {
  if (!isCloud) return

  const authStore = useFirebaseAuthStore()
  const { isInitialized, currentUser } = storeToRefs(authStore)

  try {
    // Step 1: Wait for Firebase auth to resolve
    // This is shared with router guard - both wait for the same thing,
    // but this gate blocks rendering while router guard blocks navigation
    if (!isInitialized.value) {
      await until(isInitialized).toBe(true, {
        timeout: FIREBASE_INIT_TIMEOUT_MS
      })
    }

    // Step 2: If not authenticated, nothing more to do
    // Unauthenticated users don't have workspace context
    if (!currentUser.value) {
      isReady.value = true
      return
    }

    // Step 3: Refresh feature flags with auth context
    // This ensures teamWorkspacesEnabled reflects the authenticated user's state
    // Timeout prevents hanging if server is slow/unresponsive
    try {
      await Promise.race([
        refreshRemoteConfig({ useAuth: true }),
        promiseTimeout(CONFIG_REFRESH_TIMEOUT_MS).then(() => {
          throw new Error('Config refresh timeout')
        })
      ])
    } catch (error) {
      console.warn(
        '[WorkspaceAuthGate] Failed to refresh remote config:',
        error
      )
      // Continue - feature flags will use defaults (teamWorkspacesEnabled=false)
      // App will render with Firebase auth fallback
    }

    // Step 4: THE CHECKPOINT - Are we in workspace mode?
    const { flags } = useFeatureFlags()
    if (!flags.teamWorkspacesEnabled) {
      // Not in workspace mode - use existing Firebase auth flow
      // No additional initialization needed
      isReady.value = true
      return
    }

    // Step 5: WORKSPACE MODE - Full initialization
    await initializeWorkspaceMode()
  } catch (error) {
    console.error('[WorkspaceAuthGate] Initialization failed:', error)
  } finally {
    // Always render (graceful degradation)
    // If workspace init failed, API calls fall back to Firebase token
    isReady.value = true
  }
}

async function initializeWorkspaceMode(): Promise<void> {
  // Initialize the full workspace store which handles:
  // - Restoring workspace token from session (fast path for refresh)
  // - Fetching workspace list
  // - Switching to last used workspace if needed
  // - Setting active workspace
  try {
    const workspaceStore = useTeamWorkspaceStore()
    if (workspaceStore.initState === 'uninitialized') {
      await workspaceStore.initialize()
    }
  } catch (error) {
    // Log but don't block - workspace UI features may not work but app will render
    // API calls will fall back to Firebase token
    console.warn(
      '[WorkspaceAuthGate] Failed to initialize workspace store:',
      error
    )
  }
}

// Initialize on mount. This gate should be placed on the authenticated layout
// (LayoutDefault) so it mounts fresh after login and unmounts on logout.
// The router guard ensures only authenticated users reach this layout.
onMounted(() => {
  void initialize()
})
</script>
