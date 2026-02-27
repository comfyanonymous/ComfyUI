import { defineStore } from 'pinia'
import { computed, ref, shallowRef } from 'vue'
import { z } from 'zod'
import { fromZodError } from 'zod-validation-error'

import { t } from '@/i18n'
import {
  TOKEN_REFRESH_BUFFER_MS,
  WORKSPACE_STORAGE_KEYS
} from '@/platform/workspace/workspaceConstants'
import { api } from '@/scripts/api'
import { useFirebaseAuthStore } from '@/stores/firebaseAuthStore'
import type { AuthHeader } from '@/types/authTypes'
import type { WorkspaceWithRole } from '@/platform/workspace/workspaceTypes'
import { useFeatureFlags } from '@/composables/useFeatureFlags'

const WorkspaceWithRoleSchema = z.object({
  id: z.string(),
  name: z.string(),
  type: z.enum(['personal', 'team']),
  role: z.enum(['owner', 'member'])
})

const WorkspaceTokenResponseSchema = z.object({
  token: z.string(),
  expires_at: z.string(),
  workspace: z.object({
    id: z.string(),
    name: z.string(),
    type: z.enum(['personal', 'team'])
  }),
  role: z.enum(['owner', 'member']),
  permissions: z.array(z.string())
})

export class WorkspaceAuthError extends Error {
  constructor(
    message: string,
    public readonly code?: string
  ) {
    super(message)
    this.name = 'WorkspaceAuthError'
  }
}

export const useWorkspaceAuthStore = defineStore('workspaceAuth', () => {
  const { flags } = useFeatureFlags()

  // State
  const currentWorkspace = shallowRef<WorkspaceWithRole | null>(null)
  const workspaceToken = ref<string | null>(null)
  const isLoading = ref(false)
  const error = ref<Error | null>(null)

  // Timer state
  let refreshTimerId: ReturnType<typeof setTimeout> | null = null

  // Request ID to prevent stale refresh operations from overwriting newer workspace contexts
  let refreshRequestId = 0

  // Getters
  const isAuthenticated = computed(
    () => currentWorkspace.value !== null && workspaceToken.value !== null
  )

  // Private helpers
  function stopRefreshTimer(): void {
    if (refreshTimerId !== null) {
      clearTimeout(refreshTimerId)
      refreshTimerId = null
    }
  }

  function scheduleTokenRefresh(expiresAt: number): void {
    stopRefreshTimer()
    const now = Date.now()
    const refreshAt = expiresAt - TOKEN_REFRESH_BUFFER_MS
    const delay = Math.max(0, refreshAt - now)

    refreshTimerId = setTimeout(() => {
      void refreshToken()
    }, delay)
  }

  function persistToSession(
    workspace: WorkspaceWithRole,
    token: string,
    expiresAt: number
  ): void {
    try {
      sessionStorage.setItem(
        WORKSPACE_STORAGE_KEYS.CURRENT_WORKSPACE,
        JSON.stringify(workspace)
      )
      sessionStorage.setItem(WORKSPACE_STORAGE_KEYS.TOKEN, token)
      sessionStorage.setItem(
        WORKSPACE_STORAGE_KEYS.EXPIRES_AT,
        expiresAt.toString()
      )
    } catch {
      console.warn('Failed to persist workspace context to sessionStorage')
    }
  }

  function clearSessionStorage(): void {
    try {
      sessionStorage.removeItem(WORKSPACE_STORAGE_KEYS.CURRENT_WORKSPACE)
      sessionStorage.removeItem(WORKSPACE_STORAGE_KEYS.TOKEN)
      sessionStorage.removeItem(WORKSPACE_STORAGE_KEYS.EXPIRES_AT)
    } catch {
      console.warn('Failed to clear workspace context from sessionStorage')
    }
  }

  // Actions
  function init(): void {
    initializeFromSession()
  }

  function destroy(): void {
    stopRefreshTimer()
  }

  function initializeFromSession(): boolean {
    if (!flags.teamWorkspacesEnabled) {
      return false
    }

    try {
      const workspaceJson = sessionStorage.getItem(
        WORKSPACE_STORAGE_KEYS.CURRENT_WORKSPACE
      )
      const token = sessionStorage.getItem(WORKSPACE_STORAGE_KEYS.TOKEN)
      const expiresAtStr = sessionStorage.getItem(
        WORKSPACE_STORAGE_KEYS.EXPIRES_AT
      )

      if (!workspaceJson || !token || !expiresAtStr) {
        return false
      }

      const expiresAt = parseInt(expiresAtStr, 10)
      if (isNaN(expiresAt) || expiresAt <= Date.now()) {
        clearSessionStorage()
        return false
      }

      const parsedWorkspace = JSON.parse(workspaceJson)
      const parseResult = WorkspaceWithRoleSchema.safeParse(parsedWorkspace)

      if (!parseResult.success) {
        clearSessionStorage()
        return false
      }

      currentWorkspace.value = parseResult.data
      workspaceToken.value = token
      error.value = null

      scheduleTokenRefresh(expiresAt)
      return true
    } catch {
      clearSessionStorage()
      return false
    }
  }

  async function switchWorkspace(workspaceId: string): Promise<void> {
    if (!flags.teamWorkspacesEnabled) {
      return
    }

    // Only increment request ID when switching to a different workspace
    // This invalidates stale refresh operations for the old workspace
    // but allows refresh operations for the same workspace to complete
    if (currentWorkspace.value?.id !== workspaceId) {
      refreshRequestId++
    }

    isLoading.value = true
    error.value = null

    try {
      const firebaseAuthStore = useFirebaseAuthStore()
      const firebaseToken = await firebaseAuthStore.getIdToken()
      if (!firebaseToken) {
        throw new WorkspaceAuthError(
          t('workspaceAuth.errors.notAuthenticated'),
          'NOT_AUTHENTICATED'
        )
      }

      const response = await fetch(api.apiURL('/auth/token'), {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${firebaseToken}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ workspace_id: workspaceId })
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        const message = errorData.message || response.statusText

        if (response.status === 401) {
          throw new WorkspaceAuthError(
            t('workspaceAuth.errors.invalidFirebaseToken'),
            'INVALID_FIREBASE_TOKEN'
          )
        }
        if (response.status === 403) {
          throw new WorkspaceAuthError(
            t('workspaceAuth.errors.accessDenied'),
            'ACCESS_DENIED'
          )
        }
        if (response.status === 404) {
          throw new WorkspaceAuthError(
            t('workspaceAuth.errors.workspaceNotFound'),
            'WORKSPACE_NOT_FOUND'
          )
        }

        throw new WorkspaceAuthError(
          t('workspaceAuth.errors.tokenExchangeFailed', { error: message }),
          'TOKEN_EXCHANGE_FAILED'
        )
      }

      const rawData = await response.json()
      const parseResult = WorkspaceTokenResponseSchema.safeParse(rawData)

      if (!parseResult.success) {
        throw new WorkspaceAuthError(
          t('workspaceAuth.errors.tokenExchangeFailed', {
            error: fromZodError(parseResult.error).message
          }),
          'TOKEN_EXCHANGE_FAILED'
        )
      }

      const data = parseResult.data
      const expiresAt = new Date(data.expires_at).getTime()

      if (isNaN(expiresAt)) {
        throw new WorkspaceAuthError(
          t('workspaceAuth.errors.tokenExchangeFailed', {
            error: 'Invalid expiry timestamp'
          }),
          'TOKEN_EXCHANGE_FAILED'
        )
      }

      const workspaceWithRole: WorkspaceWithRole = {
        ...data.workspace,
        role: data.role
      }

      currentWorkspace.value = workspaceWithRole
      workspaceToken.value = data.token

      persistToSession(workspaceWithRole, data.token, expiresAt)
      scheduleTokenRefresh(expiresAt)
    } catch (err) {
      error.value = err instanceof Error ? err : new Error(String(err))
      throw error.value
    } finally {
      isLoading.value = false
    }
  }

  async function refreshToken(): Promise<void> {
    if (!currentWorkspace.value) {
      return
    }

    const workspaceId = currentWorkspace.value.id
    // Capture the current request ID to detect if workspace context changed during refresh
    const capturedRequestId = refreshRequestId
    const maxRetries = 3
    const baseDelayMs = 1000

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      // Check if workspace context changed since refresh started (user switched workspaces)
      if (capturedRequestId !== refreshRequestId) {
        console.warn(
          'Aborting stale token refresh: workspace context changed during refresh'
        )
        return
      }

      try {
        await switchWorkspace(workspaceId)
        return
      } catch (err) {
        const isAuthError = err instanceof WorkspaceAuthError

        const isPermanentError =
          isAuthError &&
          (err.code === 'ACCESS_DENIED' ||
            err.code === 'WORKSPACE_NOT_FOUND' ||
            err.code === 'INVALID_FIREBASE_TOKEN' ||
            err.code === 'NOT_AUTHENTICATED')

        if (isPermanentError) {
          // Only clear context if this refresh is still for the current workspace
          if (capturedRequestId === refreshRequestId) {
            console.error('Workspace access revoked or auth invalid:', err)
            clearWorkspaceContext()
          }
          return
        }

        const isTransientError =
          isAuthError && err.code === 'TOKEN_EXCHANGE_FAILED'

        if (isTransientError && attempt < maxRetries) {
          const delay = baseDelayMs * Math.pow(2, attempt)
          console.warn(
            `Token refresh failed (attempt ${attempt + 1}/${maxRetries + 1}), retrying in ${delay}ms:`,
            err
          )
          await new Promise((resolve) => setTimeout(resolve, delay))
          continue
        }

        // Only clear context if this refresh is still for the current workspace
        if (capturedRequestId === refreshRequestId) {
          console.error('Failed to refresh workspace token after retries:', err)
          clearWorkspaceContext()
        }
      }
    }
  }

  function getWorkspaceAuthHeader(): AuthHeader | null {
    if (!workspaceToken.value) {
      return null
    }
    return {
      Authorization: `Bearer ${workspaceToken.value}`
    }
  }

  function clearWorkspaceContext(): void {
    // Increment request ID to invalidate any in-flight stale refresh operations
    refreshRequestId++
    stopRefreshTimer()
    currentWorkspace.value = null
    workspaceToken.value = null
    error.value = null
    clearSessionStorage()
  }

  return {
    // State
    currentWorkspace,
    workspaceToken,
    isLoading,
    error,

    // Getters
    isAuthenticated,

    // Actions
    init,
    destroy,
    initializeFromSession,
    switchWorkspace,
    refreshToken,
    getWorkspaceAuthHeader,
    clearWorkspaceContext
  }
})
