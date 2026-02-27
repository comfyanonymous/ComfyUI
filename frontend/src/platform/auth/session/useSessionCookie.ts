import { useFeatureFlags } from '@/composables/useFeatureFlags'
import { isCloud } from '@/platform/distribution/types'
import { api } from '@/scripts/api'
import { useFirebaseAuthStore } from '@/stores/firebaseAuthStore'

/**
 * Session cookie management for cloud authentication.
 * Creates and deletes session cookies on the ComfyUI server.
 */
export const useSessionCookie = () => {
  /**
   * Creates or refreshes the session cookie.
   * Called after login and on token refresh.
   *
   * When team_workspaces_enabled is true, uses Firebase token directly
   * (since getAuthHeader() returns workspace token which shouldn't be used for session creation).
   * When disabled, uses getAuthHeader() for backward compatibility.
   */
  const createSession = async (): Promise<void> => {
    if (!isCloud) return

    const { flags } = useFeatureFlags()
    try {
      const authStore = useFirebaseAuthStore()

      let authHeader: Record<string, string>

      if (flags.teamWorkspacesEnabled) {
        const firebaseToken = await authStore.getIdToken()
        if (!firebaseToken) {
          console.warn(
            'Failed to create session cookie:',
            'No Firebase token available for session creation'
          )
          return
        }
        authHeader = { Authorization: `Bearer ${firebaseToken}` }
      } else {
        const header = await authStore.getAuthHeader()
        if (!header) {
          console.warn(
            'Failed to create session cookie:',
            'No auth header available for session creation'
          )
          return
        }
        authHeader = header
      }

      const response = await fetch(api.apiURL('/auth/session'), {
        method: 'POST',
        credentials: 'include',
        headers: {
          ...authHeader,
          'Content-Type': 'application/json'
        }
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        console.warn(
          'Failed to create session cookie:',
          errorData.message || response.statusText
        )
      }
    } catch (error) {
      console.warn('Failed to create session cookie:', error)
    }
  }

  /**
   * Deletes the session cookie.
   * Called on logout.
   */
  const deleteSession = async (): Promise<void> => {
    if (!isCloud) return

    try {
      const response = await fetch(api.apiURL('/auth/session'), {
        method: 'DELETE',
        credentials: 'include'
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        console.warn(
          'Failed to delete session cookie:',
          errorData.message || response.statusText
        )
      }
    } catch (error) {
      console.warn('Failed to delete session cookie:', error)
    }
  }

  return {
    createSession,
    deleteSession
  }
}
