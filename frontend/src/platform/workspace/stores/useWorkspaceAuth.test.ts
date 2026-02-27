import { createPinia, setActivePinia, storeToRefs } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  useWorkspaceAuthStore,
  WorkspaceAuthError
} from '@/platform/workspace/stores/workspaceAuthStore'

import { WORKSPACE_STORAGE_KEYS } from '@/platform/workspace/workspaceConstants'

const mockGetIdToken = vi.fn()

vi.mock('@/stores/firebaseAuthStore', () => ({
  useFirebaseAuthStore: () => ({
    getIdToken: mockGetIdToken
  })
}))

vi.mock('@/scripts/api', () => ({
  api: {
    apiURL: (route: string) => `https://api.example.com/api${route}`
  }
}))

vi.mock('@/i18n', () => ({
  t: (key: string) => key
}))

const mockTeamWorkspacesEnabled = vi.hoisted(() => ({ value: true }))

vi.mock('@/composables/useFeatureFlags', () => ({
  useFeatureFlags: () => ({
    flags: {
      get teamWorkspacesEnabled() {
        return mockTeamWorkspacesEnabled.value
      }
    }
  })
}))

const mockWorkspace = {
  id: 'workspace-123',
  name: 'Test Workspace',
  type: 'team' as const
}

const mockWorkspaceWithRole = {
  ...mockWorkspace,
  role: 'owner' as const
}

const mockTokenResponse = {
  token: 'workspace-token-abc',
  expires_at: new Date(Date.now() + 3600 * 1000).toISOString(),
  workspace: mockWorkspace,
  role: 'owner' as const,
  permissions: ['owner:*']
}

describe('useWorkspaceAuthStore', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()
    vi.useFakeTimers()
    sessionStorage.clear()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  describe('initial state', () => {
    it('has correct initial state values', () => {
      const store = useWorkspaceAuthStore()
      const {
        currentWorkspace,
        workspaceToken,
        isAuthenticated,
        isLoading,
        error
      } = storeToRefs(store)

      expect(currentWorkspace.value).toBeNull()
      expect(workspaceToken.value).toBeNull()
      expect(isAuthenticated.value).toBe(false)
      expect(isLoading.value).toBe(false)
      expect(error.value).toBeNull()
    })
  })

  describe('initializeFromSession', () => {
    it('returns true and populates state when valid session data exists', () => {
      const futureExpiry = Date.now() + 3600 * 1000
      sessionStorage.setItem(
        WORKSPACE_STORAGE_KEYS.CURRENT_WORKSPACE,
        JSON.stringify(mockWorkspaceWithRole)
      )
      sessionStorage.setItem(WORKSPACE_STORAGE_KEYS.TOKEN, 'valid-token')
      sessionStorage.setItem(
        WORKSPACE_STORAGE_KEYS.EXPIRES_AT,
        futureExpiry.toString()
      )

      const store = useWorkspaceAuthStore()
      const { currentWorkspace, workspaceToken } = storeToRefs(store)

      const result = store.initializeFromSession()

      expect(result).toBe(true)
      expect(currentWorkspace.value).toEqual(mockWorkspaceWithRole)
      expect(workspaceToken.value).toBe('valid-token')
    })

    it('returns false when sessionStorage is empty', () => {
      const store = useWorkspaceAuthStore()

      const result = store.initializeFromSession()

      expect(result).toBe(false)
    })

    it('returns false and clears storage when token is expired', () => {
      const pastExpiry = Date.now() - 1000
      sessionStorage.setItem(
        WORKSPACE_STORAGE_KEYS.CURRENT_WORKSPACE,
        JSON.stringify(mockWorkspaceWithRole)
      )
      sessionStorage.setItem(WORKSPACE_STORAGE_KEYS.TOKEN, 'expired-token')
      sessionStorage.setItem(
        WORKSPACE_STORAGE_KEYS.EXPIRES_AT,
        pastExpiry.toString()
      )

      const store = useWorkspaceAuthStore()

      const result = store.initializeFromSession()

      expect(result).toBe(false)
      expect(
        sessionStorage.getItem(WORKSPACE_STORAGE_KEYS.CURRENT_WORKSPACE)
      ).toBeNull()
      expect(sessionStorage.getItem(WORKSPACE_STORAGE_KEYS.TOKEN)).toBeNull()
      expect(
        sessionStorage.getItem(WORKSPACE_STORAGE_KEYS.EXPIRES_AT)
      ).toBeNull()
    })

    it('returns false and clears storage when data is malformed', () => {
      sessionStorage.setItem(
        WORKSPACE_STORAGE_KEYS.CURRENT_WORKSPACE,
        'invalid-json{'
      )
      sessionStorage.setItem(WORKSPACE_STORAGE_KEYS.TOKEN, 'some-token')
      sessionStorage.setItem(WORKSPACE_STORAGE_KEYS.EXPIRES_AT, 'not-a-number')

      const store = useWorkspaceAuthStore()

      const result = store.initializeFromSession()

      expect(result).toBe(false)
      expect(
        sessionStorage.getItem(WORKSPACE_STORAGE_KEYS.CURRENT_WORKSPACE)
      ).toBeNull()
      expect(sessionStorage.getItem(WORKSPACE_STORAGE_KEYS.TOKEN)).toBeNull()
      expect(
        sessionStorage.getItem(WORKSPACE_STORAGE_KEYS.EXPIRES_AT)
      ).toBeNull()
    })

    it('returns false when partial session data exists (missing token)', () => {
      sessionStorage.setItem(
        WORKSPACE_STORAGE_KEYS.CURRENT_WORKSPACE,
        JSON.stringify(mockWorkspaceWithRole)
      )
      sessionStorage.setItem(
        WORKSPACE_STORAGE_KEYS.EXPIRES_AT,
        (Date.now() + 3600 * 1000).toString()
      )

      const store = useWorkspaceAuthStore()

      const result = store.initializeFromSession()

      expect(result).toBe(false)
    })
  })

  describe('switchWorkspace', () => {
    it('successfully exchanges Firebase token for workspace token', async () => {
      mockGetIdToken.mockResolvedValue('firebase-token-xyz')
      vi.stubGlobal(
        'fetch',
        vi.fn().mockResolvedValue({
          ok: true,
          json: () => Promise.resolve(mockTokenResponse)
        })
      )

      const store = useWorkspaceAuthStore()
      const { currentWorkspace, workspaceToken, isAuthenticated } =
        storeToRefs(store)

      await store.switchWorkspace('workspace-123')

      expect(currentWorkspace.value).toEqual(mockWorkspaceWithRole)
      expect(workspaceToken.value).toBe('workspace-token-abc')
      expect(isAuthenticated.value).toBe(true)
    })

    it('stores workspace data in sessionStorage', async () => {
      mockGetIdToken.mockResolvedValue('firebase-token-xyz')
      vi.stubGlobal(
        'fetch',
        vi.fn().mockResolvedValue({
          ok: true,
          json: () => Promise.resolve(mockTokenResponse)
        })
      )

      const store = useWorkspaceAuthStore()

      await store.switchWorkspace('workspace-123')

      expect(
        sessionStorage.getItem(WORKSPACE_STORAGE_KEYS.CURRENT_WORKSPACE)
      ).toBe(JSON.stringify(mockWorkspaceWithRole))
      expect(sessionStorage.getItem(WORKSPACE_STORAGE_KEYS.TOKEN)).toBe(
        'workspace-token-abc'
      )
      expect(
        sessionStorage.getItem(WORKSPACE_STORAGE_KEYS.EXPIRES_AT)
      ).toBeTruthy()
    })

    it('sets isLoading to true during operation', async () => {
      mockGetIdToken.mockResolvedValue('firebase-token-xyz')
      let resolveResponse: (value: unknown) => void
      const responsePromise = new Promise((resolve) => {
        resolveResponse = resolve
      })
      vi.stubGlobal('fetch', vi.fn().mockReturnValue(responsePromise))

      const store = useWorkspaceAuthStore()
      const { isLoading } = storeToRefs(store)

      const switchPromise = store.switchWorkspace('workspace-123')
      expect(isLoading.value).toBe(true)

      resolveResponse!({
        ok: true,
        json: () => Promise.resolve(mockTokenResponse)
      })
      await switchPromise

      expect(isLoading.value).toBe(false)
    })

    it('throws WorkspaceAuthError with code NOT_AUTHENTICATED when Firebase token unavailable', async () => {
      mockGetIdToken.mockResolvedValue(undefined)

      const store = useWorkspaceAuthStore()
      const { error } = storeToRefs(store)

      await expect(store.switchWorkspace('workspace-123')).rejects.toThrow(
        WorkspaceAuthError
      )

      expect(error.value).toBeInstanceOf(WorkspaceAuthError)
      expect((error.value as WorkspaceAuthError).code).toBe('NOT_AUTHENTICATED')
    })

    it('throws WorkspaceAuthError with code ACCESS_DENIED on 403 response', async () => {
      mockGetIdToken.mockResolvedValue('firebase-token-xyz')
      vi.stubGlobal(
        'fetch',
        vi.fn().mockResolvedValue({
          ok: false,
          status: 403,
          statusText: 'Forbidden',
          json: () => Promise.resolve({ message: 'Access denied' })
        })
      )

      const store = useWorkspaceAuthStore()
      const { error } = storeToRefs(store)

      await expect(store.switchWorkspace('workspace-123')).rejects.toThrow(
        WorkspaceAuthError
      )

      expect(error.value).toBeInstanceOf(WorkspaceAuthError)
      expect((error.value as WorkspaceAuthError).code).toBe('ACCESS_DENIED')
    })

    it('throws WorkspaceAuthError with code WORKSPACE_NOT_FOUND on 404 response', async () => {
      mockGetIdToken.mockResolvedValue('firebase-token-xyz')
      vi.stubGlobal(
        'fetch',
        vi.fn().mockResolvedValue({
          ok: false,
          status: 404,
          statusText: 'Not Found',
          json: () => Promise.resolve({ message: 'Workspace not found' })
        })
      )

      const store = useWorkspaceAuthStore()
      const { error } = storeToRefs(store)

      await expect(store.switchWorkspace('workspace-123')).rejects.toThrow(
        WorkspaceAuthError
      )

      expect(error.value).toBeInstanceOf(WorkspaceAuthError)
      expect((error.value as WorkspaceAuthError).code).toBe(
        'WORKSPACE_NOT_FOUND'
      )
    })

    it('throws WorkspaceAuthError with code INVALID_FIREBASE_TOKEN on 401 response', async () => {
      mockGetIdToken.mockResolvedValue('firebase-token-xyz')
      vi.stubGlobal(
        'fetch',
        vi.fn().mockResolvedValue({
          ok: false,
          status: 401,
          statusText: 'Unauthorized',
          json: () => Promise.resolve({ message: 'Invalid token' })
        })
      )

      const store = useWorkspaceAuthStore()
      const { error } = storeToRefs(store)

      await expect(store.switchWorkspace('workspace-123')).rejects.toThrow(
        WorkspaceAuthError
      )

      expect(error.value).toBeInstanceOf(WorkspaceAuthError)
      expect((error.value as WorkspaceAuthError).code).toBe(
        'INVALID_FIREBASE_TOKEN'
      )
    })

    it('throws WorkspaceAuthError with code TOKEN_EXCHANGE_FAILED on other errors', async () => {
      mockGetIdToken.mockResolvedValue('firebase-token-xyz')
      vi.stubGlobal(
        'fetch',
        vi.fn().mockResolvedValue({
          ok: false,
          status: 500,
          statusText: 'Internal Server Error',
          json: () => Promise.resolve({ message: 'Server error' })
        })
      )

      const store = useWorkspaceAuthStore()
      const { error } = storeToRefs(store)

      await expect(store.switchWorkspace('workspace-123')).rejects.toThrow(
        WorkspaceAuthError
      )

      expect(error.value).toBeInstanceOf(WorkspaceAuthError)
      expect((error.value as WorkspaceAuthError).code).toBe(
        'TOKEN_EXCHANGE_FAILED'
      )
    })

    it('sends correct request to API', async () => {
      mockGetIdToken.mockResolvedValue('firebase-token-xyz')
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockTokenResponse)
      })
      vi.stubGlobal('fetch', mockFetch)

      const store = useWorkspaceAuthStore()

      await store.switchWorkspace('workspace-123')

      expect(mockFetch).toHaveBeenCalledWith(
        'https://api.example.com/api/auth/token',
        {
          method: 'POST',
          headers: {
            Authorization: 'Bearer firebase-token-xyz',
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ workspace_id: 'workspace-123' })
        }
      )
    })
  })

  describe('clearWorkspaceContext', () => {
    it('clears all state refs', async () => {
      mockGetIdToken.mockResolvedValue('firebase-token-xyz')
      vi.stubGlobal(
        'fetch',
        vi.fn().mockResolvedValue({
          ok: true,
          json: () => Promise.resolve(mockTokenResponse)
        })
      )

      const store = useWorkspaceAuthStore()
      const { currentWorkspace, workspaceToken, error, isAuthenticated } =
        storeToRefs(store)

      await store.switchWorkspace('workspace-123')
      expect(isAuthenticated.value).toBe(true)

      store.clearWorkspaceContext()

      expect(currentWorkspace.value).toBeNull()
      expect(workspaceToken.value).toBeNull()
      expect(error.value).toBeNull()
      expect(isAuthenticated.value).toBe(false)
    })

    it('clears sessionStorage', async () => {
      sessionStorage.setItem(
        WORKSPACE_STORAGE_KEYS.CURRENT_WORKSPACE,
        JSON.stringify(mockWorkspaceWithRole)
      )
      sessionStorage.setItem(WORKSPACE_STORAGE_KEYS.TOKEN, 'some-token')
      sessionStorage.setItem(WORKSPACE_STORAGE_KEYS.EXPIRES_AT, '12345')

      const store = useWorkspaceAuthStore()

      store.clearWorkspaceContext()

      expect(
        sessionStorage.getItem(WORKSPACE_STORAGE_KEYS.CURRENT_WORKSPACE)
      ).toBeNull()
      expect(sessionStorage.getItem(WORKSPACE_STORAGE_KEYS.TOKEN)).toBeNull()
      expect(
        sessionStorage.getItem(WORKSPACE_STORAGE_KEYS.EXPIRES_AT)
      ).toBeNull()
    })
  })

  describe('getWorkspaceAuthHeader', () => {
    it('returns null when no workspace token', () => {
      const store = useWorkspaceAuthStore()

      const header = store.getWorkspaceAuthHeader()

      expect(header).toBeNull()
    })

    it('returns proper Authorization header when workspace token exists', async () => {
      mockGetIdToken.mockResolvedValue('firebase-token-xyz')
      vi.stubGlobal(
        'fetch',
        vi.fn().mockResolvedValue({
          ok: true,
          json: () => Promise.resolve(mockTokenResponse)
        })
      )

      const store = useWorkspaceAuthStore()

      await store.switchWorkspace('workspace-123')
      const header = store.getWorkspaceAuthHeader()

      expect(header).toEqual({
        Authorization: 'Bearer workspace-token-abc'
      })
    })
  })

  describe('token refresh scheduling', () => {
    it('schedules token refresh 5 minutes before expiry', async () => {
      mockGetIdToken.mockResolvedValue('firebase-token-xyz')
      const expiresInMs = 3600 * 1000
      const tokenResponseWithFutureExpiry = {
        ...mockTokenResponse,
        expires_at: new Date(Date.now() + expiresInMs).toISOString()
      }
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(tokenResponseWithFutureExpiry)
      })
      vi.stubGlobal('fetch', mockFetch)

      const store = useWorkspaceAuthStore()

      await store.switchWorkspace('workspace-123')

      expect(mockFetch).toHaveBeenCalledTimes(1)

      const refreshBufferMs = 5 * 60 * 1000
      const refreshDelay = expiresInMs - refreshBufferMs

      vi.advanceTimersByTime(refreshDelay - 1)
      expect(mockFetch).toHaveBeenCalledTimes(1)

      await vi.advanceTimersByTimeAsync(1)

      expect(mockFetch).toHaveBeenCalledTimes(2)
    })

    it('clears context when refresh fails with ACCESS_DENIED', async () => {
      mockGetIdToken.mockResolvedValue('firebase-token-xyz')
      const expiresInMs = 3600 * 1000
      const tokenResponseWithFutureExpiry = {
        ...mockTokenResponse,
        expires_at: new Date(Date.now() + expiresInMs).toISOString()
      }
      const mockFetch = vi
        .fn()
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(tokenResponseWithFutureExpiry)
        })
        .mockResolvedValueOnce({
          ok: false,
          status: 403,
          statusText: 'Forbidden',
          json: () => Promise.resolve({ message: 'Access denied' })
        })
      vi.stubGlobal('fetch', mockFetch)

      const store = useWorkspaceAuthStore()
      const { currentWorkspace, workspaceToken } = storeToRefs(store)

      await store.switchWorkspace('workspace-123')
      expect(workspaceToken.value).toBe('workspace-token-abc')

      const refreshBufferMs = 5 * 60 * 1000
      const refreshDelay = expiresInMs - refreshBufferMs

      vi.advanceTimersByTime(refreshDelay)
      await vi.waitFor(() => {
        expect(currentWorkspace.value).toBeNull()
      })

      expect(workspaceToken.value).toBeNull()
    })
  })

  describe('refreshToken', () => {
    it('does nothing when no current workspace', async () => {
      const mockFetch = vi.fn()
      vi.stubGlobal('fetch', mockFetch)

      const store = useWorkspaceAuthStore()

      await store.refreshToken()

      expect(mockFetch).not.toHaveBeenCalled()
    })

    it('refreshes token for current workspace', async () => {
      mockGetIdToken.mockResolvedValue('firebase-token-xyz')
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockTokenResponse)
      })
      vi.stubGlobal('fetch', mockFetch)

      const store = useWorkspaceAuthStore()
      const { workspaceToken } = storeToRefs(store)

      await store.switchWorkspace('workspace-123')
      expect(mockFetch).toHaveBeenCalledTimes(1)

      mockFetch.mockResolvedValue({
        ok: true,
        json: () =>
          Promise.resolve({
            ...mockTokenResponse,
            token: 'refreshed-token'
          })
      })

      await store.refreshToken()
      expect(mockFetch).toHaveBeenCalledTimes(2)
      expect(workspaceToken.value).toBe('refreshed-token')
    })
  })

  describe('isAuthenticated computed', () => {
    it('returns true when both workspace and token are present', async () => {
      mockGetIdToken.mockResolvedValue('firebase-token-xyz')
      vi.stubGlobal(
        'fetch',
        vi.fn().mockResolvedValue({
          ok: true,
          json: () => Promise.resolve(mockTokenResponse)
        })
      )

      const store = useWorkspaceAuthStore()
      const { isAuthenticated } = storeToRefs(store)

      await store.switchWorkspace('workspace-123')

      expect(isAuthenticated.value).toBe(true)
    })

    it('returns false when workspace is null', () => {
      const store = useWorkspaceAuthStore()
      const { isAuthenticated } = storeToRefs(store)

      expect(isAuthenticated.value).toBe(false)
    })

    it('returns false when currentWorkspace is set but workspaceToken is null', async () => {
      mockGetIdToken.mockResolvedValue(null)

      const store = useWorkspaceAuthStore()
      const { currentWorkspace, workspaceToken, isAuthenticated } =
        storeToRefs(store)

      currentWorkspace.value = mockWorkspaceWithRole
      workspaceToken.value = null

      expect(isAuthenticated.value).toBe(false)
    })
  })

  describe('feature flag disabled', () => {
    beforeEach(() => {
      mockTeamWorkspacesEnabled.value = false
    })

    afterEach(() => {
      mockTeamWorkspacesEnabled.value = true
    })

    it('initializeFromSession returns false when flag disabled', () => {
      const futureExpiry = Date.now() + 3600 * 1000
      sessionStorage.setItem(
        WORKSPACE_STORAGE_KEYS.CURRENT_WORKSPACE,
        JSON.stringify(mockWorkspaceWithRole)
      )
      sessionStorage.setItem(WORKSPACE_STORAGE_KEYS.TOKEN, 'valid-token')
      sessionStorage.setItem(
        WORKSPACE_STORAGE_KEYS.EXPIRES_AT,
        futureExpiry.toString()
      )

      const store = useWorkspaceAuthStore()
      const { currentWorkspace, workspaceToken } = storeToRefs(store)

      const result = store.initializeFromSession()

      expect(result).toBe(false)
      expect(currentWorkspace.value).toBeNull()
      expect(workspaceToken.value).toBeNull()
    })

    it('switchWorkspace is a no-op when flag disabled', async () => {
      mockGetIdToken.mockResolvedValue('firebase-token-xyz')
      const mockFetch = vi.fn()
      vi.stubGlobal('fetch', mockFetch)

      const store = useWorkspaceAuthStore()
      const { currentWorkspace, workspaceToken, isLoading } = storeToRefs(store)

      await store.switchWorkspace('workspace-123')

      expect(mockFetch).not.toHaveBeenCalled()
      expect(currentWorkspace.value).toBeNull()
      expect(workspaceToken.value).toBeNull()
      expect(isLoading.value).toBe(false)
    })
  })
})
