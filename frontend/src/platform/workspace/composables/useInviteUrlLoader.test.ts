import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useInviteUrlLoader } from './useInviteUrlLoader'

/**
 * Unit tests for useInviteUrlLoader composable
 *
 * Tests the behavior of accepting workspace invites via URL query parameters:
 * - ?invite=TOKEN accepts the invite and shows success toast
 * - Invalid/missing token is handled gracefully
 * - API errors show error toast
 * - URL is cleaned up after processing
 * - Preserved query is restored after login redirect
 */

const preservedQueryMocks = vi.hoisted(() => ({
  clearPreservedQuery: vi.fn(),
  hydratePreservedQuery: vi.fn(),
  mergePreservedQueryIntoQuery: vi.fn()
}))

vi.mock(
  '@/platform/navigation/preservedQueryManager',
  () => preservedQueryMocks
)

const mockRouteQuery = vi.hoisted(() => ({
  value: {} as Record<string, string>
}))
const mockRouterReplace = vi.hoisted(() => vi.fn())

vi.mock('vue-router', () => ({
  useRoute: () => ({
    query: mockRouteQuery.value
  }),
  useRouter: () => ({
    replace: mockRouterReplace
  })
}))

const mockToastAdd = vi.hoisted(() => vi.fn())
vi.mock('primevue/usetoast', () => ({
  useToast: () => ({
    add: mockToastAdd
  })
}))

vi.mock('vue-i18n', () => ({
  createI18n: () => ({
    global: {
      t: (key: string) => key
    }
  }),
  useI18n: () => ({
    t: vi.fn((key: string, params?: Record<string, unknown>) => {
      if (key === 'workspace.inviteAccepted') return 'Invite Accepted'
      if (key === 'workspace.addedToWorkspace') {
        return `You have been added to ${params?.workspaceName}`
      }
      if (key === 'workspace.inviteFailed') return 'Failed to Accept Invite'
      if (key === 'g.unknownError') return 'Unknown error'
      return key
    })
  })
}))

const mockAcceptInvite = vi.hoisted(() => vi.fn())
vi.mock('../stores/teamWorkspaceStore', () => ({
  useTeamWorkspaceStore: () => ({
    acceptInvite: mockAcceptInvite
  })
}))

describe('useInviteUrlLoader', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockRouteQuery.value = {}
    preservedQueryMocks.mergePreservedQueryIntoQuery.mockReturnValue(null)
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  describe('loadInviteFromUrl', () => {
    it('does nothing when no invite param present', async () => {
      mockRouteQuery.value = {}

      const { loadInviteFromUrl } = useInviteUrlLoader()
      await loadInviteFromUrl()

      expect(mockAcceptInvite).not.toHaveBeenCalled()
      expect(mockToastAdd).not.toHaveBeenCalled()
      expect(mockRouterReplace).not.toHaveBeenCalled()
    })

    it('restores preserved query and processes invite', async () => {
      mockRouteQuery.value = {}
      preservedQueryMocks.mergePreservedQueryIntoQuery.mockReturnValue({
        invite: 'preserved-token'
      })
      mockAcceptInvite.mockResolvedValue({
        workspaceId: 'ws-123',
        workspaceName: 'Test Workspace'
      })

      const { loadInviteFromUrl } = useInviteUrlLoader()
      await loadInviteFromUrl()

      expect(preservedQueryMocks.hydratePreservedQuery).toHaveBeenCalledWith(
        'invite'
      )
      expect(mockRouterReplace).toHaveBeenCalledWith({
        query: { invite: 'preserved-token' }
      })
      expect(mockAcceptInvite).toHaveBeenCalledWith('preserved-token')
    })

    it('accepts invite and shows success toast on success', async () => {
      mockRouteQuery.value = { invite: 'valid-token' }
      mockAcceptInvite.mockResolvedValue({
        workspaceId: 'ws-123',
        workspaceName: 'Test Workspace'
      })

      const { loadInviteFromUrl } = useInviteUrlLoader()
      await loadInviteFromUrl()

      expect(mockAcceptInvite).toHaveBeenCalledWith('valid-token')
      expect(mockToastAdd).toHaveBeenCalledWith({
        severity: 'success',
        summary: 'Invite Accepted',
        detail: {
          text: 'You have been added to Test Workspace',
          workspaceId: 'ws-123',
          workspaceName: 'Test Workspace'
        },
        group: 'invite-accepted',
        closable: true
      })
    })

    it('shows error toast when invite acceptance fails', async () => {
      mockRouteQuery.value = { invite: 'invalid-token' }
      mockAcceptInvite.mockRejectedValue(new Error('Invalid invite'))

      const { loadInviteFromUrl } = useInviteUrlLoader()
      await loadInviteFromUrl()

      expect(mockAcceptInvite).toHaveBeenCalledWith('invalid-token')
      expect(mockToastAdd).toHaveBeenCalledWith({
        severity: 'error',
        summary: 'Failed to Accept Invite',
        detail: 'Invalid invite',
        life: 5000
      })
    })

    it('cleans up URL after processing invite', async () => {
      mockRouteQuery.value = { invite: 'valid-token', other: 'param' }
      mockAcceptInvite.mockResolvedValue({
        workspaceId: 'ws-123',
        workspaceName: 'Test Workspace'
      })

      const { loadInviteFromUrl } = useInviteUrlLoader()
      await loadInviteFromUrl()

      // Should replace with query without invite param
      expect(mockRouterReplace).toHaveBeenCalledWith({
        query: { other: 'param' }
      })
    })

    it('clears preserved query after processing', async () => {
      mockRouteQuery.value = { invite: 'valid-token' }
      mockAcceptInvite.mockResolvedValue({
        workspaceId: 'ws-123',
        workspaceName: 'Test Workspace'
      })

      const { loadInviteFromUrl } = useInviteUrlLoader()
      await loadInviteFromUrl()

      expect(preservedQueryMocks.clearPreservedQuery).toHaveBeenCalledWith(
        'invite'
      )
    })

    it('clears preserved query even on error', async () => {
      mockRouteQuery.value = { invite: 'invalid-token' }
      mockAcceptInvite.mockRejectedValue(new Error('Invalid invite'))

      const { loadInviteFromUrl } = useInviteUrlLoader()
      await loadInviteFromUrl()

      expect(preservedQueryMocks.clearPreservedQuery).toHaveBeenCalledWith(
        'invite'
      )
    })

    it('sends any token format to backend for validation', async () => {
      mockRouteQuery.value = { invite: 'any-token-format==' }
      mockAcceptInvite.mockRejectedValue(new Error('Invalid token'))

      const { loadInviteFromUrl } = useInviteUrlLoader()
      await loadInviteFromUrl()

      // Token is sent to backend, which validates and rejects
      expect(mockAcceptInvite).toHaveBeenCalledWith('any-token-format==')
      expect(mockToastAdd).toHaveBeenCalledWith({
        severity: 'error',
        summary: 'Failed to Accept Invite',
        detail: 'Invalid token',
        life: 5000
      })
    })

    it('ignores empty invite param', async () => {
      mockRouteQuery.value = { invite: '' }

      const { loadInviteFromUrl } = useInviteUrlLoader()
      await loadInviteFromUrl()

      expect(mockAcceptInvite).not.toHaveBeenCalled()
    })

    it('ignores non-string invite param', async () => {
      mockRouteQuery.value = { invite: ['array', 'value'] as unknown as string }

      const { loadInviteFromUrl } = useInviteUrlLoader()
      await loadInviteFromUrl()

      expect(mockAcceptInvite).not.toHaveBeenCalled()
    })
  })
})
