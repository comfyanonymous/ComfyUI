import { createTestingPinia } from '@pinia/testing'
import { setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { sortWorkspaces, useTeamWorkspaceStore } from './teamWorkspaceStore'

// Mock workspaceAuthStore
const mockWorkspaceAuthStore = vi.hoisted(() => ({
  currentWorkspace: null as {
    id: string
    name: string
    type: 'personal' | 'team'
    role: 'owner' | 'member'
  } | null,
  workspaceToken: null as string | null,
  isLoading: false,
  error: null as Error | null,
  isAuthenticated: false,
  init: vi.fn(),
  destroy: vi.fn(),
  initializeFromSession: vi.fn(),
  switchWorkspace: vi.fn(),
  refreshToken: vi.fn(),
  getWorkspaceAuthHeader: vi.fn(),
  clearWorkspaceContext: vi.fn()
}))

vi.mock('@/platform/workspace/stores/workspaceAuthStore', () => ({
  useWorkspaceAuthStore: () => mockWorkspaceAuthStore
}))

// Mock workspaceApi
const mockWorkspaceApi = vi.hoisted(() => ({
  list: vi.fn(),
  create: vi.fn(),
  update: vi.fn(),
  delete: vi.fn(),
  leave: vi.fn(),
  listMembers: vi.fn(),
  removeMember: vi.fn(),
  listInvites: vi.fn(),
  createInvite: vi.fn(),
  revokeInvite: vi.fn(),
  acceptInvite: vi.fn(),
  accessBillingPortal: vi.fn()
}))

const mockWorkspaceApiError = vi.hoisted(
  () =>
    class WorkspaceApiError extends Error {
      constructor(
        message: string,
        public readonly status?: number,
        public readonly code?: string
      ) {
        super(message)
        this.name = 'WorkspaceApiError'
      }
    }
)

vi.mock('../api/workspaceApi', () => ({
  workspaceApi: mockWorkspaceApi,
  WorkspaceApiError: mockWorkspaceApiError
}))

// Mock localStorage
const mockLocalStorage = vi.hoisted(() => {
  const store: Record<string, string> = {}
  return {
    getItem: vi.fn((_key: string): string | null => store[_key] ?? null),
    setItem: vi.fn((key: string, value: string) => {
      store[key] = value
    }),
    removeItem: vi.fn((key: string) => {
      delete store[key]
    }),
    clear: vi.fn(() => {
      Object.keys(store).forEach((key) => delete store[key])
    })
  }
})

// Mock window.location.reload
const mockReload = vi.fn()
Object.defineProperty(window, 'location', {
  value: { reload: mockReload, origin: 'http://localhost' },
  writable: true
})

// Test data
const mockPersonalWorkspace = {
  id: 'ws-personal-123',
  name: 'Personal',
  type: 'personal' as const,
  role: 'owner' as const,
  created_at: '2026-01-01T00:00:00Z',
  joined_at: '2026-01-01T00:00:00Z'
}

const mockTeamWorkspace = {
  id: 'ws-team-456',
  name: 'Team Alpha',
  type: 'team' as const,
  role: 'owner' as const,
  created_at: '2026-02-01T00:00:00Z',
  joined_at: '2026-02-01T00:00:00Z'
}

const mockMemberWorkspace = {
  id: 'ws-team-789',
  name: 'Team Beta',
  type: 'team' as const,
  role: 'member' as const,
  created_at: '2026-03-01T00:00:00Z',
  joined_at: '2026-03-01T00:00:00Z'
}

describe('useTeamWorkspaceStore', () => {
  beforeEach(() => {
    setActivePinia(createTestingPinia({ stubActions: false }))
    vi.clearAllMocks()
    vi.stubGlobal('localStorage', mockLocalStorage)
    sessionStorage.clear()

    // Reset workspaceAuthStore mock state
    mockWorkspaceAuthStore.currentWorkspace = null
    mockWorkspaceAuthStore.workspaceToken = null
    mockWorkspaceAuthStore.isLoading = false
    mockWorkspaceAuthStore.error = null
    mockWorkspaceAuthStore.isAuthenticated = false
    mockWorkspaceAuthStore.initializeFromSession.mockReturnValue(false)
    mockWorkspaceAuthStore.switchWorkspace.mockResolvedValue(undefined)

    // Default mock responses
    mockWorkspaceApi.list.mockResolvedValue({
      workspaces: [mockPersonalWorkspace, mockTeamWorkspace]
    })
    mockLocalStorage.getItem.mockReturnValue(null)
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  describe('initial state', () => {
    it('has correct initial state values', () => {
      const store = useTeamWorkspaceStore()

      expect(store.initState).toBe('uninitialized')
      expect(store.workspaces).toEqual([])
      expect(store.activeWorkspaceId).toBeNull()
      expect(store.error).toBeNull()
      expect(store.isCreating).toBe(false)
      expect(store.isDeleting).toBe(false)
      expect(store.isSwitching).toBe(false)
      expect(store.isFetchingWorkspaces).toBe(false)
    })

    it('computed properties return correct defaults', () => {
      const store = useTeamWorkspaceStore()

      expect(store.activeWorkspace).toBeNull()
      expect(store.personalWorkspace).toBeNull()
      expect(store.isInPersonalWorkspace).toBe(false)
      expect(store.sharedWorkspaces).toEqual([])
      expect(store.ownedWorkspacesCount).toBe(0)
      expect(store.canCreateWorkspace).toBe(true)
      expect(store.members).toEqual([])
      expect(store.pendingInvites).toEqual([])
    })
  })

  describe('initialize', () => {
    it('fetches workspaces and sets active workspace to personal by default', async () => {
      const store = useTeamWorkspaceStore()

      await store.initialize()

      expect(mockWorkspaceApi.list).toHaveBeenCalledTimes(1)
      expect(store.initState).toBe('ready')
      expect(store.workspaces).toHaveLength(2)
      expect(store.activeWorkspaceId).toBe(mockPersonalWorkspace.id)
      expect(mockWorkspaceAuthStore.switchWorkspace).toHaveBeenCalledWith(
        mockPersonalWorkspace.id
      )
    })

    it('restores workspace from session if valid', async () => {
      mockWorkspaceAuthStore.initializeFromSession.mockReturnValue(true)
      mockWorkspaceAuthStore.currentWorkspace = mockTeamWorkspace

      const store = useTeamWorkspaceStore()
      await store.initialize()

      expect(store.activeWorkspaceId).toBe(mockTeamWorkspace.id)
      expect(mockWorkspaceAuthStore.switchWorkspace).not.toHaveBeenCalled()
    })

    it('falls back to localStorage if no session', async () => {
      mockLocalStorage.getItem.mockReturnValue(mockTeamWorkspace.id)

      const store = useTeamWorkspaceStore()
      await store.initialize()

      expect(store.activeWorkspaceId).toBe(mockTeamWorkspace.id)
    })

    it('falls back to personal if stored workspace not in list', async () => {
      mockLocalStorage.getItem.mockReturnValue('non-existent-workspace')

      const store = useTeamWorkspaceStore()
      await store.initialize()

      expect(store.activeWorkspaceId).toBe(mockPersonalWorkspace.id)
    })

    it('sets error state when workspaces fetch fails after retries', async () => {
      vi.useFakeTimers()
      mockWorkspaceApi.list.mockRejectedValue(new Error('Network error'))

      const store = useTeamWorkspaceStore()

      // Start initialization and catch rejections to prevent unhandled promise warning
      let initError: unknown = null
      const initPromise = store.initialize().catch((e: unknown) => {
        initError = e
      })

      // Fast-forward through all retry delays (1s, 2s, 4s)
      await vi.advanceTimersByTimeAsync(1000)
      await vi.advanceTimersByTimeAsync(2000)
      await vi.advanceTimersByTimeAsync(4000)

      await initPromise

      expect(initError).toBeInstanceOf(Error)
      expect((initError as Error).message).toBe('Network error')
      expect(store.initState).toBe('error')
      expect(store.error).toBeInstanceOf(Error)
      // Should have been called 4 times (initial + 3 retries)
      expect(mockWorkspaceApi.list).toHaveBeenCalledTimes(4)

      vi.useRealTimers()
    })

    it('does not reinitialize if already initialized', async () => {
      const store = useTeamWorkspaceStore()

      await store.initialize()
      await store.initialize()

      expect(mockWorkspaceApi.list).toHaveBeenCalledTimes(1)
    })

    it('throws when no workspaces available', async () => {
      mockWorkspaceApi.list.mockResolvedValue({ workspaces: [] })

      const store = useTeamWorkspaceStore()

      await expect(store.initialize()).rejects.toThrow(
        'No workspaces available'
      )
      expect(store.initState).toBe('error')
    })

    it('continues initialization even if token exchange fails', async () => {
      mockWorkspaceAuthStore.switchWorkspace.mockRejectedValue(
        new Error('Token exchange failed')
      )

      const store = useTeamWorkspaceStore()
      await store.initialize()

      expect(store.initState).toBe('ready')
      expect(store.activeWorkspaceId).toBe(mockPersonalWorkspace.id)
    })
  })

  describe('switchWorkspace', () => {
    it('does nothing if switching to current workspace', async () => {
      const store = useTeamWorkspaceStore()
      await store.initialize()

      const currentId = store.activeWorkspaceId
      await store.switchWorkspace(currentId!)

      expect(mockReload).not.toHaveBeenCalled()
    })

    it('clears context and reloads for valid workspace', async () => {
      const store = useTeamWorkspaceStore()
      await store.initialize()

      await store.switchWorkspace(mockTeamWorkspace.id)

      expect(mockWorkspaceAuthStore.clearWorkspaceContext).toHaveBeenCalled()
      expect(mockReload).toHaveBeenCalled()
    })

    it('sets isSwitching flag during operation', async () => {
      const store = useTeamWorkspaceStore()
      await store.initialize()

      expect(store.isSwitching).toBe(false)

      const switchPromise = store.switchWorkspace(mockTeamWorkspace.id)
      expect(store.isSwitching).toBe(true)

      await switchPromise
    })

    it('refreshes workspace list if target not found', async () => {
      const newWorkspace = {
        id: 'ws-new-999',
        name: 'New Workspace',
        type: 'team' as const,
        role: 'member' as const,
        created_at: '2026-04-01T00:00:00Z',
        joined_at: '2026-04-01T00:00:00Z'
      }

      mockWorkspaceApi.list
        .mockResolvedValueOnce({
          workspaces: [mockPersonalWorkspace, mockTeamWorkspace]
        })
        .mockResolvedValueOnce({
          workspaces: [mockPersonalWorkspace, mockTeamWorkspace, newWorkspace]
        })

      const store = useTeamWorkspaceStore()
      await store.initialize()

      await store.switchWorkspace(newWorkspace.id)

      expect(mockWorkspaceApi.list).toHaveBeenCalledTimes(2)
      expect(mockReload).toHaveBeenCalled()
    })

    it('throws if workspace not found after refresh', async () => {
      const store = useTeamWorkspaceStore()
      await store.initialize()

      await expect(
        store.switchWorkspace('non-existent-workspace')
      ).rejects.toThrow('Workspace not found or access denied')

      expect(store.isSwitching).toBe(false)
    })
  })

  describe('createWorkspace', () => {
    it('creates workspace and triggers reload', async () => {
      const newWorkspace = {
        id: 'ws-new-created',
        name: 'Created Workspace',
        type: 'team' as const,
        role: 'owner' as const,
        created_at: '2026-05-01T00:00:00Z',
        joined_at: '2026-05-01T00:00:00Z'
      }
      mockWorkspaceApi.create.mockResolvedValue(newWorkspace)

      const store = useTeamWorkspaceStore()
      await store.initialize()

      const result = await store.createWorkspace('Created Workspace')

      expect(mockWorkspaceApi.create).toHaveBeenCalledWith({
        name: 'Created Workspace'
      })
      expect(result.id).toBe(newWorkspace.id)
      expect(store.workspaces).toContainEqual(
        expect.objectContaining({ id: newWorkspace.id })
      )
      expect(mockWorkspaceAuthStore.clearWorkspaceContext).toHaveBeenCalled()
      expect(mockReload).toHaveBeenCalled()
    })

    it('sets isCreating flag during operation', async () => {
      let resolveCreate: (value: unknown) => void
      const createPromise = new Promise((resolve) => {
        resolveCreate = resolve
      })
      mockWorkspaceApi.create.mockReturnValue(createPromise)

      const store = useTeamWorkspaceStore()
      await store.initialize()

      expect(store.isCreating).toBe(false)

      const resultPromise = store.createWorkspace('New Workspace')
      expect(store.isCreating).toBe(true)

      resolveCreate!({
        id: 'ws-new',
        name: 'New Workspace',
        type: 'team',
        role: 'owner',
        created_at: '2026-06-01T00:00:00Z',
        joined_at: '2026-06-01T00:00:00Z'
      })
      await resultPromise
    })

    it('resets isCreating on error', async () => {
      mockWorkspaceApi.create.mockRejectedValue(new Error('Creation failed'))

      const store = useTeamWorkspaceStore()
      await store.initialize()

      await expect(store.createWorkspace('New Workspace')).rejects.toThrow(
        'Creation failed'
      )
      expect(store.isCreating).toBe(false)
    })
  })

  describe('deleteWorkspace', () => {
    it('deletes non-active workspace without reload', async () => {
      const store = useTeamWorkspaceStore()
      await store.initialize()

      expect(store.activeWorkspaceId).toBe(mockPersonalWorkspace.id)

      await store.deleteWorkspace(mockTeamWorkspace.id)

      expect(mockWorkspaceApi.delete).toHaveBeenCalledWith(mockTeamWorkspace.id)
      expect(store.workspaces).not.toContainEqual(
        expect.objectContaining({ id: mockTeamWorkspace.id })
      )
      expect(mockReload).not.toHaveBeenCalled()
    })

    it('deletes active workspace and reloads to personal', async () => {
      mockWorkspaceAuthStore.initializeFromSession.mockReturnValue(true)
      mockWorkspaceAuthStore.currentWorkspace = mockTeamWorkspace

      const store = useTeamWorkspaceStore()
      await store.initialize()

      expect(store.activeWorkspaceId).toBe(mockTeamWorkspace.id)

      await store.deleteWorkspace()

      expect(mockWorkspaceApi.delete).toHaveBeenCalledWith(mockTeamWorkspace.id)
      expect(mockWorkspaceAuthStore.clearWorkspaceContext).toHaveBeenCalled()
      expect(mockReload).toHaveBeenCalled()
    })

    it('throws when trying to delete personal workspace', async () => {
      const store = useTeamWorkspaceStore()
      await store.initialize()

      await expect(
        store.deleteWorkspace(mockPersonalWorkspace.id)
      ).rejects.toThrow('Cannot delete personal workspace')
    })

    it('throws when workspace not found', async () => {
      const store = useTeamWorkspaceStore()
      await store.initialize()

      await expect(store.deleteWorkspace('non-existent')).rejects.toThrow(
        'Workspace not found'
      )
    })
  })

  describe('renameWorkspace', () => {
    it('updates workspace name locally', async () => {
      mockWorkspaceApi.update.mockResolvedValue({
        ...mockTeamWorkspace,
        name: 'Renamed Workspace'
      })

      const store = useTeamWorkspaceStore()
      await store.initialize()

      await store.renameWorkspace(mockTeamWorkspace.id, 'Renamed Workspace')

      expect(mockWorkspaceApi.update).toHaveBeenCalledWith(
        mockTeamWorkspace.id,
        { name: 'Renamed Workspace' }
      )

      const updated = store.workspaces.find(
        (w) => w.id === mockTeamWorkspace.id
      )
      expect(updated?.name).toBe('Renamed Workspace')
    })
  })

  describe('leaveWorkspace', () => {
    it('leaves workspace and reloads to personal', async () => {
      mockWorkspaceAuthStore.initializeFromSession.mockReturnValue(true)
      mockWorkspaceAuthStore.currentWorkspace = mockMemberWorkspace
      mockWorkspaceApi.list.mockResolvedValue({
        workspaces: [mockPersonalWorkspace, mockMemberWorkspace]
      })

      const store = useTeamWorkspaceStore()
      await store.initialize()

      await store.leaveWorkspace()

      expect(mockWorkspaceApi.leave).toHaveBeenCalled()
      expect(mockWorkspaceAuthStore.clearWorkspaceContext).toHaveBeenCalled()
      expect(mockReload).toHaveBeenCalled()
    })

    it('throws when trying to leave personal workspace', async () => {
      const store = useTeamWorkspaceStore()
      await store.initialize()

      await expect(store.leaveWorkspace()).rejects.toThrow(
        'Cannot leave personal workspace'
      )
    })
  })

  describe('computed properties', () => {
    it('activeWorkspace returns correct workspace', async () => {
      mockWorkspaceAuthStore.initializeFromSession.mockReturnValue(true)
      mockWorkspaceAuthStore.currentWorkspace = mockTeamWorkspace

      const store = useTeamWorkspaceStore()
      await store.initialize()

      expect(store.activeWorkspace?.id).toBe(mockTeamWorkspace.id)
      expect(store.activeWorkspace?.name).toBe(mockTeamWorkspace.name)
    })

    it('personalWorkspace returns personal workspace', async () => {
      const store = useTeamWorkspaceStore()
      await store.initialize()

      expect(store.personalWorkspace?.id).toBe(mockPersonalWorkspace.id)
      expect(store.personalWorkspace?.type).toBe('personal')
    })

    it('isInPersonalWorkspace returns true when in personal', async () => {
      const store = useTeamWorkspaceStore()
      await store.initialize()

      expect(store.isInPersonalWorkspace).toBe(true)
    })

    it('isInPersonalWorkspace returns false when in team', async () => {
      mockWorkspaceAuthStore.initializeFromSession.mockReturnValue(true)
      mockWorkspaceAuthStore.currentWorkspace = mockTeamWorkspace

      const store = useTeamWorkspaceStore()
      await store.initialize()

      expect(store.isInPersonalWorkspace).toBe(false)
    })

    it('sharedWorkspaces excludes personal workspace', async () => {
      const store = useTeamWorkspaceStore()
      await store.initialize()

      expect(store.sharedWorkspaces).toHaveLength(1)
      expect(store.sharedWorkspaces[0].id).toBe(mockTeamWorkspace.id)
    })

    it('ownedWorkspacesCount counts owned workspaces', async () => {
      mockWorkspaceApi.list.mockResolvedValue({
        workspaces: [
          mockPersonalWorkspace,
          mockTeamWorkspace,
          mockMemberWorkspace
        ]
      })

      const store = useTeamWorkspaceStore()
      await store.initialize()

      expect(store.ownedWorkspacesCount).toBe(2)
    })

    it('canCreateWorkspace respects limit', async () => {
      const manyWorkspaces = Array.from({ length: 10 }, (_, i) => ({
        id: `ws-owned-${i}`,
        name: `Owned ${i}`,
        type: 'team' as const,
        role: 'owner' as const,
        created_at: `2026-${String(i + 1).padStart(2, '0')}-01T00:00:00Z`,
        joined_at: `2026-${String(i + 1).padStart(2, '0')}-01T00:00:00Z`
      }))

      mockWorkspaceApi.list.mockResolvedValue({
        workspaces: [mockPersonalWorkspace, ...manyWorkspaces]
      })

      const store = useTeamWorkspaceStore()
      await store.initialize()

      expect(store.ownedWorkspacesCount).toBe(11)
      expect(store.canCreateWorkspace).toBe(false)
    })
  })

  describe('member actions', () => {
    it('fetchMembers updates active workspace members', async () => {
      const mockMembers = [
        {
          id: 'user-1',
          name: 'User One',
          email: 'one@test.com',
          joined_at: '2024-01-01T00:00:00Z'
        },
        {
          id: 'user-2',
          name: 'User Two',
          email: 'two@test.com',
          joined_at: '2024-01-02T00:00:00Z'
        }
      ]
      mockWorkspaceApi.listMembers.mockResolvedValue({
        members: mockMembers,
        pagination: { offset: 0, limit: 50, total: 2 }
      })
      mockWorkspaceAuthStore.initializeFromSession.mockReturnValue(true)
      mockWorkspaceAuthStore.currentWorkspace = mockTeamWorkspace

      const store = useTeamWorkspaceStore()
      await store.initialize()

      const result = await store.fetchMembers()

      expect(result).toHaveLength(2)
      expect(store.members).toHaveLength(2)
      expect(store.members[0].name).toBe('User One')
    })

    it('fetchMembers returns empty for personal workspace', async () => {
      const store = useTeamWorkspaceStore()
      await store.initialize()

      const result = await store.fetchMembers()

      expect(result).toEqual([])
      expect(mockWorkspaceApi.listMembers).not.toHaveBeenCalled()
    })

    it('removeMember removes from local list', async () => {
      const mockMembers = [
        {
          id: 'user-1',
          name: 'User One',
          email: 'one@test.com',
          joined_at: '2024-01-01T00:00:00Z'
        },
        {
          id: 'user-2',
          name: 'User Two',
          email: 'two@test.com',
          joined_at: '2024-01-02T00:00:00Z'
        }
      ]
      mockWorkspaceApi.listMembers.mockResolvedValue({
        members: mockMembers,
        pagination: { offset: 0, limit: 50, total: 2 }
      })
      mockWorkspaceAuthStore.initializeFromSession.mockReturnValue(true)
      mockWorkspaceAuthStore.currentWorkspace = mockTeamWorkspace

      const store = useTeamWorkspaceStore()
      await store.initialize()
      await store.fetchMembers()

      expect(store.members).toHaveLength(2)

      await store.removeMember('user-1')

      expect(mockWorkspaceApi.removeMember).toHaveBeenCalledWith('user-1')
      expect(store.members).toHaveLength(1)
      expect(store.members[0].id).toBe('user-2')
    })
  })

  describe('invite actions', () => {
    it('fetchPendingInvites updates active workspace invites', async () => {
      const mockInvites = [
        {
          id: 'inv-1',
          email: 'invite@test.com',
          token: 'token-abc',
          invited_at: '2024-01-01T00:00:00Z',
          expires_at: '2024-01-08T00:00:00Z'
        }
      ]
      mockWorkspaceApi.listInvites.mockResolvedValue({ invites: mockInvites })
      mockWorkspaceAuthStore.initializeFromSession.mockReturnValue(true)
      mockWorkspaceAuthStore.currentWorkspace = mockTeamWorkspace

      const store = useTeamWorkspaceStore()
      await store.initialize()

      const result = await store.fetchPendingInvites()

      expect(result).toHaveLength(1)
      expect(store.pendingInvites).toHaveLength(1)
      expect(store.pendingInvites[0].email).toBe('invite@test.com')
    })

    it('createInvite adds to local list', async () => {
      const newInvite = {
        id: 'inv-new',
        email: 'new@test.com',
        token: 'token-new',
        invited_at: '2024-01-01T00:00:00Z',
        expires_at: '2024-01-08T00:00:00Z'
      }
      mockWorkspaceApi.createInvite.mockResolvedValue(newInvite)
      mockWorkspaceAuthStore.initializeFromSession.mockReturnValue(true)
      mockWorkspaceAuthStore.currentWorkspace = mockTeamWorkspace

      const store = useTeamWorkspaceStore()
      await store.initialize()

      const result = await store.createInvite('new@test.com')

      expect(mockWorkspaceApi.createInvite).toHaveBeenCalledWith({
        email: 'new@test.com'
      })
      expect(result.email).toBe('new@test.com')
      expect(store.pendingInvites).toContainEqual(
        expect.objectContaining({ email: 'new@test.com' })
      )
    })

    it('revokeInvite removes from local list', async () => {
      const mockInvites = [
        {
          id: 'inv-1',
          email: 'one@test.com',
          token: 'token-1',
          invited_at: '2024-01-01T00:00:00Z',
          expires_at: '2024-01-08T00:00:00Z'
        },
        {
          id: 'inv-2',
          email: 'two@test.com',
          token: 'token-2',
          invited_at: '2024-01-01T00:00:00Z',
          expires_at: '2024-01-08T00:00:00Z'
        }
      ]
      mockWorkspaceApi.listInvites.mockResolvedValue({ invites: mockInvites })
      mockWorkspaceAuthStore.initializeFromSession.mockReturnValue(true)
      mockWorkspaceAuthStore.currentWorkspace = mockTeamWorkspace

      const store = useTeamWorkspaceStore()
      await store.initialize()
      await store.fetchPendingInvites()

      await store.revokeInvite('inv-1')

      expect(mockWorkspaceApi.revokeInvite).toHaveBeenCalledWith('inv-1')
      expect(store.pendingInvites).toHaveLength(1)
      expect(store.pendingInvites[0].id).toBe('inv-2')
    })

    it('acceptInvite refreshes workspace list', async () => {
      mockWorkspaceApi.acceptInvite.mockResolvedValue({
        workspace_id: 'ws-joined',
        workspace_name: 'Joined Workspace'
      })

      const store = useTeamWorkspaceStore()
      await store.initialize()

      const result = await store.acceptInvite('invite-token')

      expect(mockWorkspaceApi.acceptInvite).toHaveBeenCalledWith('invite-token')
      expect(result.workspaceId).toBe('ws-joined')
      expect(result.workspaceName).toBe('Joined Workspace')
      expect(mockWorkspaceApi.list).toHaveBeenCalledTimes(2)
    })
  })

  describe('invite link helpers', () => {
    it('getInviteLink returns link for existing invite', async () => {
      const mockInvites = [
        {
          id: 'inv-1',
          email: 'test@test.com',
          token: 'secret-token',
          invited_at: '2024-01-01T00:00:00Z',
          expires_at: '2024-01-08T00:00:00Z'
        }
      ]
      mockWorkspaceApi.listInvites.mockResolvedValue({ invites: mockInvites })
      mockWorkspaceAuthStore.initializeFromSession.mockReturnValue(true)
      mockWorkspaceAuthStore.currentWorkspace = mockTeamWorkspace

      const store = useTeamWorkspaceStore()
      await store.initialize()
      await store.fetchPendingInvites()

      const link = store.getInviteLink('inv-1')

      expect(link).toContain('?invite=secret-token')
    })

    it('getInviteLink returns null for non-existent invite', async () => {
      const store = useTeamWorkspaceStore()
      await store.initialize()

      const link = store.getInviteLink('non-existent')

      expect(link).toBeNull()
    })

    it('createInviteLink creates invite and returns link', async () => {
      const newInvite = {
        id: 'inv-new',
        email: 'new@test.com',
        token: 'new-token',
        invited_at: '2024-01-01T00:00:00Z',
        expires_at: '2024-01-08T00:00:00Z'
      }
      mockWorkspaceApi.createInvite.mockResolvedValue(newInvite)
      mockWorkspaceAuthStore.initializeFromSession.mockReturnValue(true)
      mockWorkspaceAuthStore.currentWorkspace = mockTeamWorkspace

      const store = useTeamWorkspaceStore()
      await store.initialize()

      const link = await store.createInviteLink('new@test.com')

      expect(link).toContain('?invite=new-token')
    })
  })

  describe('cleanup', () => {
    it('destroy calls workspaceAuthStore.destroy', async () => {
      const store = useTeamWorkspaceStore()
      await store.initialize()

      store.destroy()

      expect(mockWorkspaceAuthStore.destroy).toHaveBeenCalled()
    })
  })

  describe('totalMemberSlots and isInviteLimitReached', () => {
    it('calculates total slots from members and invites', async () => {
      const mockMembers = [
        {
          id: 'user-1',
          name: 'User One',
          email: 'one@test.com',
          joined_at: '2024-01-01T00:00:00Z'
        }
      ]
      const mockInvites = [
        {
          id: 'inv-1',
          email: 'invite@test.com',
          token: 'token-1',
          invited_at: '2024-01-01T00:00:00Z',
          expires_at: '2024-01-08T00:00:00Z'
        },
        {
          id: 'inv-2',
          email: 'invite2@test.com',
          token: 'token-2',
          invited_at: '2024-01-01T00:00:00Z',
          expires_at: '2024-01-08T00:00:00Z'
        }
      ]
      mockWorkspaceApi.listMembers.mockResolvedValue({
        members: mockMembers,
        pagination: { offset: 0, limit: 50, total: 1 }
      })
      mockWorkspaceApi.listInvites.mockResolvedValue({ invites: mockInvites })
      mockWorkspaceAuthStore.initializeFromSession.mockReturnValue(true)
      mockWorkspaceAuthStore.currentWorkspace = mockTeamWorkspace

      const store = useTeamWorkspaceStore()
      await store.initialize()
      await store.fetchMembers()
      await store.fetchPendingInvites()

      expect(store.totalMemberSlots).toBe(3)
      expect(store.isInviteLimitReached).toBe(false)
    })

    it('isInviteLimitReached returns true at 50 slots', async () => {
      const mockMembers = Array.from({ length: 48 }, (_, i) => ({
        id: `user-${i}`,
        name: `User ${i}`,
        email: `user${i}@test.com`,
        joined_at: '2024-01-01T00:00:00Z'
      }))
      const mockInvites = [
        {
          id: 'inv-1',
          email: 'invite1@test.com',
          token: 'token-1',
          invited_at: '2024-01-01T00:00:00Z',
          expires_at: '2024-01-08T00:00:00Z'
        },
        {
          id: 'inv-2',
          email: 'invite2@test.com',
          token: 'token-2',
          invited_at: '2024-01-01T00:00:00Z',
          expires_at: '2024-01-08T00:00:00Z'
        }
      ]
      mockWorkspaceApi.listMembers.mockResolvedValue({
        members: mockMembers,
        pagination: { offset: 0, limit: 50, total: 48 }
      })
      mockWorkspaceApi.listInvites.mockResolvedValue({ invites: mockInvites })
      mockWorkspaceAuthStore.initializeFromSession.mockReturnValue(true)
      mockWorkspaceAuthStore.currentWorkspace = mockTeamWorkspace

      const store = useTeamWorkspaceStore()
      await store.initialize()
      await store.fetchMembers()
      await store.fetchPendingInvites()

      expect(store.totalMemberSlots).toBe(50)
      expect(store.isInviteLimitReached).toBe(true)
    })
  })
})

describe('sortWorkspaces', () => {
  it('places personal first, then sorts ascending by created_at for owners and joined_at for members', () => {
    const input = [
      {
        created_at: '2026-06-01T00:00:00Z',
        id: 'w-team-new-owner',
        joined_at: '2026-01-01T00:00:00Z',
        name: 'Newest Owner Team',
        role: 'owner' as const,
        type: 'team' as const
      },
      {
        created_at: '2026-12-01T00:00:00Z',
        id: 'w-personal',
        joined_at: '2026-12-01T00:00:00Z',
        name: 'Personal Workspace',
        role: 'owner' as const,
        type: 'personal' as const
      },
      {
        created_at: '2026-01-01T00:00:00Z',
        id: 'w-team-member',
        joined_at: '2026-04-01T00:00:00Z',
        name: 'Member Team',
        role: 'member' as const,
        type: 'team' as const
      },
      {
        created_at: '2026-02-01T00:00:00Z',
        id: 'w-team-old-owner',
        joined_at: '2026-09-01T00:00:00Z',
        name: 'Oldest Owner Team',
        role: 'owner' as const,
        type: 'team' as const
      }
    ]

    expect(sortWorkspaces(input).map((w) => w.id)).toEqual([
      'w-personal',
      'w-team-old-owner',
      'w-team-member',
      'w-team-new-owner'
    ])
  })
})
