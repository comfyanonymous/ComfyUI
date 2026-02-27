import { defineStore } from 'pinia'
import { computed, ref, shallowRef } from 'vue'

import { WORKSPACE_STORAGE_KEYS } from '@/platform/workspace/workspaceConstants'
import { clearPreservedQuery } from '@/platform/navigation/preservedQueryManager'
import { PRESERVED_QUERY_NAMESPACES } from '@/platform/navigation/preservedQueryNamespaces'
import { useWorkspaceAuthStore } from '@/platform/workspace/stores/workspaceAuthStore'

import type {
  ListMembersParams,
  Member,
  PendingInvite as ApiPendingInvite,
  SubscriptionTier,
  WorkspaceWithRole
} from '../api/workspaceApi'
import { workspaceApi } from '../api/workspaceApi'

export interface WorkspaceMember {
  id: string
  name: string
  email: string
  joinDate: Date
  role: 'owner' | 'member'
}

export interface PendingInvite {
  id: string
  email: string
  token: string
  inviteDate: Date
  expiryDate: Date
}

type SubscriptionPlan = string | null

interface WorkspaceState extends WorkspaceWithRole {
  isSubscribed: boolean
  subscriptionPlan: SubscriptionPlan
  subscriptionTier: SubscriptionTier | null
  members: WorkspaceMember[]
  pendingInvites: PendingInvite[]
}

type InitState = 'uninitialized' | 'loading' | 'ready' | 'error'

function mapApiMemberToWorkspaceMember(member: Member): WorkspaceMember {
  return {
    id: member.id,
    name: member.name,
    email: member.email,
    joinDate: new Date(member.joined_at),
    role: member.role
  }
}

function mapApiInviteToPendingInvite(invite: ApiPendingInvite): PendingInvite {
  return {
    id: invite.id,
    email: invite.email,
    token: invite.token,
    inviteDate: new Date(invite.invited_at),
    expiryDate: new Date(invite.expires_at)
  }
}

function createWorkspaceState(workspace: WorkspaceWithRole): WorkspaceState {
  return {
    ...workspace,
    // Personal workspaces use user-scoped subscription from useSubscription()
    isSubscribed:
      workspace.type === 'personal' || !!workspace.subscription_tier,
    subscriptionPlan: null,
    subscriptionTier: workspace.subscription_tier ?? null,
    members: [],
    pendingInvites: []
  }
}

export function sortWorkspaces<T extends WorkspaceWithRole>(list: T[]): T[] {
  return [...list].sort((a, b) => {
    if (a.type === 'personal') return -1
    if (b.type === 'personal') return 1
    const dateA = a.role === 'owner' ? a.created_at : a.joined_at
    const dateB = b.role === 'owner' ? b.created_at : b.joined_at
    return dateA.localeCompare(dateB)
  })
}

function getLastWorkspaceId(): string | null {
  try {
    return localStorage.getItem(WORKSPACE_STORAGE_KEYS.LAST_WORKSPACE_ID)
  } catch {
    return null
  }
}

function setLastWorkspaceId(workspaceId: string): void {
  try {
    localStorage.setItem(WORKSPACE_STORAGE_KEYS.LAST_WORKSPACE_ID, workspaceId)
  } catch {
    console.warn('Failed to persist last workspace ID to localStorage')
  }
}

const MAX_OWNED_WORKSPACES = 10
const MAX_WORKSPACE_MEMBERS = 50
const MAX_INIT_RETRIES = 3
const BASE_RETRY_DELAY_MS = 1000

export const useTeamWorkspaceStore = defineStore('teamWorkspace', () => {
  const initState = ref<InitState>('uninitialized')
  const workspaces = shallowRef<WorkspaceState[]>([])
  const activeWorkspaceId = ref<string | null>(null)
  const error = ref<Error | null>(null)

  const isCreating = ref(false)
  const isDeleting = ref(false)
  const isSwitching = ref(false)
  const isFetchingWorkspaces = ref(false)

  const activeWorkspace = computed(
    () => workspaces.value.find((w) => w.id === activeWorkspaceId.value) ?? null
  )

  const personalWorkspace = computed(
    () => workspaces.value.find((w) => w.type === 'personal') ?? null
  )

  const isInPersonalWorkspace = computed(
    () => activeWorkspace.value?.type === 'personal'
  )

  const sharedWorkspaces = computed(() =>
    workspaces.value.filter((w) => w.type !== 'personal')
  )

  const ownedWorkspacesCount = computed(
    () => workspaces.value.filter((w) => w.role === 'owner').length
  )

  const canCreateWorkspace = computed(
    () => ownedWorkspacesCount.value < MAX_OWNED_WORKSPACES
  )

  const members = computed<WorkspaceMember[]>(
    () => activeWorkspace.value?.members ?? []
  )

  const pendingInvites = computed<PendingInvite[]>(
    () => activeWorkspace.value?.pendingInvites ?? []
  )

  const totalMemberSlots = computed(
    () => members.value.length + pendingInvites.value.length
  )

  const isInviteLimitReached = computed(
    () => totalMemberSlots.value >= MAX_WORKSPACE_MEMBERS
  )

  const workspaceId = computed(() => activeWorkspace.value?.id ?? null)

  const workspaceName = computed(() => activeWorkspace.value?.name ?? '')

  const isWorkspaceSubscribed = computed(
    () => activeWorkspace.value?.isSubscribed ?? false
  )

  const subscriptionPlan = computed(
    () => activeWorkspace.value?.subscriptionPlan ?? null
  )

  function updateWorkspace(
    workspaceId: string,
    updates: Partial<WorkspaceState>
  ) {
    const index = workspaces.value.findIndex((w) => w.id === workspaceId)
    if (index === -1) return

    const current = workspaces.value[index]
    const updated = { ...current, ...updates }
    workspaces.value = [
      ...workspaces.value.slice(0, index),
      updated,
      ...workspaces.value.slice(index + 1)
    ]
  }

  function updateActiveWorkspace(updates: Partial<WorkspaceState>) {
    if (!activeWorkspaceId.value) return
    updateWorkspace(activeWorkspaceId.value, updates)
  }

  /**
   * Initialize the workspace store.
   * Fetches workspaces and resolves the active workspace from session/localStorage.
   * Delegates token management to workspaceAuthStore.
   * Retries on transient failures with exponential backoff.
   * Call once on app boot.
   */
  async function initialize(): Promise<void> {
    if (initState.value !== 'uninitialized') return

    initState.value = 'loading'
    isFetchingWorkspaces.value = true
    error.value = null

    const workspaceAuthStore = useWorkspaceAuthStore()

    for (let attempt = 0; attempt <= MAX_INIT_RETRIES; attempt++) {
      try {
        // 1. Try to restore workspace context from session (page refresh case)
        const hasValidSession = workspaceAuthStore.initializeFromSession()

        if (hasValidSession && workspaceAuthStore.currentWorkspace) {
          // Valid session exists - fetch workspace list and verify access
          const response = await workspaceApi.list()
          workspaces.value = sortWorkspaces(
            response.workspaces.map(createWorkspaceState)
          )

          if (workspaces.value.length === 0) {
            throw new Error('No workspaces available')
          }

          // Verify session workspace exists in fetched list
          const sessionWorkspaceId = workspaceAuthStore.currentWorkspace.id
          const sessionWorkspaceExists = workspaces.value.some(
            (w) => w.id === sessionWorkspaceId
          )

          if (sessionWorkspaceExists) {
            activeWorkspaceId.value = sessionWorkspaceId
            initState.value = 'ready'
            isFetchingWorkspaces.value = false
            return
          }

          // Session workspace not found (deleted/access revoked) - fallback to default
          workspaceAuthStore.clearWorkspaceContext()

          const personal = workspaces.value.find((w) => w.type === 'personal')
          const fallbackWorkspaceId = personal?.id ?? workspaces.value[0].id

          try {
            await workspaceAuthStore.switchWorkspace(fallbackWorkspaceId)
          } catch {
            console.error(
              '[teamWorkspaceStore] Token exchange failed during fallback'
            )
          }

          activeWorkspaceId.value = fallbackWorkspaceId
          setLastWorkspaceId(fallbackWorkspaceId)
          initState.value = 'ready'
          isFetchingWorkspaces.value = false
          return
        }

        // 2. No valid session - fetch workspaces and pick default
        const response = await workspaceApi.list()
        workspaces.value = sortWorkspaces(
          response.workspaces.map(createWorkspaceState)
        )

        if (workspaces.value.length === 0) {
          throw new Error('No workspaces available')
        }

        // 3. Determine target workspace (priority: localStorage > personal)
        let targetWorkspaceId: string | null = null

        const lastId = getLastWorkspaceId()
        if (lastId && workspaces.value.some((w) => w.id === lastId)) {
          targetWorkspaceId = lastId
        }

        if (!targetWorkspaceId) {
          const personal = workspaces.value.find((w) => w.type === 'personal')
          targetWorkspaceId = personal?.id ?? workspaces.value[0].id
        }

        // 4. Exchange Firebase token for workspace token
        try {
          await workspaceAuthStore.switchWorkspace(targetWorkspaceId)
        } catch {
          // Log but don't fail initialization - API calls will fall back to Firebase token
          console.error(
            '[teamWorkspaceStore] Token exchange failed during init'
          )
        }

        // 5. Set active workspace
        activeWorkspaceId.value = targetWorkspaceId
        setLastWorkspaceId(targetWorkspaceId)

        initState.value = 'ready'
        isFetchingWorkspaces.value = false
        return
      } catch (e) {
        const isNoWorkspacesError =
          e instanceof Error && e.message === 'No workspaces available'

        // Don't retry on permanent errors (no workspaces available)
        if (isNoWorkspacesError || attempt >= MAX_INIT_RETRIES) {
          error.value = e instanceof Error ? e : new Error('Unknown error')
          initState.value = 'error'
          isFetchingWorkspaces.value = false
          throw e
        }

        // Retry with exponential backoff for transient errors
        const delay = BASE_RETRY_DELAY_MS * Math.pow(2, attempt)
        const errorMessage = e instanceof Error ? e.message : String(e)
        console.warn(
          `[teamWorkspaceStore] Init failed (attempt ${attempt + 1}/${MAX_INIT_RETRIES + 1}), retrying in ${delay}ms: ${errorMessage}`
        )
        await new Promise((resolve) => setTimeout(resolve, delay))
      }
    }

    isFetchingWorkspaces.value = false
  }

  /**
   * Re-fetch workspaces from API without changing active workspace.
   */
  async function refreshWorkspaces(): Promise<void> {
    isFetchingWorkspaces.value = true
    try {
      const response = await workspaceApi.list()
      workspaces.value = sortWorkspaces(
        response.workspaces.map(createWorkspaceState)
      )
    } finally {
      isFetchingWorkspaces.value = false
    }
  }

  /**
   * Switch to a different workspace.
   * Clears workspace context and reloads the page.
   */
  async function switchWorkspace(workspaceId: string): Promise<void> {
    if (workspaceId === activeWorkspaceId.value) return

    const workspaceAuthStore = useWorkspaceAuthStore()

    isSwitching.value = true

    try {
      // Verify workspace exists in our list (user has access)
      const workspace = workspaces.value.find((w) => w.id === workspaceId)
      if (!workspace) {
        // Workspace not in list - try refetching in case it was added
        await refreshWorkspaces()
        const refreshedWorkspace = workspaces.value.find(
          (w) => w.id === workspaceId
        )
        if (!refreshedWorkspace) {
          throw new Error('Workspace not found or access denied')
        }
      }

      // Clear current workspace context and persist new workspace ID
      workspaceAuthStore.clearWorkspaceContext()
      setLastWorkspaceId(workspaceId)

      // Reload to reinitialize with new workspace
      window.location.reload()
      // Code after this won't run (page reloads)
    } catch (e) {
      isSwitching.value = false
      throw e
    }
  }

  /**
   * Create a new workspace and switch to it.
   */
  async function createWorkspace(name: string): Promise<WorkspaceState> {
    const workspaceAuthStore = useWorkspaceAuthStore()

    isCreating.value = true

    try {
      const newWorkspace = await workspaceApi.create({ name })
      const workspaceState = createWorkspaceState(newWorkspace)

      // Add to local list
      workspaces.value = [...workspaces.value, workspaceState]

      // Clear context and switch to new workspace
      workspaceAuthStore.clearWorkspaceContext()
      // Clear any preserved invite query to prevent stale invites from being
      // processed after the reload (prevents owner adding themselves as member)
      clearPreservedQuery(PRESERVED_QUERY_NAMESPACES.INVITE)
      setLastWorkspaceId(newWorkspace.id)
      window.location.reload()

      // Code after this won't run (page reloads)
      return workspaceState
    } catch (e) {
      isCreating.value = false
      throw e
    }
  }

  /**
   * Delete a workspace.
   * If deleting active workspace, switches to personal.
   */
  async function deleteWorkspace(workspaceId?: string): Promise<void> {
    const targetId = workspaceId ?? activeWorkspaceId.value
    if (!targetId) throw new Error('No workspace to delete')

    const workspace = workspaces.value.find((w) => w.id === targetId)
    if (!workspace) throw new Error('Workspace not found')
    if (workspace.type === 'personal') {
      throw new Error('Cannot delete personal workspace')
    }

    const workspaceAuthStore = useWorkspaceAuthStore()

    isDeleting.value = true

    try {
      await workspaceApi.delete(targetId)

      if (targetId === activeWorkspaceId.value) {
        // Deleted active workspace - go to personal
        const personal = personalWorkspace.value
        workspaceAuthStore.clearWorkspaceContext()
        if (personal) {
          setLastWorkspaceId(personal.id)
        }
        window.location.reload()
        // Code after this won't run (page reloads)
      } else {
        // Deleted non-active workspace - just update local list
        workspaces.value = workspaces.value.filter((w) => w.id !== targetId)
        isDeleting.value = false
      }
    } catch (e) {
      isDeleting.value = false
      throw e
    }
  }

  /**
   * Rename a workspace. No reload needed.
   */
  async function renameWorkspace(
    workspaceId: string,
    newName: string
  ): Promise<void> {
    const updated = await workspaceApi.update(workspaceId, { name: newName })
    updateWorkspace(workspaceId, { name: updated.name })
  }

  /**
   * Update workspace name (convenience for current workspace).
   */
  async function updateWorkspaceName(name: string): Promise<void> {
    if (!activeWorkspaceId.value) {
      throw new Error('No active workspace')
    }
    await renameWorkspace(activeWorkspaceId.value, name)
  }

  /**
   * Leave the current workspace.
   * Switches to personal workspace after leaving.
   */
  async function leaveWorkspace(): Promise<void> {
    const current = activeWorkspace.value
    if (!current || current.type === 'personal') {
      throw new Error('Cannot leave personal workspace')
    }

    const workspaceAuthStore = useWorkspaceAuthStore()

    await workspaceApi.leave()

    // Go to personal workspace
    const personal = personalWorkspace.value
    workspaceAuthStore.clearWorkspaceContext()
    if (personal) {
      setLastWorkspaceId(personal.id)
    }
    window.location.reload()
    // Code after this won't run (page reloads)
  }

  /**
   * Fetch members for the current workspace.
   */
  async function fetchMembers(
    params?: ListMembersParams
  ): Promise<WorkspaceMember[]> {
    if (!activeWorkspaceId.value) return []
    if (activeWorkspace.value?.type === 'personal') return []

    const response = await workspaceApi.listMembers(params)
    const members = response.members.map(mapApiMemberToWorkspaceMember)
    updateActiveWorkspace({ members })
    return members
  }

  /**
   * Remove a member from the current workspace.
   */
  async function removeMember(userId: string): Promise<void> {
    await workspaceApi.removeMember(userId)
    const current = activeWorkspace.value
    if (current) {
      updateActiveWorkspace({
        members: current.members.filter((m) => m.id !== userId)
      })
    }
  }

  /**
   * Fetch pending invites for the current workspace.
   */
  async function fetchPendingInvites(): Promise<PendingInvite[]> {
    if (!activeWorkspaceId.value) return []
    if (activeWorkspace.value?.type === 'personal') return []

    const response = await workspaceApi.listInvites()
    const invites = response.invites.map(mapApiInviteToPendingInvite)
    updateActiveWorkspace({ pendingInvites: invites })
    return invites
  }

  /**
   * Create an invite for the current workspace.
   */
  async function createInvite(email: string): Promise<PendingInvite> {
    const response = await workspaceApi.createInvite({ email })
    const invite = mapApiInviteToPendingInvite(response)

    const current = activeWorkspace.value
    if (current) {
      updateActiveWorkspace({
        pendingInvites: [...current.pendingInvites, invite]
      })
    }

    return invite
  }

  /**
   * Revoke a pending invite.
   */
  async function revokeInvite(inviteId: string): Promise<void> {
    await workspaceApi.revokeInvite(inviteId)
    const current = activeWorkspace.value
    if (current) {
      updateActiveWorkspace({
        pendingInvites: current.pendingInvites.filter((i) => i.id !== inviteId)
      })
    }
  }

  /**
   * Accept a workspace invite.
   * Returns workspace info so UI can offer "View Workspace" button.
   */
  async function acceptInvite(
    token: string
  ): Promise<{ workspaceId: string; workspaceName: string }> {
    const response = await workspaceApi.acceptInvite(token)

    // Refresh workspace list to include newly joined workspace
    await refreshWorkspaces()

    return {
      workspaceId: response.workspace_id,
      workspaceName: response.workspace_name
    }
  }

  function buildInviteLink(token: string): string {
    const baseUrl = window.location.origin
    return `${baseUrl}?invite=${encodeURIComponent(token)}`
  }

  /**
   * Get the invite link for a pending invite.
   */
  function getInviteLink(inviteId: string): string | null {
    const invite = activeWorkspace.value?.pendingInvites.find(
      (i) => i.id === inviteId
    )
    return invite ? buildInviteLink(invite.token) : null
  }

  /**
   * Create an invite link for a given email.
   */
  async function createInviteLink(email: string): Promise<string> {
    const invite = await createInvite(email)
    return buildInviteLink(invite.token)
  }

  /**
   * Copy an invite link to clipboard.
   */
  async function copyInviteLink(inviteId: string): Promise<string> {
    const invite = activeWorkspace.value?.pendingInvites.find(
      (i) => i.id === inviteId
    )
    if (!invite) {
      throw new Error('Invite not found')
    }
    const inviteLink = buildInviteLink(invite.token)
    await navigator.clipboard.writeText(inviteLink)
    return inviteLink
  }

  //TODO: when billing lands update this
  function subscribeWorkspace(plan: SubscriptionPlan = 'PRO_MONTHLY') {
    console.warn(plan, 'Billing endpoint has not been added yet.')
  }

  /**
   * Clean up store resources.
   * Delegates to workspaceAuthStore for token cleanup.
   */
  function destroy(): void {
    const workspaceAuthStore = useWorkspaceAuthStore()
    workspaceAuthStore.destroy()
  }

  return {
    // State
    initState,
    workspaces,
    activeWorkspaceId,
    error,
    isCreating,
    isDeleting,
    isSwitching,
    isFetchingWorkspaces,

    // Computed
    activeWorkspace,
    personalWorkspace,
    isInPersonalWorkspace,
    sharedWorkspaces,
    ownedWorkspacesCount,
    canCreateWorkspace,
    members,
    pendingInvites,
    totalMemberSlots,
    isInviteLimitReached,
    workspaceId,
    workspaceName,
    isWorkspaceSubscribed,
    subscriptionPlan,

    // Initialization & Cleanup
    initialize,
    destroy,
    refreshWorkspaces,

    // Workspace Actions
    switchWorkspace,
    createWorkspace,
    deleteWorkspace,
    renameWorkspace,
    updateWorkspaceName,
    leaveWorkspace,

    // Member Actions
    fetchMembers,
    removeMember,

    // Invite Actions
    fetchPendingInvites,
    createInvite,
    revokeInvite,
    acceptInvite,
    getInviteLink,
    createInviteLink,
    copyInviteLink,

    // Subscription
    subscribeWorkspace,
    updateActiveWorkspace
  }
})
