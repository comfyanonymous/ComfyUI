import { computed } from 'vue'
import { createSharedComposable } from '@vueuse/core'

import type { WorkspaceRole, WorkspaceType } from '../api/workspaceApi'
import { useTeamWorkspaceStore } from '../stores/teamWorkspaceStore'

/** Permission flags for workspace actions */
interface WorkspacePermissions {
  canViewOtherMembers: boolean
  canViewPendingInvites: boolean
  canInviteMembers: boolean
  canManageInvites: boolean
  canRemoveMembers: boolean
  canLeaveWorkspace: boolean
  canAccessWorkspaceMenu: boolean
  canManageSubscription: boolean
  canTopUp: boolean
}

/** UI configuration for workspace role */
interface WorkspaceUIConfig {
  showMembersList: boolean
  showPendingTab: boolean
  showSearch: boolean
  showDateColumn: boolean
  showRoleBadge: boolean
  membersGridCols: string
  pendingGridCols: string
  headerGridCols: string
  showEditWorkspaceMenuItem: boolean
  workspaceMenuAction: 'leave' | 'delete' | null
  workspaceMenuDisabledTooltip: string | null
}

function getPermissions(
  type: WorkspaceType,
  role: WorkspaceRole
): WorkspacePermissions {
  if (type === 'personal') {
    return {
      canViewOtherMembers: false,
      canViewPendingInvites: false,
      canInviteMembers: false,
      canManageInvites: false,
      canRemoveMembers: false,
      canLeaveWorkspace: false,
      canAccessWorkspaceMenu: false,
      canManageSubscription: true,
      canTopUp: true
    }
  }

  if (role === 'owner') {
    return {
      canViewOtherMembers: true,
      canViewPendingInvites: true,
      canInviteMembers: true,
      canManageInvites: true,
      canRemoveMembers: true,
      canLeaveWorkspace: true,
      canAccessWorkspaceMenu: true,
      canManageSubscription: true,
      canTopUp: true
    }
  }

  // member role
  return {
    canViewOtherMembers: true,
    canViewPendingInvites: false,
    canInviteMembers: false,
    canManageInvites: false,
    canRemoveMembers: false,
    canLeaveWorkspace: true,
    canAccessWorkspaceMenu: true,
    canManageSubscription: false,
    canTopUp: false
  }
}

function getUIConfig(
  type: WorkspaceType,
  role: WorkspaceRole
): WorkspaceUIConfig {
  if (type === 'personal') {
    return {
      showMembersList: false,
      showPendingTab: false,
      showSearch: false,
      showDateColumn: false,
      showRoleBadge: false,
      membersGridCols: 'grid-cols-1',
      pendingGridCols: 'grid-cols-[50%_20%_20%_10%]',
      headerGridCols: 'grid-cols-1',
      showEditWorkspaceMenuItem: false,
      workspaceMenuAction: null,
      workspaceMenuDisabledTooltip: null
    }
  }

  if (role === 'owner') {
    return {
      showMembersList: true,
      showPendingTab: true,
      showSearch: true,
      showDateColumn: true,
      showRoleBadge: true,
      membersGridCols: 'grid-cols-[50%_40%_10%]',
      pendingGridCols: 'grid-cols-[50%_20%_20%_10%]',
      headerGridCols: 'grid-cols-[50%_40%_10%]',
      showEditWorkspaceMenuItem: true,
      workspaceMenuAction: 'delete',
      workspaceMenuDisabledTooltip:
        'workspacePanel.menu.deleteWorkspaceDisabledTooltip'
    }
  }

  // member role
  return {
    showMembersList: true,
    showPendingTab: false,
    showSearch: true,
    showDateColumn: true,
    showRoleBadge: true,
    membersGridCols: 'grid-cols-[1fr_auto]',
    pendingGridCols: 'grid-cols-[50%_20%_20%_10%]',
    headerGridCols: 'grid-cols-[1fr_auto]',
    showEditWorkspaceMenuItem: false,
    workspaceMenuAction: 'leave',
    workspaceMenuDisabledTooltip: null
  }
}

/**
 * Internal implementation of UI configuration composable.
 */
function useWorkspaceUIInternal() {
  const store = useTeamWorkspaceStore()

  const workspaceType = computed<WorkspaceType>(
    () => store.activeWorkspace?.type ?? 'personal'
  )

  const workspaceRole = computed<WorkspaceRole>(
    () => store.activeWorkspace?.role ?? 'owner'
  )

  const permissions = computed<WorkspacePermissions>(() =>
    getPermissions(workspaceType.value, workspaceRole.value)
  )

  const uiConfig = computed<WorkspaceUIConfig>(() =>
    getUIConfig(workspaceType.value, workspaceRole.value)
  )

  return {
    // Permissions and config
    permissions,
    uiConfig,
    workspaceType,
    workspaceRole
  }
}

/**
 * UI configuration composable derived from workspace state.
 * Controls what UI elements are visible/enabled based on role and workspace type.
 * Uses createSharedComposable to ensure tab state is shared across components.
 */
export const useWorkspaceUI = createSharedComposable(useWorkspaceUIInternal)
