import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useWorkspaceSwitch } from '@/platform/workspace/composables/useWorkspaceSwitch'
import type { WorkspaceWithRole } from '@/platform/workspace/api/workspaceApi'

const mockSwitchWorkspace = vi.hoisted(() => vi.fn())
const mockActiveWorkspace = vi.hoisted(() => ({
  value: null as WorkspaceWithRole | null
}))

vi.mock('@/platform/workspace/stores/teamWorkspaceStore', () => ({
  useTeamWorkspaceStore: () => ({
    switchWorkspace: mockSwitchWorkspace
  })
}))

vi.mock('pinia', () => ({
  storeToRefs: () => ({
    activeWorkspace: mockActiveWorkspace
  })
}))

const mockModifiedWorkflows = vi.hoisted(
  () => [] as Array<{ isModified: boolean }>
)

vi.mock('@/platform/workflow/management/stores/workflowStore', () => ({
  useWorkflowStore: () => ({
    get modifiedWorkflows() {
      return mockModifiedWorkflows
    }
  })
}))

const mockConfirm = vi.hoisted(() => vi.fn())

vi.mock('@/services/dialogService', () => ({
  useDialogService: () => ({
    confirm: mockConfirm
  })
}))

vi.mock('vue-i18n', () => ({
  useI18n: () => ({
    t: (key: string) => key
  })
}))

describe('useWorkspaceSwitch', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockActiveWorkspace.value = {
      id: 'workspace-1',
      name: 'Test Workspace',
      type: 'personal',
      role: 'owner',
      created_at: '2026-01-01T00:00:00Z',
      joined_at: '2026-01-01T00:00:00Z'
    }
    mockModifiedWorkflows.length = 0
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  describe('hasUnsavedChanges', () => {
    it('returns true when there are modified workflows', () => {
      mockModifiedWorkflows.push({ isModified: true })
      const { hasUnsavedChanges } = useWorkspaceSwitch()

      expect(hasUnsavedChanges()).toBe(true)
    })

    it('returns true when multiple workflows are modified', () => {
      mockModifiedWorkflows.push({ isModified: true }, { isModified: true })
      const { hasUnsavedChanges } = useWorkspaceSwitch()

      expect(hasUnsavedChanges()).toBe(true)
    })

    it('returns false when no workflows are modified', () => {
      mockModifiedWorkflows.length = 0
      const { hasUnsavedChanges } = useWorkspaceSwitch()

      expect(hasUnsavedChanges()).toBe(false)
    })
  })

  describe('switchWithConfirmation', () => {
    it('returns true immediately if switching to the same workspace', async () => {
      const { switchWithConfirmation } = useWorkspaceSwitch()

      const result = await switchWithConfirmation('workspace-1')

      expect(result).toBe(true)
      expect(mockSwitchWorkspace).not.toHaveBeenCalled()
      expect(mockConfirm).not.toHaveBeenCalled()
    })

    it('switches directly without dialog when no unsaved changes', async () => {
      mockModifiedWorkflows.length = 0
      mockSwitchWorkspace.mockResolvedValue(undefined)
      const { switchWithConfirmation } = useWorkspaceSwitch()

      const result = await switchWithConfirmation('workspace-2')

      expect(result).toBe(true)
      expect(mockConfirm).not.toHaveBeenCalled()
      expect(mockSwitchWorkspace).toHaveBeenCalledWith('workspace-2')
    })

    it('shows confirmation dialog when there are unsaved changes', async () => {
      mockModifiedWorkflows.push({ isModified: true })
      mockConfirm.mockResolvedValue(true)
      mockSwitchWorkspace.mockResolvedValue(undefined)
      const { switchWithConfirmation } = useWorkspaceSwitch()

      await switchWithConfirmation('workspace-2')

      expect(mockConfirm).toHaveBeenCalledWith({
        title: 'workspace.unsavedChanges.title',
        message: 'workspace.unsavedChanges.message',
        type: 'dirtyClose'
      })
    })

    it('returns false if user cancels the confirmation dialog', async () => {
      mockModifiedWorkflows.push({ isModified: true })
      mockConfirm.mockResolvedValue(false)
      const { switchWithConfirmation } = useWorkspaceSwitch()

      const result = await switchWithConfirmation('workspace-2')

      expect(result).toBe(false)
      expect(mockSwitchWorkspace).not.toHaveBeenCalled()
    })

    it('calls switchWorkspace after user confirms', async () => {
      mockModifiedWorkflows.push({ isModified: true })
      mockConfirm.mockResolvedValue(true)
      mockSwitchWorkspace.mockResolvedValue(undefined)
      const { switchWithConfirmation } = useWorkspaceSwitch()

      const result = await switchWithConfirmation('workspace-2')

      expect(result).toBe(true)
      expect(mockSwitchWorkspace).toHaveBeenCalledWith('workspace-2')
    })

    it('returns false if switchWorkspace throws an error', async () => {
      mockModifiedWorkflows.length = 0
      mockSwitchWorkspace.mockRejectedValue(new Error('Switch failed'))
      const { switchWithConfirmation } = useWorkspaceSwitch()

      const result = await switchWithConfirmation('workspace-2')

      expect(result).toBe(false)
    })
  })
})
