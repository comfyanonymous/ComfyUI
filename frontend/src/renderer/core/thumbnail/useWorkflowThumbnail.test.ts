import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { ComfyWorkflow } from '@/platform/workflow/management/stores/workflowStore'
import { useWorkflowStore } from '@/platform/workflow/management/stores/workflowStore'
import { createGraphThumbnail } from '@/renderer/core/thumbnail/graphThumbnailRenderer'
import { useWorkflowThumbnail } from '@/renderer/core/thumbnail/useWorkflowThumbnail'
import { api } from '@/scripts/api'

vi.mock('@/renderer/core/thumbnail/graphThumbnailRenderer', () => ({
  createGraphThumbnail: vi.fn()
}))

vi.mock('@/scripts/api', () => ({
  api: {
    moveUserData: vi.fn(),
    listUserDataFullInfo: vi.fn(),
    addEventListener: vi.fn(),
    getUserData: vi.fn(),
    storeUserData: vi.fn(),
    apiURL: vi.fn((path: string) => `/api${path}`)
  }
}))

describe('useWorkflowThumbnail', () => {
  let workflowStore: ReturnType<typeof useWorkflowStore>

  beforeEach(() => {
    setActivePinia(createPinia())
    workflowStore = useWorkflowStore()

    // Clear any existing thumbnails from previous tests BEFORE mocking
    const { clearAllThumbnails } = useWorkflowThumbnail()
    clearAllThumbnails()

    // Now set up mocks
    vi.clearAllMocks()

    global.URL.createObjectURL = vi.fn(() => 'data:image/png;base64,test')
    global.URL.revokeObjectURL = vi.fn()

    // Mock API responses
    vi.mocked(api.moveUserData).mockResolvedValue({ status: 200 } as Response)

    // Default createGraphThumbnail to return test value
    vi.mocked(createGraphThumbnail).mockReturnValue(
      'data:image/png;base64,test'
    )
  })

  it('should capture minimap thumbnail', async () => {
    const { createMinimapPreview } = useWorkflowThumbnail()
    const thumbnail = await createMinimapPreview()

    expect(createGraphThumbnail).toHaveBeenCalledOnce()
    expect(thumbnail).toBe('data:image/png;base64,test')
  })

  it('should store and retrieve thumbnails', async () => {
    const { storeThumbnail, getThumbnail } = useWorkflowThumbnail()

    const mockWorkflow = { key: 'test-workflow-key' } as ComfyWorkflow

    await storeThumbnail(mockWorkflow)

    const thumbnail = getThumbnail('test-workflow-key')
    expect(thumbnail).toBe('data:image/png;base64,test')
  })

  it('should clear thumbnail', async () => {
    const { storeThumbnail, getThumbnail, clearThumbnail } =
      useWorkflowThumbnail()

    const mockWorkflow = { key: 'test-workflow-key' } as ComfyWorkflow

    await storeThumbnail(mockWorkflow)

    expect(getThumbnail('test-workflow-key')).toBeDefined()

    clearThumbnail('test-workflow-key')

    expect(URL.revokeObjectURL).toHaveBeenCalledWith(
      'data:image/png;base64,test'
    )
    expect(getThumbnail('test-workflow-key')).toBeUndefined()
  })

  it('should clear all thumbnails', async () => {
    const { storeThumbnail, getThumbnail, clearAllThumbnails } =
      useWorkflowThumbnail()

    const mockWorkflow1 = { key: 'workflow-1' } as ComfyWorkflow
    const mockWorkflow2 = { key: 'workflow-2' } as ComfyWorkflow

    await storeThumbnail(mockWorkflow1)
    await storeThumbnail(mockWorkflow2)

    expect(getThumbnail('workflow-1')).toBeDefined()
    expect(getThumbnail('workflow-2')).toBeDefined()

    clearAllThumbnails()

    expect(URL.revokeObjectURL).toHaveBeenCalledTimes(2)
    expect(getThumbnail('workflow-1')).toBeUndefined()
    expect(getThumbnail('workflow-2')).toBeUndefined()
  })

  it('should automatically handle thumbnail cleanup when workflow is renamed', async () => {
    const { storeThumbnail, getThumbnail, workflowThumbnails } =
      useWorkflowThumbnail()

    // Create a temporary workflow
    const workflow = workflowStore.createTemporary('test-workflow.json')
    const originalKey = workflow.key

    // Store thumbnail for the workflow
    await storeThumbnail(workflow)
    expect(getThumbnail(originalKey)).toBe('data:image/png;base64,test')
    expect(workflowThumbnails.value.size).toBe(1)

    // Rename the workflow - this should automatically handle thumbnail cleanup
    const newPath = 'workflows/renamed-workflow.json'
    await workflowStore.renameWorkflow(workflow, newPath)

    const newKey = workflow.key // The workflow's key should now be the new path

    // The thumbnail should be moved from old key to new key
    expect(getThumbnail(originalKey)).toBeUndefined()
    expect(getThumbnail(newKey)).toBe('data:image/png;base64,test')
    expect(workflowThumbnails.value.size).toBe(1)

    // No URL should be revoked since we're moving the thumbnail, not deleting it
    expect(URL.revokeObjectURL).not.toHaveBeenCalled()
  })

  it('should properly revoke old URL when storing thumbnail over existing one', async () => {
    const { storeThumbnail, getThumbnail } = useWorkflowThumbnail()

    const mockWorkflow = { key: 'test-workflow' } as ComfyWorkflow

    // Store first thumbnail
    await storeThumbnail(mockWorkflow)
    const firstThumbnail = getThumbnail('test-workflow')
    expect(firstThumbnail).toBe('data:image/png;base64,test')

    // Reset the mock to track new calls and create different URL
    vi.clearAllMocks()
    global.URL.createObjectURL = vi.fn(() => 'data:image/png;base64,test2')
    vi.mocked(createGraphThumbnail).mockReturnValue(
      'data:image/png;base64,test2'
    )

    // Store second thumbnail for same workflow - should revoke the first URL
    await storeThumbnail(mockWorkflow)
    const secondThumbnail = getThumbnail('test-workflow')
    expect(secondThumbnail).toBe('data:image/png;base64,test2')

    // URL.revokeObjectURL should have been called for the first thumbnail
    expect(URL.revokeObjectURL).toHaveBeenCalledWith(
      'data:image/png;base64,test'
    )
    expect(URL.revokeObjectURL).toHaveBeenCalledTimes(1)
  })

  it('should clear thumbnail when workflow is deleted', async () => {
    const { storeThumbnail, getThumbnail, workflowThumbnails } =
      useWorkflowThumbnail()

    // Create a workflow and store thumbnail
    const workflow = workflowStore.createTemporary('test-delete.json')
    await storeThumbnail(workflow)

    expect(getThumbnail(workflow.key)).toBe('data:image/png;base64,test')
    expect(workflowThumbnails.value.size).toBe(1)

    // Delete the workflow - this should clear the thumbnail
    await workflowStore.deleteWorkflow(workflow)

    // Thumbnail should be cleared and URL revoked
    expect(getThumbnail(workflow.key)).toBeUndefined()
    expect(workflowThumbnails.value.size).toBe(0)
    expect(URL.revokeObjectURL).toHaveBeenCalledWith(
      'data:image/png;base64,test'
    )
  })

  it('should clear thumbnail when temporary workflow is closed', async () => {
    const { storeThumbnail, getThumbnail, workflowThumbnails } =
      useWorkflowThumbnail()

    // Create a temporary workflow and store thumbnail
    const workflow = workflowStore.createTemporary('temp-workflow.json')
    await storeThumbnail(workflow)

    expect(getThumbnail(workflow.key)).toBe('data:image/png;base64,test')
    expect(workflowThumbnails.value.size).toBe(1)

    // Close the workflow - this should clear the thumbnail for temporary workflows
    await workflowStore.closeWorkflow(workflow)

    // Thumbnail should be cleared and URL revoked
    expect(getThumbnail(workflow.key)).toBeUndefined()
    expect(workflowThumbnails.value.size).toBe(0)
    expect(URL.revokeObjectURL).toHaveBeenCalledWith(
      'data:image/png;base64,test'
    )
  })

  it('should handle multiple renames without leaking', async () => {
    const { storeThumbnail, getThumbnail, workflowThumbnails } =
      useWorkflowThumbnail()

    // Create workflow and store thumbnail
    const workflow = workflowStore.createTemporary('original.json')
    await storeThumbnail(workflow)
    const originalKey = workflow.key

    expect(getThumbnail(originalKey)).toBe('data:image/png;base64,test')
    expect(workflowThumbnails.value.size).toBe(1)

    // Rename multiple times
    await workflowStore.renameWorkflow(workflow, 'workflows/renamed1.json')
    const firstRenameKey = workflow.key

    expect(getThumbnail(originalKey)).toBeUndefined()
    expect(getThumbnail(firstRenameKey)).toBe('data:image/png;base64,test')
    expect(workflowThumbnails.value.size).toBe(1)

    await workflowStore.renameWorkflow(workflow, 'workflows/renamed2.json')
    const secondRenameKey = workflow.key

    expect(getThumbnail(originalKey)).toBeUndefined()
    expect(getThumbnail(firstRenameKey)).toBeUndefined()
    expect(getThumbnail(secondRenameKey)).toBe('data:image/png;base64,test')
    expect(workflowThumbnails.value.size).toBe(1)

    // No URLs should be revoked since we're just moving thumbnails
    expect(URL.revokeObjectURL).not.toHaveBeenCalled()
  })

  it('should handle edge cases like empty keys or invalid operations', async () => {
    const {
      getThumbnail,
      clearThumbnail,
      moveWorkflowThumbnail,
      workflowThumbnails
    } = useWorkflowThumbnail()

    // Test getting non-existent thumbnail
    expect(getThumbnail('non-existent')).toBeUndefined()

    // Test clearing non-existent thumbnail (should not throw)
    expect(() => clearThumbnail('non-existent')).not.toThrow()
    expect(URL.revokeObjectURL).not.toHaveBeenCalled()

    // Test moving non-existent thumbnail (should not throw)
    expect(() => moveWorkflowThumbnail('non-existent', 'target')).not.toThrow()
    expect(workflowThumbnails.value.size).toBe(0)

    // Test moving to same key (should not cause issues)
    const { storeThumbnail } = useWorkflowThumbnail()
    const mockWorkflow = { key: 'test-key' } as ComfyWorkflow
    await storeThumbnail(mockWorkflow)

    expect(workflowThumbnails.value.size).toBe(1)
    moveWorkflowThumbnail('test-key', 'test-key')
    expect(workflowThumbnails.value.size).toBe(1)
    expect(getThumbnail('test-key')).toBe('data:image/png;base64,test')
  })
})
