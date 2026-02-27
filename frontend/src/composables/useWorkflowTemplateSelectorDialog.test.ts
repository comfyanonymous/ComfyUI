import { beforeEach, describe, expect, it, vi } from 'vitest'

const mockDialogService = vi.hoisted(() => ({
  showLayoutDialog: vi.fn()
}))

const mockDialogStore = vi.hoisted(() => ({
  closeDialog: vi.fn()
}))

const mockNewUserService = vi.hoisted(() => ({
  isNewUser: vi.fn()
}))

const mockTelemetry = vi.hoisted(() => ({
  trackTemplateLibraryOpened: vi.fn()
}))

vi.mock('@/services/dialogService', () => ({
  useDialogService: () => mockDialogService
}))

vi.mock('@/stores/dialogStore', () => ({
  useDialogStore: () => mockDialogStore
}))

vi.mock('@/services/useNewUserService', () => ({
  useNewUserService: () => mockNewUserService
}))

vi.mock('@/platform/telemetry', () => ({
  useTelemetry: () => mockTelemetry
}))

vi.mock(
  '@/components/custom/widget/WorkflowTemplateSelectorDialog.vue',
  () => ({
    default: { name: 'MockWorkflowTemplateSelectorDialog' }
  })
)

import { useWorkflowTemplateSelectorDialog } from './useWorkflowTemplateSelectorDialog'

describe('useWorkflowTemplateSelectorDialog', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('show', () => {
    it('defaults to "all" category for non-new users', () => {
      mockNewUserService.isNewUser.mockReturnValue(false)

      const dialog = useWorkflowTemplateSelectorDialog()
      dialog.show()

      expect(mockDialogService.showLayoutDialog).toHaveBeenCalledWith(
        expect.objectContaining({
          props: expect.objectContaining({
            initialCategory: 'all'
          })
        })
      )
    })

    it('defaults to "basics-getting-started" category for new users', () => {
      mockNewUserService.isNewUser.mockReturnValue(true)

      const dialog = useWorkflowTemplateSelectorDialog()
      dialog.show()

      expect(mockDialogService.showLayoutDialog).toHaveBeenCalledWith(
        expect.objectContaining({
          props: expect.objectContaining({
            initialCategory: 'basics-getting-started'
          })
        })
      )
    })

    it('defaults to "all" when new user status is undetermined', () => {
      mockNewUserService.isNewUser.mockReturnValue(null)

      const dialog = useWorkflowTemplateSelectorDialog()
      dialog.show()

      expect(mockDialogService.showLayoutDialog).toHaveBeenCalledWith(
        expect.objectContaining({
          props: expect.objectContaining({
            initialCategory: 'all'
          })
        })
      )
    })

    it('uses explicit initialCategory when provided', () => {
      mockNewUserService.isNewUser.mockReturnValue(true)

      const dialog = useWorkflowTemplateSelectorDialog()
      dialog.show('command', { initialCategory: 'custom-category' })

      expect(mockDialogService.showLayoutDialog).toHaveBeenCalledWith(
        expect.objectContaining({
          props: expect.objectContaining({
            initialCategory: 'custom-category'
          })
        })
      )
    })

    it('tracks telemetry with source', () => {
      mockNewUserService.isNewUser.mockReturnValue(false)

      const dialog = useWorkflowTemplateSelectorDialog()
      dialog.show('sidebar')

      expect(mockTelemetry.trackTemplateLibraryOpened).toHaveBeenCalledWith({
        source: 'sidebar'
      })
    })
  })

  describe('hide', () => {
    it('closes the dialog', () => {
      const dialog = useWorkflowTemplateSelectorDialog()
      dialog.hide()

      expect(mockDialogStore.closeDialog).toHaveBeenCalledWith({
        key: 'global-workflow-template-selector'
      })
    })
  })
})
