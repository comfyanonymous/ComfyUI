// import { useSelectedLiteGraphItems } from '@/composables/canvas/useSelectedLiteGraphItems' // Unused for now
import { t } from '@/i18n'
import { LGraphNode } from '@/lib/litegraph/src/litegraph'
import { useToastStore } from '@/platform/updates/common/toastStore'
import { useWorkflowStore } from '@/platform/workflow/management/stores/workflowStore'
import {
  useCanvasStore,
  useTitleEditorStore
} from '@/renderer/core/canvas/canvasStore'
import { app } from '@/scripts/app'
import { useDialogService } from '@/services/dialogService'

/**
 * Composable for handling basic selection operations like copy, paste, duplicate, delete, rename
 */
export function useSelectionOperations() {
  // const { getSelectedNodes } = useSelectedLiteGraphItems() // Unused for now
  const canvasStore = useCanvasStore()
  const toastStore = useToastStore()
  const dialogService = useDialogService()
  const titleEditorStore = useTitleEditorStore()
  const workflowStore = useWorkflowStore()

  const copySelection = () => {
    const canvas = app.canvas
    if (!canvas.selectedItems || canvas.selectedItems.size === 0) {
      toastStore.add({
        severity: 'warn',
        summary: t('g.nothingToCopy'),
        detail: t('g.selectItemsToCopy'),
        life: 3000
      })
      return
    }

    canvas.copyToClipboard()
    toastStore.add({
      severity: 'success',
      summary: t('g.copied'),
      detail: t('g.itemsCopiedToClipboard'),
      life: 2000
    })
  }

  const pasteSelection = () => {
    const canvas = app.canvas
    canvas.pasteFromClipboard({ connectInputs: false })

    // Trigger change tracking
    workflowStore.activeWorkflow?.changeTracker?.checkState()
  }

  const duplicateSelection = () => {
    const canvas = app.canvas
    if (!canvas.selectedItems || canvas.selectedItems.size === 0) {
      toastStore.add({
        severity: 'warn',
        summary: t('g.nothingToDuplicate'),
        detail: t('g.selectItemsToDuplicate'),
        life: 3000
      })
      return
    }

    // Copy current selection
    canvas.copyToClipboard()

    // Clear selection to avoid confusion
    canvas.selectedItems.clear()
    canvasStore.updateSelectedItems()

    // Paste to create duplicates
    canvas.pasteFromClipboard({ connectInputs: false })

    // Trigger change tracking
    workflowStore.activeWorkflow?.changeTracker?.checkState()
  }

  const deleteSelection = () => {
    const canvas = app.canvas
    if (!canvas.selectedItems || canvas.selectedItems.size === 0) {
      toastStore.add({
        severity: 'warn',
        summary: t('g.nothingToDelete'),
        detail: t('g.selectItemsToDelete'),
        life: 3000
      })
      return
    }

    canvas.deleteSelected()
    canvas.setDirty(true, true)

    // Trigger change tracking
    workflowStore.activeWorkflow?.changeTracker?.checkState()
  }

  const renameSelection = async () => {
    const selectedItems = Array.from(canvasStore.selectedItems)

    // Handle single node selection
    if (selectedItems.length === 1) {
      const item = selectedItems[0]

      // For nodes, use the title editor
      if (item instanceof LGraphNode) {
        titleEditorStore.titleEditorTarget = item
        return
      }

      // For other items like groups, use prompt dialog
      const currentTitle = 'title' in item ? (item.title as string) : ''
      const newTitle = await dialogService.prompt({
        title: t('g.rename'),
        message: t('g.enterNewName'),
        defaultValue: currentTitle
      })

      if (newTitle && newTitle !== currentTitle) {
        if ('title' in item) {
          // Type-safe assignment for items with title property
          const titledItem = item as { title: string }
          titledItem.title = newTitle
          app.canvas.setDirty(true, true)
          workflowStore.activeWorkflow?.changeTracker?.checkState()
        }
      }
      return
    }

    // Handle multiple selections - batch rename
    if (selectedItems.length > 1) {
      const baseTitle = await dialogService.prompt({
        title: t('g.batchRename'),
        message: t('g.enterBaseName'),
        defaultValue: 'Item'
      })

      if (baseTitle) {
        selectedItems.forEach((item, index) => {
          if ('title' in item) {
            // Type-safe assignment for items with title property
            const titledItem = item as { title: string }
            titledItem.title = `${baseTitle} ${index + 1}`
          }
        })
        app.canvas.setDirty(true, true)
        workflowStore.activeWorkflow?.changeTracker?.checkState()
      }
      return
    }

    toastStore.add({
      severity: 'warn',
      summary: t('g.nothingToRename'),
      detail: t('g.selectItemsToRename'),
      life: 3000
    })
  }

  return {
    copySelection,
    pasteSelection,
    duplicateSelection,
    deleteSelection,
    renameSelection
  }
}
