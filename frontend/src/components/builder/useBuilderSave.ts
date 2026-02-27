import { ref } from 'vue'

import { useErrorHandling } from '@/composables/useErrorHandling'
import { useWorkflowService } from '@/platform/workflow/core/services/workflowService'
import { useWorkflowStore } from '@/platform/workflow/management/stores/workflowStore'
import { useDialogService } from '@/services/dialogService'
import { useAppMode } from '@/composables/useAppMode'
import { useAppModeStore } from '@/stores/appModeStore'
import { useDialogStore } from '@/stores/dialogStore'

import BuilderSaveDialogContent from './BuilderSaveDialogContent.vue'
import BuilderSaveSuccessDialogContent from './BuilderSaveSuccessDialogContent.vue'
import { whenever } from '@vueuse/core'

const SAVE_DIALOG_KEY = 'builder-save'
const SUCCESS_DIALOG_KEY = 'builder-save-success'

export function useBuilderSave() {
  const { setMode } = useAppMode()
  const { toastErrorHandler } = useErrorHandling()
  const workflowStore = useWorkflowStore()
  const workflowService = useWorkflowService()
  const dialogService = useDialogService()
  const appModeStore = useAppModeStore()
  const dialogStore = useDialogStore()

  const saving = ref(false)

  whenever(saving, onBuilderSave)

  function setSaving(value: boolean) {
    saving.value = value
  }

  async function onBuilderSave() {
    const workflow = workflowStore.activeWorkflow
    if (!workflow) {
      resetSaving()
      return
    }

    if (!workflow.isTemporary && workflow.initialMode != null) {
      // Re-save with the previously chosen mode — no dialog needed.
      try {
        appModeStore.flushSelections()
        await workflowService.saveWorkflow(workflow)
        showSuccessDialog(workflow.filename, workflow.initialMode === 'app')
      } catch (e) {
        toastErrorHandler(e)
        resetSaving()
      }
      return
    }

    showSaveDialog(workflow.filename)
  }

  function showSaveDialog(defaultFilename: string) {
    dialogService.showLayoutDialog({
      key: SAVE_DIALOG_KEY,
      component: BuilderSaveDialogContent,
      props: {
        defaultFilename,
        onSave: handleSave,
        onClose: handleCancelSave
      },
      dialogComponentProps: {
        onClose: resetSaving
      }
    })
  }

  function handleCancelSave() {
    closeSaveDialog()
    resetSaving()
  }

  async function handleSave(filename: string, openAsApp: boolean) {
    try {
      const workflow = workflowStore.activeWorkflow
      if (!workflow) return

      appModeStore.flushSelections()
      const mode = openAsApp ? 'app' : 'graph'
      const saved = await workflowService.saveWorkflowAs(workflow, {
        filename,
        initialMode: mode
      })

      if (!saved) return

      closeSaveDialog()
      showSuccessDialog(filename, openAsApp)
    } catch (e) {
      toastErrorHandler(e)
      closeSaveDialog()
      resetSaving()
    }
  }

  function showSuccessDialog(workflowName: string, savedAsApp: boolean) {
    dialogService.showLayoutDialog({
      key: SUCCESS_DIALOG_KEY,
      component: BuilderSaveSuccessDialogContent,
      props: {
        workflowName,
        savedAsApp,
        onViewApp: () => {
          setMode('app')
          closeSuccessDialog()
        },
        onClose: closeSuccessDialog
      },
      dialogComponentProps: {
        onClose: resetSaving
      }
    })
  }

  function closeSaveDialog() {
    dialogStore.closeDialog({ key: SAVE_DIALOG_KEY })
  }

  function closeSuccessDialog() {
    dialogStore.closeDialog({ key: SUCCESS_DIALOG_KEY })
    resetSaving()
  }

  function resetSaving() {
    saving.value = false
  }

  return { saving, setSaving }
}
