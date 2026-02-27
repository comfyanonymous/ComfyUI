import WorkflowTemplateSelectorDialog from '@/components/custom/widget/WorkflowTemplateSelectorDialog.vue'
import { useTelemetry } from '@/platform/telemetry'
import { useDialogService } from '@/services/dialogService'
import { useNewUserService } from '@/services/useNewUserService'
import { useDialogStore } from '@/stores/dialogStore'

const DIALOG_KEY = 'global-workflow-template-selector'
const GETTING_STARTED_CATEGORY_ID = 'basics-getting-started'

export const useWorkflowTemplateSelectorDialog = () => {
  const dialogService = useDialogService()
  const dialogStore = useDialogStore()
  const newUserService = useNewUserService()

  function hide() {
    dialogStore.closeDialog({ key: DIALOG_KEY })
  }

  function show(
    source: 'sidebar' | 'menu' | 'command' = 'command',
    options?: { initialCategory?: string }
  ) {
    useTelemetry()?.trackTemplateLibraryOpened({ source })

    const initialCategory =
      options?.initialCategory ??
      (newUserService.isNewUser() ? GETTING_STARTED_CATEGORY_ID : 'all')

    dialogService.showLayoutDialog({
      key: DIALOG_KEY,
      component: WorkflowTemplateSelectorDialog,
      props: {
        onClose: hide,
        initialCategory
      }
    })
  }

  return {
    show,
    hide
  }
}
