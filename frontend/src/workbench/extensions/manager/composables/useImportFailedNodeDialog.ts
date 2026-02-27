import { useDialogService } from '@/services/dialogService'
import type { DialogComponentProps } from '@/stores/dialogStore'
import { useDialogStore } from '@/stores/dialogStore'
import ImportFailedNodeContent from '@/workbench/extensions/manager/components/manager/ImportFailedNodeContent.vue'
import ImportFailedNodeFooter from '@/workbench/extensions/manager/components/manager/ImportFailedNodeFooter.vue'
import ImportFailedNodeHeader from '@/workbench/extensions/manager/components/manager/ImportFailedNodeHeader.vue'
import type { ConflictDetectionResult } from '@/workbench/extensions/manager/types/conflictDetectionTypes'

const DIALOG_KEY = 'global-import-failed'

export function useImportFailedNodeDialog() {
  const { showSmallLayoutDialog } = useDialogService()
  const dialogStore = useDialogStore()

  function hide() {
    dialogStore.closeDialog({ key: DIALOG_KEY })
  }

  function show(
    options: {
      conflictedPackages?: ConflictDetectionResult[]
      dialogComponentProps?: Omit<DialogComponentProps, 'pt'>
    } = {}
  ) {
    const { dialogComponentProps, conflictedPackages = [] } = options

    showSmallLayoutDialog({
      key: DIALOG_KEY,
      headerComponent: ImportFailedNodeHeader,
      footerComponent: ImportFailedNodeFooter,
      component: ImportFailedNodeContent,
      dialogComponentProps,
      props: {
        conflictedPackages
      },
      footerProps: {
        conflictedPackages
      }
    })
  }

  return { show, hide }
}
