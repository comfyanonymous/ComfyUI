import { useDialogService } from '@/services/dialogService'
import type { DialogComponentProps } from '@/stores/dialogStore'
import { useDialogStore } from '@/stores/dialogStore'
import NodeConflictDialogContent from '@/workbench/extensions/manager/components/manager/NodeConflictDialogContent.vue'
import NodeConflictFooter from '@/workbench/extensions/manager/components/manager/NodeConflictFooter.vue'
import NodeConflictHeader from '@/workbench/extensions/manager/components/manager/NodeConflictHeader.vue'
import type { ConflictDetectionResult } from '@/workbench/extensions/manager/types/conflictDetectionTypes'

const DIALOG_KEY = 'global-node-conflict'

export function useNodeConflictDialog() {
  const { showSmallLayoutDialog } = useDialogService()
  const dialogStore = useDialogStore()

  function hide() {
    dialogStore.closeDialog({ key: DIALOG_KEY })
  }

  function show(
    options: {
      showAfterWhatsNew?: boolean
      conflictedPackages?: ConflictDetectionResult[]
      dialogComponentProps?: Omit<DialogComponentProps, 'pt'>
      buttonText?: string
      onButtonClick?: () => void
    } = {}
  ) {
    const {
      dialogComponentProps,
      buttonText,
      onButtonClick,
      showAfterWhatsNew,
      conflictedPackages
    } = options

    showSmallLayoutDialog({
      key: DIALOG_KEY,
      headerComponent: NodeConflictHeader,
      footerComponent: NodeConflictFooter,
      component: NodeConflictDialogContent,
      dialogComponentProps,
      props: {
        showAfterWhatsNew,
        conflictedPackages
      },
      footerProps: {
        buttonText,
        onButtonClick
      }
    })
  }

  return { show, hide }
}
