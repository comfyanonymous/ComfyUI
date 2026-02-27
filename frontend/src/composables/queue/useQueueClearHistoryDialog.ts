import QueueClearHistoryDialog from '@/components/queue/dialogs/QueueClearHistoryDialog.vue'
import { useDialogStore } from '@/stores/dialogStore'

export const useQueueClearHistoryDialog = () => {
  const dialogStore = useDialogStore()

  const showQueueClearHistoryDialog = () => {
    dialogStore.showDialog({
      key: 'queue-clear-history',
      component: QueueClearHistoryDialog,
      dialogComponentProps: {
        headless: true,
        closable: false,
        closeOnEscape: true,
        dismissableMask: true,
        pt: {
          root: {
            class: 'max-w-90 w-auto bg-transparent border-none shadow-none'
          },
          content: {
            class: 'bg-transparent',
            style: 'padding: 0'
          }
        }
      }
    })
  }

  return {
    showQueueClearHistoryDialog
  }
}
