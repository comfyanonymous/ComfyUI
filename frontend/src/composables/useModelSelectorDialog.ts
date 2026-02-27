import SampleModelSelector from '@/components/widget/SampleModelSelector.vue'
import { useDialogService } from '@/services/dialogService'
import { useDialogStore } from '@/stores/dialogStore'

const DIALOG_KEY = 'global-model-selector'

export const useModelSelectorDialog = () => {
  const dialogService = useDialogService()
  const dialogStore = useDialogStore()

  function hide() {
    dialogStore.closeDialog({ key: DIALOG_KEY })
  }

  function show() {
    dialogService.showLayoutDialog({
      key: DIALOG_KEY,
      component: SampleModelSelector,
      props: {
        onClose: hide
      }
    })
  }

  return {
    show,
    hide
  }
}
