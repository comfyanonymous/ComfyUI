import type { ComponentAttrs } from 'vue-component-type-helpers'

import MissingModelsContent from '@/components/dialog/content/MissingModelsContent.vue'
import MissingModelsFooter from '@/components/dialog/content/MissingModelsFooter.vue'
import MissingModelsHeader from '@/components/dialog/content/MissingModelsHeader.vue'
import { useDialogService } from '@/services/dialogService'
import { useDialogStore } from '@/stores/dialogStore'

const DIALOG_KEY = 'global-missing-models-warning'

export function useMissingModelsDialog() {
  const { showSmallLayoutDialog } = useDialogService()
  const dialogStore = useDialogStore()

  function hide() {
    dialogStore.closeDialog({ key: DIALOG_KEY })
  }

  function show(props: ComponentAttrs<typeof MissingModelsContent>) {
    showSmallLayoutDialog({
      key: DIALOG_KEY,
      headerComponent: MissingModelsHeader,
      footerComponent: MissingModelsFooter,
      component: MissingModelsContent,
      props,
      footerProps: props
    })
  }

  return { show, hide }
}
