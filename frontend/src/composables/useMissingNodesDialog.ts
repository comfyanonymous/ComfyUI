import type { ComponentAttrs } from 'vue-component-type-helpers'

import MissingNodesContent from '@/components/dialog/content/MissingNodesContent.vue'
import MissingNodesFooter from '@/components/dialog/content/MissingNodesFooter.vue'
import MissingNodesHeader from '@/components/dialog/content/MissingNodesHeader.vue'
import { useDialogService } from '@/services/dialogService'
import { useDialogStore } from '@/stores/dialogStore'

const DIALOG_KEY = 'global-missing-nodes'

export function useMissingNodesDialog() {
  const { showSmallLayoutDialog } = useDialogService()
  const dialogStore = useDialogStore()

  function hide() {
    dialogStore.closeDialog({ key: DIALOG_KEY })
  }

  function show(props: ComponentAttrs<typeof MissingNodesContent>) {
    showSmallLayoutDialog({
      key: DIALOG_KEY,
      headerComponent: MissingNodesHeader,
      footerComponent: MissingNodesFooter,
      component: MissingNodesContent,
      props
    })
  }

  return { show, hide }
}
