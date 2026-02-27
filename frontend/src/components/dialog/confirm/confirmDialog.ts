import ConfirmBody from '@/components/dialog/confirm/ConfirmBody.vue'
import ConfirmFooter from '@/components/dialog/confirm/ConfirmFooter.vue'
import ConfirmHeader from '@/components/dialog/confirm/ConfirmHeader.vue'
import { useDialogStore } from '@/stores/dialogStore'
import type { ComponentAttrs } from 'vue-component-type-helpers'

interface ConfirmDialogOptions {
  headerProps?: ComponentAttrs<typeof ConfirmHeader>
  props?: ComponentAttrs<typeof ConfirmBody>
  footerProps?: ComponentAttrs<typeof ConfirmFooter>
}

export function showConfirmDialog(options: ConfirmDialogOptions = {}) {
  const dialogStore = useDialogStore()
  const { headerProps, props, footerProps } = options
  return dialogStore.showDialog({
    headerComponent: ConfirmHeader,
    component: ConfirmBody,
    footerComponent: ConfirmFooter,
    headerProps,
    props,
    footerProps,
    dialogComponentProps: {
      pt: {
        header: 'py-0! px-0!',
        content: 'p-0!',
        footer: 'p-0!'
      }
    }
  })
}
