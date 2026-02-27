import type { LGraphNode } from '@/lib/litegraph/src/LGraphNode'
import { useDialogStore } from '@/stores/dialogStore'
import TopBarHeader from '@/components/maskeditor/dialog/TopBarHeader.vue'
import MaskEditorContent from '@/components/maskeditor/MaskEditorContent.vue'

export function useMaskEditor() {
  const openMaskEditor = (node: LGraphNode) => {
    if (!node) {
      console.error('[MaskEditor] No node provided')
      return
    }

    if (!node.imgs?.length && node.previewMediaType !== 'image') {
      console.error('[MaskEditor] Node has no images')
      return
    }

    useDialogStore().showDialog({
      key: 'global-mask-editor',
      headerComponent: TopBarHeader,
      component: MaskEditorContent,
      props: {
        node
      },
      dialogComponentProps: {
        style: 'width: 90vw; height: 90vh;',
        modal: true,
        maximizable: true,
        closable: true,
        pt: {
          root: {
            class: 'mask-editor-dialog flex flex-col'
          },
          content: {
            class: 'flex flex-col min-h-0 flex-1 !p-0'
          },
          header: {
            class: '!p-2'
          }
        }
      }
    })
  }

  return {
    openMaskEditor
  }
}
