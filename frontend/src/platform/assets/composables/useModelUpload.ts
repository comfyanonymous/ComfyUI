import { computed } from 'vue'

import { useFeatureFlags } from '@/composables/useFeatureFlags'
import UploadModelDialog from '@/platform/assets/components/UploadModelDialog.vue'
import UploadModelDialogHeader from '@/platform/assets/components/UploadModelDialogHeader.vue'
import UploadModelUpgradeModal from '@/platform/assets/components/UploadModelUpgradeModal.vue'
import UploadModelUpgradeModalHeader from '@/platform/assets/components/UploadModelUpgradeModalHeader.vue'
import { useDialogStore } from '@/stores/dialogStore'

export function useModelUpload(
  onUploadSuccess?: () => Promise<unknown> | void
) {
  const dialogStore = useDialogStore()
  const { flags } = useFeatureFlags()
  const isUploadButtonEnabled = computed(() => flags.modelUploadButtonEnabled)

  function showUploadDialog() {
    if (!flags.privateModelsEnabled) {
      dialogStore.showDialog({
        key: 'upload-model-upgrade',
        headerComponent: UploadModelUpgradeModalHeader,
        component: UploadModelUpgradeModal,
        dialogComponentProps: {
          pt: {
            header: 'py-0! pl-0!',
            content: 'p-0! overflow-y-hidden!'
          }
        }
      })
    } else {
      dialogStore.showDialog({
        key: 'upload-model',
        headerComponent: UploadModelDialogHeader,
        component: UploadModelDialog,
        props: {
          onUploadSuccess: async () => {
            await onUploadSuccess?.()
          }
        },
        dialogComponentProps: {
          pt: {
            header: 'py-0! pl-0!',
            content: 'p-0! overflow-y-hidden!'
          }
        }
      })
    }
  }
  return { isUploadButtonEnabled, showUploadDialog }
}
