import { computed, ref, toValue } from 'vue'
import type { MaybeRefOrGetter } from 'vue'

import { SUPPORTED_EXTENSIONS } from '@/extensions/core/load3d/constants'
import { t } from '@/i18n'
import { useToastStore } from '@/platform/updates/common/toastStore'

interface UseLoad3dDragOptions {
  onModelDrop: (file: File) => void | Promise<void>
  disabled?: MaybeRefOrGetter<boolean>
}

export function useLoad3dDrag(options: UseLoad3dDragOptions) {
  const isDragging = ref(false)
  const dragMessage = ref('')

  const isDisabled = computed(() => toValue(options.disabled) ?? false)

  function isValidModelFile(file: File): boolean {
    const fileName = file.name.toLowerCase()
    const extension = fileName.substring(fileName.lastIndexOf('.'))
    return SUPPORTED_EXTENSIONS.has(extension)
  }

  function handleDragOver(event: DragEvent) {
    if (isDisabled.value) return

    if (!event.dataTransfer) return

    const hasFiles = event.dataTransfer.types.includes('Files')

    if (!hasFiles) return

    isDragging.value = true
    event.dataTransfer.dropEffect = 'copy'
    dragMessage.value = t('load3d.dropToLoad')
  }

  function handleDragLeave() {
    isDragging.value = false
  }

  async function handleDrop(event: DragEvent) {
    isDragging.value = false

    if (isDisabled.value) return

    if (!event.dataTransfer) return

    const files = Array.from(event.dataTransfer.files)

    if (files.length === 0) return

    const modelFile = files.find(isValidModelFile)

    if (modelFile) {
      await options.onModelDrop(modelFile)
    } else {
      useToastStore().addAlert(t('load3d.unsupportedFileType'))
    }
  }

  return {
    isDragging,
    dragMessage,
    handleDragOver,
    handleDragLeave,
    handleDrop
  }
}
