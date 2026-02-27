import { useFavicon } from '@vueuse/core'
import { watch } from 'vue'

import { useExecutionStore } from '@/stores/executionStore'

export const useProgressFavicon = () => {
  const defaultFavicon = '/assets/images/favicon_progress_16x16/frame_9.png'
  const favicon = useFavicon(defaultFavicon)
  const executionStore = useExecutionStore()
  const totalFrames = 10

  watch(
    [() => executionStore.executionProgress, () => executionStore.isIdle],
    ([progress, isIdle]) => {
      if (isIdle) {
        favicon.value = defaultFavicon
      } else {
        const frame = Math.min(
          Math.max(0, Math.floor(progress * totalFrames)),
          totalFrames - 1
        )
        favicon.value = `/assets/images/favicon_progress_16x16/frame_${frame}.png`
      }
    }
  )
}
