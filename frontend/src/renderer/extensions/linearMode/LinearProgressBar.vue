<script setup lang="ts">
import { useQueueProgress } from '@/composables/queue/useQueueProgress'
import { useQueueStore } from '@/stores/queueStore'
import { cn } from '@/utils/tailwindUtil'

defineOptions({ inheritAttrs: false })

const {
  class: className,
  overallOpacity = 1,
  activeOpacity = 1
} = defineProps<{
  class?: string
  overallOpacity?: number
  activeOpacity?: number
}>()

const { totalPercent, currentNodePercent } = useQueueProgress()
const queueStore = useQueueStore()
</script>
<template>
  <div
    :class="
      cn(
        'relative h-2 bg-secondary-background transition-opacity',
        queueStore.runningTasks.length === 0 && 'opacity-0',
        className
      )
    "
  >
    <div
      class="absolute inset-0 h-full bg-interface-panel-job-progress-primary transition-[width]"
      :style="{ width: `${totalPercent}%`, opacity: overallOpacity }"
    />
    <div
      class="absolute inset-0 h-full bg-interface-panel-job-progress-secondary transition-[width]"
      :style="{ width: `${currentNodePercent}%`, opacity: activeOpacity }"
    />
  </div>
</template>
