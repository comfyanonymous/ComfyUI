<template>
  <div
    class="flex h-12 items-center gap-2 border-b border-interface-stroke px-2"
  >
    <div class="min-w-0 flex-1 px-2 text-[14px] font-normal text-text-primary">
      <span>{{ headerTitle }}</span>
    </div>
    <div
      class="inline-flex h-6 items-center gap-2 text-[12px] leading-none text-text-primary"
    >
      <span :class="{ 'opacity-50': queuedCount === 0 }">{{
        t('sideToolbar.queueProgressOverlay.clearQueueTooltip')
      }}</span>
      <Button
        v-tooltip.top="clearAllJobsTooltip"
        variant="destructive"
        size="icon"
        :aria-label="t('sideToolbar.queueProgressOverlay.clearQueued')"
        :disabled="queuedCount === 0"
        @click="$emit('clearQueued')"
      >
        <i class="icon-[lucide--list-x] size-4" />
      </Button>
    </div>
    <JobHistoryActionsMenu @clear-history="$emit('clearHistory')" />
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'

import JobHistoryActionsMenu from '@/components/queue/JobHistoryActionsMenu.vue'
import Button from '@/components/ui/button/Button.vue'
import { buildTooltipConfig } from '@/composables/useTooltipConfig'

defineProps<{
  headerTitle: string
  queuedCount: number
}>()

defineEmits<{
  (e: 'clearHistory'): void
  (e: 'clearQueued'): void
}>()

const { t } = useI18n()
const clearAllJobsTooltip = computed(() =>
  buildTooltipConfig(t('sideToolbar.queueProgressOverlay.clearAllJobsTooltip'))
)
</script>
