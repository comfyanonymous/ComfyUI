<template>
  <div class="flex flex-col gap-3 p-2">
    <div class="flex flex-col gap-1">
      <div
        class="relative h-2 w-full overflow-hidden rounded-full border border-interface-stroke bg-interface-panel-surface"
      >
        <div
          class="absolute inset-0 h-full rounded-full transition-[width]"
          :style="totalProgressStyle"
        />
        <div
          class="absolute inset-0 h-full rounded-full transition-[width]"
          :style="currentNodeProgressStyle"
        />
      </div>
      <div class="flex items-start justify-end gap-4 text-[12px] leading-none">
        <div class="flex items-center gap-1 text-text-primary opacity-90">
          <i18n-t keypath="sideToolbar.queueProgressOverlay.total">
            <template #percent>
              <span class="font-bold">{{ totalPercentFormatted }}</span>
            </template>
          </i18n-t>
        </div>
        <div class="flex items-center gap-1 text-text-secondary">
          <span>{{ t('sideToolbar.queueProgressOverlay.currentNode') }}</span>
          <span class="inline-block max-w-[10rem] truncate">{{
            currentNodeName
          }}</span>
          <span class="flex items-center gap-1">
            <span>{{ currentNodePercentFormatted }}</span>
          </span>
        </div>
      </div>
    </div>

    <div :class="bottomRowClass">
      <div class="flex items-center gap-4 text-[12px] text-text-primary">
        <div class="flex items-center gap-2">
          <span class="opacity-90">
            <span class="font-bold">{{ runningCount }}</span>
            <span class="ml-1">{{
              t('sideToolbar.queueProgressOverlay.running')
            }}</span>
          </span>
          <Button
            v-if="runningCount > 0"
            v-tooltip.top="cancelJobTooltip"
            variant="destructive"
            size="icon"
            :aria-label="t('sideToolbar.queueProgressOverlay.interruptAll')"
            @click="$emit('interruptAll')"
          >
            <i
              class="icon-[lucide--x] block size-4 leading-none text-text-primary"
            />
          </Button>
        </div>

        <div class="flex items-center gap-2">
          <span class="opacity-90">
            <span class="font-bold">{{ queuedCount }}</span>
            <span class="ml-1">{{
              t('sideToolbar.queueProgressOverlay.queuedSuffix')
            }}</span>
          </span>
          <Button
            v-if="queuedCount > 0"
            v-tooltip.top="clearQueueTooltip"
            variant="destructive"
            size="icon"
            :aria-label="t('sideToolbar.queueProgressOverlay.clearQueued')"
            @click="$emit('clearQueued')"
          >
            <i
              class="icon-[lucide--list-x] block size-4 leading-none text-text-primary"
            />
          </Button>
        </div>
      </div>

      <Button
        class="min-w-30 flex-1 px-2 py-0"
        variant="secondary"
        size="md"
        @click="$emit('viewAllJobs')"
      >
        {{ t('sideToolbar.queueProgressOverlay.viewAllJobs') }}
      </Button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'

import Button from '@/components/ui/button/Button.vue'
import { buildTooltipConfig } from '@/composables/useTooltipConfig'

defineProps<{
  totalProgressStyle: Record<string, string>
  currentNodeProgressStyle: Record<string, string>
  totalPercentFormatted: string
  currentNodePercentFormatted: string
  currentNodeName: string
  runningCount: number
  queuedCount: number
  bottomRowClass: string
}>()

defineEmits<{
  (e: 'interruptAll'): void
  (e: 'clearQueued'): void
  (e: 'viewAllJobs'): void
}>()

const { t } = useI18n()
const cancelJobTooltip = computed(() =>
  buildTooltipConfig(t('sideToolbar.queueProgressOverlay.cancelJobTooltip'))
)
const clearQueueTooltip = computed(() =>
  buildTooltipConfig(t('sideToolbar.queueProgressOverlay.clearQueueTooltip'))
)
</script>
