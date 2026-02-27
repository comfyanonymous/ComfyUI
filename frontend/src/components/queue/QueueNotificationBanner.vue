<template>
  <div class="inline-flex overflow-hidden rounded-lg bg-secondary-background">
    <div class="flex items-center gap-2 p-1 pr-3">
      <div
        :class="
          cn(
            'relative shrink-0 items-center rounded-[4px]',
            showsCompletionPreview && showThumbnails
              ? 'flex h-8 overflow-visible p-0'
              : showsCompletionPreview
                ? 'flex size-8 justify-center overflow-hidden p-0'
                : 'flex size-8 justify-center p-1'
          )
        "
      >
        <template v-if="showThumbnails">
          <div class="flex h-8 shrink-0 items-center">
            <div
              v-for="(thumbnailUrl, index) in thumbnailUrls"
              :key="`completion-preview-${index}`"
              :class="
                cn(
                  'relative size-8 shrink-0 overflow-hidden rounded-[4px]',
                  index > 0 && '-ml-3 ring-2 ring-secondary-background'
                )
              "
            >
              <img
                :src="thumbnailUrl"
                :alt="t('sideToolbar.queueProgressOverlay.preview')"
                class="size-full object-cover"
              />
            </div>
          </div>
        </template>
        <div
          v-else-if="showCompletionGradientFallback"
          class="size-full bg-linear-to-br from-coral-500 via-coral-500 to-azure-600"
        />
        <i
          v-else
          :class="cn(iconClass, 'size-4', iconColorClass)"
          aria-hidden="true"
        />
      </div>
      <div class="flex h-full items-center">
        <span
          class="overflow-hidden text-ellipsis text-center font-inter text-[12px] leading-normal font-normal text-base-foreground"
        >
          {{ bannerText }}
        </span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'

import type { QueueNotificationBanner } from '@/composables/queue/useQueueNotificationBanners'
import { cn } from '@/utils/tailwindUtil'

const { notification } = defineProps<{
  notification: QueueNotificationBanner
}>()

const { t, n } = useI18n()

const thumbnailUrls = computed(() => {
  if (notification.type !== 'completed') {
    return []
  }
  return notification.thumbnailUrls?.slice(0, 2) ?? []
})

const showThumbnails = computed(() => {
  if (notification.type !== 'completed') {
    return false
  }
  return thumbnailUrls.value.length > 0
})

const showCompletionGradientFallback = computed(
  () => notification.type === 'completed' && !showThumbnails.value
)

const showsCompletionPreview = computed(
  () => showThumbnails.value || showCompletionGradientFallback.value
)

const bannerText = computed(() => {
  const count = notification.count
  if (notification.type === 'queuedPending') {
    return t('queue.jobQueueing')
  }
  if (notification.type === 'queued') {
    if (count === 1) {
      return t('queue.jobAddedToQueue')
    }
    return t(
      'sideToolbar.queueProgressOverlay.jobsAddedToQueue',
      { count: n(count) },
      count
    )
  }
  if (notification.type === 'failed') {
    if (count === 1) {
      return t('sideToolbar.queueProgressOverlay.jobFailed')
    }
    return t(
      'sideToolbar.queueProgressOverlay.jobsFailed',
      { count: n(count) },
      count
    )
  }
  if (count === 1) {
    return t('sideToolbar.queueProgressOverlay.jobCompleted')
  }
  return t(
    'sideToolbar.queueProgressOverlay.jobsCompleted',
    { count: n(count) },
    count
  )
})

const iconClass = computed(() => {
  if (notification.type === 'queuedPending') {
    return 'icon-[lucide--loader-circle]'
  }
  if (notification.type === 'queued') {
    return 'icon-[lucide--check]'
  }
  if (notification.type === 'failed') {
    return 'icon-[lucide--circle-alert]'
  }
  return 'icon-[lucide--image]'
})

const iconColorClass = computed(() => {
  if (notification.type === 'queuedPending') {
    return 'animate-spin text-slate-100'
  }
  if (notification.type === 'failed') {
    return 'text-danger-200'
  }
  return 'text-slate-100'
})
</script>
