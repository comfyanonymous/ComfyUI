<template>
  <div
    v-if="currentNotification"
    class="flex justify-end"
    role="status"
    aria-live="polite"
    aria-atomic="true"
  >
    <QueueNotificationBanner :notification="currentNotification" />
  </div>
</template>

<script setup lang="ts">
import QueueNotificationBanner from '@/components/queue/QueueNotificationBanner.vue'
import { useQueueNotificationBanners } from '@/composables/queue/useQueueNotificationBanners'

const { currentNotification } = useQueueNotificationBanners()
</script>
