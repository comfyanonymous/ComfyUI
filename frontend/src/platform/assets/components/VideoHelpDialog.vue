<template>
  <Dialog
    v-model:visible="isVisible"
    modal
    :closable="false"
    :close-on-escape="false"
    :dismissable-mask="true"
    :pt="{
      root: { class: 'video-help-dialog' },
      header: { class: '!hidden' },
      content: { class: '!p-0' },
      mask: { class: '!bg-black/70' }
    }"
    :style="{ width: '90vw' }"
  >
    <div class="relative">
      <Button
        variant="textonly"
        size="icon"
        class="absolute top-4 right-6 z-10"
        :aria-label="$t('g.close')"
        @click="isVisible = false"
      >
        <i class="pi pi-times text-sm" />
      </Button>
      <video
        autoplay
        muted
        loop
        :aria-label="ariaLabel"
        class="w-full rounded-lg"
        :src="videoUrl"
      >
        {{ $t('g.videoFailedToLoad') }}
      </video>
    </div>
  </Dialog>
</template>

<script setup lang="ts">
import { useEventListener } from '@vueuse/core'
import Dialog from 'primevue/dialog'
import { onWatcherCleanup, watch } from 'vue'

import Button from '@/components/ui/button/Button.vue'

const isVisible = defineModel<boolean>({ required: true })

const { videoUrl, ariaLabel = 'Help video' } = defineProps<{
  videoUrl: string
  ariaLabel?: string
}>()

const handleEscapeKey = (event: KeyboardEvent) => {
  if (event.key === 'Escape') {
    event.stopImmediatePropagation()
    event.stopPropagation()
    event.preventDefault()
    isVisible.value = false
  }
}

// Add listener with capture phase to intercept before parent dialogs
// Only active when dialog is visible
watch(
  isVisible,
  (visible) => {
    if (visible) {
      const stop = useEventListener(document, 'keydown', handleEscapeKey, {
        capture: true
      })
      onWatcherCleanup(stop)
    }
  },
  { immediate: true }
)
</script>
