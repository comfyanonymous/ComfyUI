<template>
  <div class="relative rounded-lg bg-backdrop/30">
    <div class="flex flex-col gap-2">
      <Button
        v-tooltip.right="{
          value: isRecording
            ? $t('load3d.stopRecording')
            : $t('load3d.startRecording'),
          showDelay: 300
        }"
        size="icon"
        variant="textonly"
        :class="
          cn(
            'rounded-full',
            isRecording && 'text-red-500 recording-button-blink'
          )
        "
        :aria-label="
          isRecording ? $t('load3d.stopRecording') : $t('load3d.startRecording')
        "
        @click="toggleRecording"
      >
        <i
          :class="[
            'pi',
            isRecording ? 'pi-circle-fill' : 'pi-video',
            'text-lg text-base-foreground'
          ]"
        />
      </Button>

      <Button
        v-if="hasRecording && !isRecording"
        v-tooltip.right="{
          value: $t('load3d.exportRecording'),
          showDelay: 300
        }"
        size="icon"
        variant="textonly"
        class="rounded-full"
        :aria-label="$t('load3d.exportRecording')"
        @click="handleExportRecording"
      >
        <i class="pi pi-download text-lg text-base-foreground" />
      </Button>

      <Button
        v-if="hasRecording && !isRecording"
        v-tooltip.right="{
          value: $t('load3d.clearRecording'),
          showDelay: 300
        }"
        size="icon"
        variant="textonly"
        class="rounded-full"
        :aria-label="$t('load3d.clearRecording')"
        @click="handleClearRecording"
      >
        <i class="pi pi-trash text-lg text-base-foreground" />
      </Button>

      <div
        v-if="recordingDuration && recordingDuration > 0 && !isRecording"
        class="mt-1 text-center text-xs text-base-foreground"
      >
        {{ formatDuration(recordingDuration) }}
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import Button from '@/components/ui/button/Button.vue'
import { cn } from '@/utils/tailwindUtil'

const hasRecording = defineModel<boolean>('hasRecording')
const isRecording = defineModel<boolean>('isRecording')
const recordingDuration = defineModel<number>('recordingDuration')

const emit = defineEmits<{
  (e: 'startRecording'): void
  (e: 'stopRecording'): void
  (e: 'exportRecording'): void
  (e: 'clearRecording'): void
}>()

function toggleRecording() {
  if (isRecording.value) {
    emit('stopRecording')
  } else {
    emit('startRecording')
  }
}

function handleExportRecording() {
  emit('exportRecording')
}

function handleClearRecording() {
  emit('clearRecording')
}

function formatDuration(seconds: number): string {
  const minutes = Math.floor(seconds / 60)
  const remainingSeconds = Math.floor(seconds % 60)
  return `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`
}
</script>

<style scoped>
.recording-button-blink {
  animation: blink 1s infinite;
}

@keyframes blink {
  0%,
  100% {
    opacity: 1;
  }
  50% {
    opacity: 0.5;
  }
}
</style>
