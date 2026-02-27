<template>
  <div
    v-if="animations && animations.length > 0"
    class="pointer-events-auto absolute top-0 left-0 z-10 flex w-full flex-col items-center gap-2 pt-2"
  >
    <div class="flex items-center justify-center gap-2">
      <Button
        size="icon"
        variant="textonly"
        class="rounded-full"
        :aria-label="$t('g.playPause')"
        @click="togglePlay"
      >
        <i
          :class="[
            'pi',
            playing ? 'pi-pause' : 'pi-play',
            'text-lg text-base-foreground'
          ]"
        />
      </Button>

      <Select
        v-model="selectedSpeed"
        :options="speedOptions"
        option-label="name"
        option-value="value"
        class="w-24"
      />

      <Select
        v-model="selectedAnimation"
        :options="animations"
        option-label="name"
        option-value="index"
        class="w-32"
      />
    </div>

    <div class="flex w-full max-w-xs items-center gap-2 px-4">
      <Slider
        :model-value="[animationProgress]"
        :min="0"
        :max="100"
        :step="0.1"
        class="flex-1"
        @update:model-value="handleSliderChange"
      />
      <span class="min-w-16 text-xs text-base-foreground">
        {{ formatTime(currentTime) }} / {{ formatTime(animationDuration) }}
      </span>
    </div>
  </div>
</template>

<script setup lang="ts">
import Select from 'primevue/select'
import { computed } from 'vue'

import Button from '@/components/ui/button/Button.vue'
import Slider from '@/components/ui/slider/Slider.vue'

type Animation = { name: string; index: number }

const animations = defineModel<Animation[]>('animations')
const playing = defineModel<boolean>('playing')
const selectedSpeed = defineModel<number>('selectedSpeed')
const selectedAnimation = defineModel<number>('selectedAnimation')
const animationProgress = defineModel<number>('animationProgress', {
  default: 0
})
const animationDuration = defineModel<number>('animationDuration', {
  default: 0
})

const emit = defineEmits<{
  seek: [progress: number]
}>()

const speedOptions = [
  { name: '0.1x', value: 0.1 },
  { name: '0.5x', value: 0.5 },
  { name: '1x', value: 1 },
  { name: '1.5x', value: 1.5 },
  { name: '2x', value: 2 }
]

const currentTime = computed(() => {
  if (!animationDuration.value) return 0
  return (animationProgress.value / 100) * animationDuration.value
})

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60)
  const secs = (seconds % 60).toFixed(1)
  return mins > 0 ? `${mins}:${secs.padStart(4, '0')}` : `${secs}s`
}

function togglePlay() {
  playing.value = !playing.value
}

function handleSliderChange(value: number[] | undefined) {
  if (!value) return
  const progress = value[0]
  animationProgress.value = progress
  emit('seek', progress)
}
</script>
