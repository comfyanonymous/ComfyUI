<template>
  <div
    class="relative size-full overflow-hidden rounded bg-black"
    @mouseenter="isHovered = true"
    @mouseleave="isHovered = false"
  >
    <video
      ref="videoElement"
      :controls="shouldShowControls"
      preload="metadata"
      muted
      loop
      playsinline
      class="relative size-full object-contain transition-transform duration-300 group-hover:scale-105 group-data-[selected=true]:scale-105"
      @click.stop="onVideoClick"
      @play="onVideoPlay"
      @pause="onVideoPause"
    >
      <source
        v-if="asset.src"
        :src="asset.src"
        :type="asset.mime_type ?? undefined"
      />
    </video>
    <VideoPlayOverlay :visible="!isPlaying" size="md" />
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'

import type { AssetMeta } from '../schemas/mediaAssetSchema'

import VideoPlayOverlay from './VideoPlayOverlay.vue'

const { asset } = defineProps<{
  asset: AssetMeta
}>()

const emit = defineEmits<{
  videoPlayingStateChanged: [isPlaying: boolean]
  videoControlsChanged: [showControls: boolean]
}>()

const videoElement = ref<HTMLVideoElement | null>(null)
const isHovered = ref(false)
const isPlaying = ref(false)

// Show native controls only while actively playing and hovered.
const shouldShowControls = computed(() => isPlaying.value && isHovered.value)

watch(shouldShowControls, (controlsVisible) => {
  emit('videoControlsChanged', controlsVisible)
})

onMounted(() => {
  emit('videoControlsChanged', shouldShowControls.value)
})

const onVideoPlay = () => {
  isPlaying.value = true
  emit('videoPlayingStateChanged', true)
}

const onVideoPause = () => {
  isPlaying.value = false
  emit('videoPlayingStateChanged', false)
}

const onVideoClick = async () => {
  if (shouldShowControls.value) return

  const video = videoElement.value
  if (!video) return

  if (video.paused || video.ended) {
    await video.play().catch(() => {})
    return
  }

  video.pause()
}
</script>
