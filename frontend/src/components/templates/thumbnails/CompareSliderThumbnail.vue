<template>
  <BaseThumbnail :is-hovered="isHovered">
    <LazyImage
      :src="baseImageSrc"
      :alt="alt"
      :image-class="
        isVideoType
          ? 'w-full h-full object-cover'
          : 'max-w-full max-h-64 object-contain'
      "
    />
    <div ref="containerRef" class="absolute inset-0">
      <LazyImage
        :src="overlayImageSrc"
        :alt="alt"
        :image-class="
          isVideoType
            ? 'w-full h-full object-cover'
            : 'max-w-full max-h-64 object-contain'
        "
        :image-style="{
          clipPath: `inset(0 ${100 - sliderPosition}% 0 0)`
        }"
      />
      <div
        class="pointer-events-none absolute inset-y-0 z-10 w-0.5 bg-white/30 backdrop-blur-sm"
        :style="{
          left: `${sliderPosition}%`
        }"
      />
    </div>
  </BaseThumbnail>
</template>

<script setup lang="ts">
import { useMouseInElement } from '@vueuse/core'
import { ref, watch } from 'vue'

import LazyImage from '@/components/common/LazyImage.vue'
import BaseThumbnail from '@/components/templates/thumbnails/BaseThumbnail.vue'

const SLIDER_START_POSITION = 50

const { baseImageSrc, overlayImageSrc, isHovered, isVideo } = defineProps<{
  baseImageSrc: string
  overlayImageSrc: string
  alt: string
  isHovered?: boolean
  isVideo?: boolean
}>()

const isVideoType =
  isVideo ||
  baseImageSrc?.toLowerCase().endsWith('.webp') ||
  overlayImageSrc?.toLowerCase().endsWith('.webp') ||
  false

const sliderPosition = ref(SLIDER_START_POSITION)
const containerRef = ref<HTMLElement | null>(null)

const { elementX, elementWidth, isOutside } = useMouseInElement(containerRef)

// Update slider position based on mouse position when hovered
watch(
  [() => isHovered, elementX, elementWidth, isOutside],
  ([isHovered, x, width, outside]) => {
    if (!isHovered) return
    if (!outside) {
      sliderPosition.value = (x / width) * 100
    }
  }
)
</script>
