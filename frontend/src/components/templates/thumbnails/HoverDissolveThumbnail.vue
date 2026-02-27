<template>
  <BaseThumbnail :is-hovered="isHovered">
    <div class="relative h-full w-full">
      <div class="absolute inset-0">
        <LazyImage
          :src="baseImageSrc"
          :alt="alt"
          :image-class="baseImageClass"
        />
      </div>
      <div class="absolute inset-0 z-10">
        <LazyImage
          :src="overlayImageSrc"
          :alt="alt"
          :image-class="overlayImageClass"
        />
      </div>
    </div>
  </BaseThumbnail>
</template>

<script setup lang="ts">
import { computed } from 'vue'

import LazyImage from '@/components/common/LazyImage.vue'
import BaseThumbnail from '@/components/templates/thumbnails/BaseThumbnail.vue'

const { baseImageSrc, overlayImageSrc, isVideo, isHovered } = defineProps<{
  baseImageSrc: string
  overlayImageSrc: string
  alt: string
  isHovered: boolean
  isVideo?: boolean
}>()

const isVideoType =
  isVideo ||
  baseImageSrc?.toLowerCase().endsWith('.webp') ||
  overlayImageSrc?.toLowerCase().endsWith('.webp') ||
  false

const baseImageClass = computed(() => {
  const sizeClasses = isVideoType
    ? 'size-full object-cover'
    : 'size-full object-contain'
  return sizeClasses
})

const overlayImageClass = computed(() => {
  const baseClasses = 'size-full transition-opacity duration-300'
  const sizeClasses = isVideoType ? 'object-cover' : 'object-contain'
  const opacityClasses = isHovered ? 'opacity-100' : 'opacity-0'
  return `${baseClasses} ${sizeClasses} ${opacityClasses}`
})
</script>
