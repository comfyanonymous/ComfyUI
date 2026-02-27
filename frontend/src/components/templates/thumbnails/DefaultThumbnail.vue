<template>
  <BaseThumbnail :hover-zoom="hoverZoom" :is-hovered="isHovered">
    <LazyImage
      :src="src"
      :alt="alt"
      :image-class="[
        'transform-gpu transition-transform duration-300 ease-out',
        isVideoType
          ? 'w-full h-full object-cover'
          : 'max-w-full max-h-64 object-contain'
      ]"
      :image-style="
        isHovered ? { transform: `scale(${1 + hoverZoom / 100})` } : undefined
      "
    />
  </BaseThumbnail>
</template>

<script setup lang="ts">
import LazyImage from '@/components/common/LazyImage.vue'
import BaseThumbnail from '@/components/templates/thumbnails/BaseThumbnail.vue'

const { src, isVideo } = defineProps<{
  src: string
  alt: string
  hoverZoom: number
  isHovered?: boolean
  isVideo?: boolean
}>()

const isVideoType = isVideo ?? (src?.toLowerCase().endsWith('.webp') || false)
</script>
