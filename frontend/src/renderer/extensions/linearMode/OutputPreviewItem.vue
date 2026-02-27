<script setup lang="ts">
import LinearProgressBar from '@/renderer/extensions/linearMode/LinearProgressBar.vue'

const { latentPreview } = defineProps<{
  latentPreview?: string
}>()
</script>
<template>
  <div class="relative size-10">
    <img
      v-if="latentPreview"
      class="block size-10 rounded-sm object-cover"
      :src="latentPreview"
    />
    <div v-else class="size-10 rounded-sm skeleton-shimmer" />
    <LinearProgressBar
      class="absolute inset-0 size-full bg-transparent"
      :overall-opacity="0.7"
    />
  </div>
</template>
