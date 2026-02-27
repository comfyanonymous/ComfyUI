<template>
  <BaseThumbnail>
    <div
      class="flex h-full w-full items-center justify-center p-4"
      :style="{
        backgroundImage: 'url(/assets/images/default-template.png)',
        backgroundRepeat: 'round'
      }"
    >
      <audio controls class="relative w-full" :src="src" @click.stop />
    </div>
  </BaseThumbnail>
</template>

<script setup lang="ts">
import BaseThumbnail from '@/components/templates/thumbnails/BaseThumbnail.vue'

defineProps<{
  src: string
}>()
</script>
