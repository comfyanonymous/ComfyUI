<template>
  <div v-if="audioUrls.length > 0" class="flex flex-col gap-2 px-2">
    <audio
      v-for="(url, index) in audioUrls"
      :key="index"
      :src="url"
      controls
      class="w-full"
    />
  </div>
</template>

<script setup lang="ts">
interface AudioPreviewProps {
  readonly audioUrls: readonly string[]
}

defineProps<AudioPreviewProps>()
</script>
