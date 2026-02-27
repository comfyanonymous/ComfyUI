<script setup lang="ts">
import { ref, useTemplateRef } from 'vue'

const { src } = defineProps<{
  src: string
}>()

const videoRef = useTemplateRef('videoRef')
const width = ref('')
const height = ref('')
</script>
<template>
  <video
    ref="videoRef"
    :src
    controls
    v-bind="$attrs"
    @loadedmetadata="
      () => {
        if (!videoRef) return
        width = `${videoRef.videoWidth}`
        height = `${videoRef.videoHeight}`
      }
    "
  />
  <span class="self-center z-10" v-text="`${width} x ${height}`" />
</template>
