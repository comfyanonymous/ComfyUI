<template>
  <div
    class="relative aspect-square w-full overflow-hidden rounded-t-lg select-none"
  >
    <div
      v-if="!error"
      ref="contentRef"
      class="h-full w-full transform-gpu transition-transform duration-1000 ease-out"
      :style="
        isHovered ? { transform: `scale(${1 + hoverZoom / 100})` } : undefined
      "
    >
      <slot />
    </div>
    <div v-else class="flex h-full w-full items-center justify-center">
      <img
        src="/assets/images/default-template.png"
        draggable="false"
        class="h-full w-full transform-gpu object-cover transition-transform duration-300 ease-out"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { useEventListener } from '@vueuse/core'
import { onMounted, ref } from 'vue'

const error = ref(false)
const contentRef = ref<HTMLElement | null>(null)

const { hoverZoom = 4 } = defineProps<{
  hoverZoom?: number
  isHovered?: boolean
}>()

onMounted(() => {
  const images = Array.from(contentRef.value?.getElementsByTagName('img') ?? [])
  images.forEach((img) => {
    useEventListener(img, 'error', () => {
      error.value = true
    })
  })
})
</script>
<style scoped>
img {
  transition: transform 1s cubic-bezier(0.2, 0, 0.4, 1);
}
</style>
