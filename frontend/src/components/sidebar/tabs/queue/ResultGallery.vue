<template>
  <Galleria
    v-model:visible="galleryVisible"
    :active-index="activeIndex"
    :value="allGalleryItems"
    :show-indicators="false"
    change-item-on-indicator-hover
    :show-item-navigators="hasMultiple"
    full-screen
    :circular="hasMultiple"
    :show-thumbnails="false"
    :pt="{
      mask: {
        onMousedown: onMaskMouseDown,
        onMouseup: onMaskMouseUp,
        'data-mask': true
      },
      prevButton: {
        style: 'position: fixed !important'
      },
      nextButton: {
        style: 'position: fixed !important'
      }
    }"
    @update:visible="handleVisibilityChange"
    @update:active-index="handleActiveIndexChange"
  >
    <template #item="{ item }">
      <ComfyImage
        v-if="item.isImage"
        :key="item.url"
        :src="item.url"
        :contain="false"
        :alt="item.filename"
        class="galleria-image"
      />
      <ResultVideo v-else-if="item.isVideo" :result="item" />
      <ResultAudio v-else-if="item.isAudio" :result="item" />
    </template>
  </Galleria>
</template>

<script setup lang="ts">
import Galleria from 'primevue/galleria'
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'

import ComfyImage from '@/components/common/ComfyImage.vue'
import type { ResultItemImpl } from '@/stores/queueStore'

import ResultAudio from './ResultAudio.vue'
import ResultVideo from './ResultVideo.vue'

const galleryVisible = ref(false)

const emit = defineEmits<{
  (e: 'update:activeIndex', value: number): void
}>()

const props = defineProps<{
  allGalleryItems: ResultItemImpl[]
  activeIndex: number
}>()

const hasMultiple = computed(() => props.allGalleryItems.length > 1)

let maskMouseDownTarget: EventTarget | null = null

const onMaskMouseDown = (event: MouseEvent) => {
  maskMouseDownTarget = event.target
}

const onMaskMouseUp = (event: MouseEvent) => {
  const maskEl = document.querySelector('[data-mask]')
  if (
    galleryVisible.value &&
    maskMouseDownTarget === event.target &&
    maskMouseDownTarget === maskEl
  ) {
    galleryVisible.value = false
    handleVisibilityChange(false)
  }
}

watch(
  () => props.activeIndex,
  (index) => {
    if (index !== -1) {
      galleryVisible.value = true
    }
  }
)

const handleVisibilityChange = (visible: boolean) => {
  if (!visible) {
    emit('update:activeIndex', -1)
  }
}

const handleActiveIndexChange = (index: number) => {
  emit('update:activeIndex', index)
}

const handleKeyDown = (event: KeyboardEvent) => {
  if (!galleryVisible.value) return

  switch (event.key) {
    case 'ArrowLeft':
      navigateImage(-1)
      break
    case 'ArrowRight':
      navigateImage(1)
      break
    case 'Escape':
      galleryVisible.value = false
      handleVisibilityChange(false)
      break
  }
}

const navigateImage = (direction: number) => {
  const newIndex =
    (props.activeIndex + direction + props.allGalleryItems.length) %
    props.allGalleryItems.length
  emit('update:activeIndex', newIndex)
}

onMounted(() => {
  window.addEventListener('keydown', handleKeyDown)
})

onUnmounted(() => {
  window.removeEventListener('keydown', handleKeyDown)
})
</script>

<style>
/* PrimeVue's galleria teleports the fullscreen gallery out of subtree so we
cannot use scoped style here. */
img.galleria-image {
  max-width: 100vw;
  max-height: 100vh;
  object-fit: contain;
}

.p-galleria-close-button {
  /* Set z-index so the close button doesn't get hidden behind the image when image is large */
  z-index: 1;
}

/* Mobile/tablet specific fixes */
@media screen and (max-width: 768px) {
  .p-galleria-prev-button,
  .p-galleria-next-button {
    z-index: 2;
  }
}
</style>
