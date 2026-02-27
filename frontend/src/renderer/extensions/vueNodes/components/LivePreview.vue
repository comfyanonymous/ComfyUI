<template>
  <template v-if="imageUrl">
    <div
      v-if="imageError"
      class="text-pure-white flex h-full w-full flex-col items-center justify-center text-center"
    >
      <i-lucide:image-off class="mb-1 size-8 text-smoke-500" />
      <p class="text-xs text-smoke-400">{{ $t('g.imageFailedToLoad') }}</p>
    </div>
    <img
      v-else
      :src="imageUrl"
      :alt="$t('g.liveSamplingPreview')"
      class="pointer-events-none w-full object-contain contain-size min-h-55 flex-1"
      @load="handleImageLoad"
      @error="handleImageError"
    />
    <div class="text-node-component-header-text mt-1 text-center text-xs">
      {{
        imageError
          ? $t('g.errorLoadingImage')
          : actualDimensions || $t('g.calculatingDimensions')
      }}
    </div>
  </template>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue'

interface LivePreviewProps {
  imageUrl: string
}

const props = defineProps<LivePreviewProps>()

const actualDimensions = ref<string | null>(null)
const imageError = ref(false)

watch(
  () => props.imageUrl,
  () => {
    // Reset states when URL changes
    actualDimensions.value = null
    imageError.value = false
  }
)

const handleImageLoad = (event: Event) => {
  if (!event.target || !(event.target instanceof HTMLImageElement)) return
  const img = event.target
  imageError.value = false
  if (img.naturalWidth && img.naturalHeight) {
    actualDimensions.value = `${img.naturalWidth} x ${img.naturalHeight}`
  }
}

const handleImageError = () => {
  imageError.value = true
  actualDimensions.value = null
}
</script>
