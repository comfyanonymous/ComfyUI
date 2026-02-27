<template>
  <div
    v-if="imageUrls.length > 0"
    class="image-preview group relative flex size-full min-h-55 min-w-16 flex-col px-2 justify-center"
    @keydown="handleKeyDown"
  >
    <!-- Image Wrapper -->
    <div
      ref="imageWrapperEl"
      class="min-h-0 flex-1 flex w-full overflow-hidden rounded-[5px] bg-muted-background relative"
      tabindex="0"
      role="img"
      :aria-label="$t('g.imagePreview')"
      :aria-busy="showLoader"
      @mouseenter="handleMouseEnter"
      @mouseleave="handleMouseLeave"
      @focusin="handleFocusIn"
      @focusout="handleFocusOut"
    >
      <!-- Error State -->
      <div
        v-if="imageError"
        role="alert"
        class="flex flex-1 self-center size-full flex-col items-center justify-around bg-muted-background text-center text-base-foreground py-8"
      >
        <i
          class="mb-2 icon-[lucide--image-off] h-12 w-12 text-base-foreground"
        />
        <p class="text-sm text-base-foreground">
          {{ $t('g.imageFailedToLoad') }}
        </p>
        <p class="mt-1 text-xs text-base-foreground">
          {{ getImageFilename(currentImageUrl) }}
        </p>
      </div>
      <!-- Loading State -->
      <div v-if="showLoader && !imageError" class="size-full">
        <Skeleton border-radius="5px" width="100%" height="100%" />
      </div>
      <!-- Main Image -->
      <img
        v-if="!imageError"
        :src="currentImageUrl"
        :alt="imageAltText"
        class="absolute inset-0 block size-full object-contain pointer-events-none"
        @load="handleImageLoad"
        @error="handleImageError"
      />

      <!-- Floating Action Buttons (appear on hover and focus) -->
      <div
        v-if="isHovered || isFocused"
        class="actions absolute top-2 right-2 flex gap-2.5"
      >
        <!-- Mask/Edit Button -->
        <button
          v-if="!hasMultipleImages"
          :class="actionButtonClass"
          :title="$t('g.editOrMaskImage')"
          :aria-label="$t('g.editOrMaskImage')"
          @click="handleEditMask"
        >
          <i-comfy:mask class="h-4 w-4" />
        </button>

        <!-- Download Button -->
        <button
          :class="actionButtonClass"
          :title="$t('g.downloadImage')"
          :aria-label="$t('g.downloadImage')"
          @click="handleDownload"
        >
          <i class="icon-[lucide--download] h-4 w-4" />
        </button>

        <!-- Close Button -->
        <button
          :class="actionButtonClass"
          :title="$t('g.removeImage')"
          :aria-label="$t('g.removeImage')"
          @click="handleRemove"
        >
          <i class="icon-[lucide--x] h-4 w-4" />
        </button>
      </div>
    </div>

    <!-- Image Dimensions -->
    <div class="pt-2 text-center text-xs text-base-foreground">
      <span v-if="imageError" class="text-red-400">
        {{ $t('g.errorLoadingImage') }}
      </span>
      <span v-else-if="showLoader" class="text-base-foreground">
        {{ $t('g.loading') }}...
      </span>
      <span v-else>
        {{ actualDimensions || $t('g.calculatingDimensions') }}
      </span>
    </div>
    <!-- Multiple Images Navigation -->
    <div
      v-if="hasMultipleImages"
      class="flex flex-wrap justify-center gap-1 pt-4"
    >
      <button
        v-for="(_, index) in imageUrls"
        :key="index"
        :class="getNavigationDotClass(index)"
        :aria-current="index === currentIndex ? 'true' : undefined"
        :aria-label="
          $t('g.viewImageOfTotal', {
            index: index + 1,
            total: imageUrls.length
          })
        "
        @click="setCurrentIndex(index)"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { useTimeoutFn } from '@vueuse/core'
import { useToast } from 'primevue'
import Skeleton from 'primevue/skeleton'
import { computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'

import { downloadFile } from '@/base/common/downloadUtil'
import { useMaskEditor } from '@/composables/maskeditor/useMaskEditor'
import { app } from '@/scripts/app'
import { useNodeOutputStore } from '@/stores/imagePreviewStore'

interface ImagePreviewProps {
  /** Array of image URLs to display */
  readonly imageUrls: readonly string[]
  /** Optional node ID for context-aware actions */
  readonly nodeId?: string
}

const props = defineProps<ImagePreviewProps>()

const { t } = useI18n()
const maskEditor = useMaskEditor()
const nodeOutputStore = useNodeOutputStore()

const actionButtonClass =
  'flex h-8 min-h-8 items-center justify-center gap-2.5 rounded-lg border-0 bg-button-surface px-2 py-2 text-button-surface-contrast shadow-sm transition-colors duration-200 hover:bg-button-hover-surface focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-button-surface-contrast focus-visible:ring-offset-2 focus-visible:ring-offset-transparent cursor-pointer'

// Component state
const currentIndex = ref(0)
const isHovered = ref(false)
const isFocused = ref(false)
const actualDimensions = ref<string | null>(null)
const imageError = ref(false)
const showLoader = ref(false)

const imageWrapperEl = ref<HTMLDivElement>()

const { start: startDelayedLoader, stop: stopDelayedLoader } = useTimeoutFn(
  () => {
    showLoader.value = true
  },
  250,
  // Make sure it doesnt run on component mount
  { immediate: false }
)

// Computed values
const currentImageUrl = computed(() => props.imageUrls[currentIndex.value])
const hasMultipleImages = computed(() => props.imageUrls.length > 1)
const imageAltText = computed(() => `Node output ${currentIndex.value + 1}`)

// Watch for URL changes and reset state
watch(
  () => props.imageUrls,
  (newUrls, oldUrls) => {
    // Only reset state if URLs actually changed (not just array reference)
    const urlsChanged =
      !oldUrls ||
      newUrls.length !== oldUrls.length ||
      newUrls.some((url, i) => url !== oldUrls[i])

    if (!urlsChanged) return

    // Reset current index if it's out of bounds
    if (currentIndex.value >= newUrls.length) {
      currentIndex.value = 0
    }

    // Reset loading and error states when URLs change
    actualDimensions.value = null

    imageError.value = false
    if (newUrls.length > 0) startDelayedLoader()
  },
  { immediate: true }
)

// Event handlers
const handleImageLoad = (event: Event) => {
  if (!event.target || !(event.target instanceof HTMLImageElement)) return
  const img = event.target
  stopDelayedLoader()
  showLoader.value = false
  imageError.value = false
  if (img.naturalWidth && img.naturalHeight) {
    actualDimensions.value = `${img.naturalWidth} x ${img.naturalHeight}`
  }

  if (props.nodeId) {
    nodeOutputStore.syncLegacyNodeImgs(props.nodeId, img, currentIndex.value)
  }
}

const handleImageError = () => {
  stopDelayedLoader()
  showLoader.value = false
  imageError.value = true
  actualDimensions.value = null
}

const handleEditMask = () => {
  if (!props.nodeId) return
  const node = app.rootGraph?.getNodeById(Number(props.nodeId))
  if (!node) return
  maskEditor.openMaskEditor(node)
}

const handleDownload = () => {
  try {
    downloadFile(currentImageUrl.value)
  } catch (error) {
    useToast().add({
      severity: 'error',
      summary: 'Error',
      detail: t('g.failedToDownloadImage'),
      life: 3000,
      group: 'image-preview'
    })
  }
}

const handleRemove = () => {
  if (!props.nodeId) return
  nodeOutputStore.removeNodeOutputs(props.nodeId)
}

const setCurrentIndex = (index: number) => {
  if (currentIndex.value === index) return
  if (index >= 0 && index < props.imageUrls.length) {
    currentIndex.value = index
    startDelayedLoader()
    imageError.value = false
  }
}

const handleMouseEnter = () => {
  isHovered.value = true
}

const handleMouseLeave = () => {
  isHovered.value = false
}

const handleFocusIn = () => {
  isFocused.value = true
}

const handleFocusOut = (event: FocusEvent) => {
  // Only unfocus if focus is leaving the wrapper entirely
  if (!imageWrapperEl.value?.contains(event.relatedTarget as Node)) {
    isFocused.value = false
  }
}

const getNavigationDotClass = (index: number) => {
  return [
    'w-2 h-2 rounded-full transition-all duration-200 border-0 cursor-pointer p-0',
    index === currentIndex.value
      ? 'bg-base-foreground'
      : 'bg-base-foreground/50 hover:bg-base-foreground/80'
  ]
}

const handleKeyDown = (event: KeyboardEvent) => {
  if (props.imageUrls.length <= 1) return

  switch (event.key) {
    case 'ArrowLeft':
      event.preventDefault()
      setCurrentIndex(
        currentIndex.value > 0
          ? currentIndex.value - 1
          : props.imageUrls.length - 1
      )
      break
    case 'ArrowRight':
      event.preventDefault()
      setCurrentIndex(
        currentIndex.value < props.imageUrls.length - 1
          ? currentIndex.value + 1
          : 0
      )
      break
    case 'Home':
      event.preventDefault()
      setCurrentIndex(0)
      break
    case 'End':
      event.preventDefault()
      setCurrentIndex(props.imageUrls.length - 1)
      break
  }
}

const getImageFilename = (url: string): string => {
  if (!url) return t('g.imageDoesNotExist')
  try {
    return new URL(url).searchParams.get('filename') || t('g.unknownFile')
  } catch {
    return t('g.imageDoesNotExist')
  }
}
</script>
