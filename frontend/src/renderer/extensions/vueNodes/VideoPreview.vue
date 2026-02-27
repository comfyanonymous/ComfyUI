<template>
  <div
    v-if="imageUrls.length > 0"
    class="video-preview group relative flex size-full min-h-16 min-w-16 flex-col px-2"
    @keydown="handleKeyDown"
  >
    <!-- Video Wrapper -->
    <div
      ref="videoWrapperEl"
      class="relative flex flex-1 overflow-hidden rounded-[5px] bg-node-component-surface"
      tabindex="0"
      role="region"
      :aria-label="$t('g.videoPreview')"
      :aria-busy="showLoader"
      @mouseenter="handleMouseEnter"
      @mouseleave="handleMouseLeave"
      @focusin="handleFocusIn"
      @focusout="handleFocusOut"
    >
      <!-- Error State -->
      <div
        v-if="videoError"
        role="alert"
        class="flex flex-auto flex-col items-center justify-center bg-muted-background text-center text-base-foreground py-8"
      >
        <i
          class="mb-2 icon-[lucide--video-off] h-12 w-12 text-base-foreground"
        />
        <p class="text-sm text-base-foreground">
          {{ $t('g.videoFailedToLoad') }}
        </p>
        <p class="mt-1 text-xs text-base-foreground">
          {{ getVideoFilename(currentVideoUrl) }}
        </p>
      </div>

      <!-- Loading State -->
      <Skeleton
        v-if="showLoader && !videoError"
        class="absolute inset-0 size-full"
        border-radius="5px"
        width="100%"
        height="100%"
      />

      <!-- Main Video -->
      <video
        v-if="!videoError"
        :src="currentVideoUrl"
        :class="cn('block size-full object-contain', showLoader && 'invisible')"
        controls
        loop
        playsinline
        @loadeddata="handleVideoLoad"
        @error="handleVideoError"
      />

      <!-- Floating Action Buttons (appear on hover) -->
      <div
        v-if="isHovered || isFocused"
        class="actions absolute top-2 right-2 flex gap-2.5"
      >
        <!-- Download Button -->
        <button
          :class="actionButtonClass"
          :title="$t('g.downloadVideo')"
          :aria-label="$t('g.downloadVideo')"
          @click="handleDownload"
        >
          <i class="icon-[lucide--download] h-4 w-4" />
        </button>

        <!-- Close Button -->
        <button
          :class="actionButtonClass"
          :title="$t('g.removeVideo')"
          :aria-label="$t('g.removeVideo')"
          @click="handleRemove"
        >
          <i class="icon-[lucide--x] h-4 w-4" />
        </button>
      </div>

      <!-- Multiple Videos Navigation -->
      <div
        v-if="hasMultipleVideos"
        class="absolute right-2 bottom-2 left-2 flex justify-center gap-1"
      >
        <button
          v-for="(_, index) in imageUrls"
          :key="index"
          :class="getNavigationDotClass(index)"
          :aria-label="
            $t('g.viewVideoOfTotal', {
              index: index + 1,
              total: imageUrls.length
            })
          "
          @click="setCurrentIndex(index)"
        />
      </div>
    </div>

    <!-- Video Dimensions -->
    <div class="mt-2 text-center text-xs text-muted-foreground">
      <span v-if="videoError" class="text-red-400">
        {{ $t('g.errorLoadingVideo') }}
      </span>
      <span v-else-if="showLoader" class="text-smoke-400">
        {{ $t('g.loading') }}...
      </span>
      <span v-else>
        {{ actualDimensions || $t('g.calculatingDimensions') }}
      </span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { useToast } from 'primevue'
import Skeleton from 'primevue/skeleton'
import { computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'

import { downloadFile } from '@/base/common/downloadUtil'
import { useNodeOutputStore } from '@/stores/imagePreviewStore'
import { cn } from '@/utils/tailwindUtil'

interface VideoPreviewProps {
  /** Array of video URLs to display */
  readonly imageUrls: readonly string[] // Named imageUrls for consistency with parent components
  /** Optional node ID for context-aware actions */
  readonly nodeId?: string
}

const props = defineProps<VideoPreviewProps>()

const { t } = useI18n()
const nodeOutputStore = useNodeOutputStore()

const actionButtonClass =
  'flex h-8 min-h-8 items-center justify-center gap-2.5 rounded-lg border-0 bg-button-surface px-2 py-2 text-button-surface-contrast shadow-sm transition-colors duration-200 hover:bg-button-hover-surface focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-button-surface-contrast focus-visible:ring-offset-2 focus-visible:ring-offset-transparent cursor-pointer'

// Component state
const currentIndex = ref(0)
const isHovered = ref(false)
const isFocused = ref(false)
const actualDimensions = ref<string | null>(null)
const videoError = ref(false)
const showLoader = ref(false)

const videoWrapperEl = ref<HTMLDivElement>()

// Computed values
const currentVideoUrl = computed(() => props.imageUrls[currentIndex.value])
const hasMultipleVideos = computed(() => props.imageUrls.length > 1)

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
    videoError.value = false
    showLoader.value = newUrls.length > 0
  },
  { immediate: true }
)

// Event handlers
const handleVideoLoad = (event: Event) => {
  if (!event.target || !(event.target instanceof HTMLVideoElement)) return
  const video = event.target
  showLoader.value = false
  videoError.value = false
  if (video.videoWidth && video.videoHeight) {
    actualDimensions.value = `${video.videoWidth} x ${video.videoHeight}`
  }
}

const handleVideoError = () => {
  showLoader.value = false
  videoError.value = true
  actualDimensions.value = null
}

const handleDownload = () => {
  try {
    downloadFile(currentVideoUrl.value)
  } catch (error) {
    useToast().add({
      severity: 'error',
      summary: 'Error',
      detail: t('g.failedToDownloadVideo'),
      life: 3000,
      group: 'video-preview'
    })
  }
}

const handleRemove = () => {
  if (!props.nodeId) return
  nodeOutputStore.removeNodeOutputs(props.nodeId)
}

const setCurrentIndex = (index: number) => {
  if (index >= 0 && index < props.imageUrls.length) {
    currentIndex.value = index
    actualDimensions.value = null
    showLoader.value = true
    videoError.value = false
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
  if (!videoWrapperEl.value?.contains(event.relatedTarget as Node)) {
    isFocused.value = false
  }
}

const getNavigationDotClass = (index: number) => {
  return [
    'w-2 h-2 rounded-full transition-all duration-200 border-0 cursor-pointer',
    index === currentIndex.value ? 'bg-white' : 'bg-white/50 hover:bg-white/80'
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

const getVideoFilename = (url: string): string => {
  try {
    return new URL(url).searchParams.get('filename') || 'Unknown file'
  } catch {
    return 'Invalid URL'
  }
}
</script>
