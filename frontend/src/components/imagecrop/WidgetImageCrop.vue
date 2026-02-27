<template>
  <div
    class="widget-expands relative flex h-full w-full flex-col gap-1"
    @pointerdown.stop
    @pointermove.stop
    @pointerup.stop
  >
    <!-- Image preview container -->
    <div
      ref="containerEl"
      class="relative min-h-0 flex-1 overflow-hidden rounded-[5px] bg-node-component-surface"
    >
      <div v-if="isLoading" class="flex size-full items-center justify-center">
        <span class="text-sm">{{ $t('imageCrop.loading') }}</span>
      </div>

      <div
        v-else-if="!imageUrl"
        class="flex size-full flex-col items-center justify-center text-center"
      >
        <i class="mb-2 icon-[lucide--image] h-12 w-12" />
        <p class="text-sm">{{ $t('imageCrop.noInputImage') }}</p>
      </div>

      <img
        v-else
        ref="imageEl"
        :src="imageUrl"
        :alt="$t('imageCrop.cropPreviewAlt')"
        draggable="false"
        class="block size-full object-contain select-none"
        @load="handleImageLoad"
        @error="handleImageError"
        @dragstart.prevent
      />

      <div
        v-if="imageUrl && !isLoading"
        class="absolute box-content cursor-move border-2 border-white shadow-[0_0_0_9999px_rgba(0,0,0,0.5)]"
        :style="cropBoxStyle"
        @pointerdown="handleDragStart"
        @pointermove="handleDragMove"
        @pointerup="handleDragEnd"
      />

      <div
        v-for="handle in resizeHandles"
        v-show="imageUrl && !isLoading"
        :key="handle.direction"
        :class="['absolute', handle.class]"
        :style="handle.style"
        @pointerdown="(e) => handleResizeStart(e, handle.direction)"
        @pointermove="handleResizeMove"
        @pointerup="handleResizeEnd"
      />
    </div>

    <div class="flex shrink-0 items-center gap-2">
      <label class="text-xs text-muted-foreground">
        {{ $t('imageCrop.ratio') }}
      </label>
      <Select v-model="selectedRatio">
        <SelectTrigger class="h-7 w-24 text-xs">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem v-for="key in ratioKeys" :key="key" :value="key">
            {{ key === 'custom' ? $t('imageCrop.custom') : key }}
          </SelectItem>
        </SelectContent>
      </Select>
      <Button
        size="icon"
        :variant="isLockEnabled ? 'primary' : 'secondary'"
        class="size-7"
        :aria-label="
          isLockEnabled
            ? $t('imageCrop.unlockRatio')
            : $t('imageCrop.lockRatio')
        "
        @click="isLockEnabled = !isLockEnabled"
      >
        <i
          :class="
            isLockEnabled
              ? 'icon-[lucide--lock] size-3.5'
              : 'icon-[lucide--lock-open] size-3.5'
          "
        />
      </Button>
    </div>

    <WidgetBoundingBox v-model="modelValue" class="shrink-0" />
  </div>
</template>

<script setup lang="ts">
import { useTemplateRef } from 'vue'

import WidgetBoundingBox from '@/components/boundingbox/WidgetBoundingBox.vue'
import Button from '@/components/ui/button/Button.vue'
import Select from '@/components/ui/select/Select.vue'
import SelectContent from '@/components/ui/select/SelectContent.vue'
import SelectItem from '@/components/ui/select/SelectItem.vue'
import SelectTrigger from '@/components/ui/select/SelectTrigger.vue'
import SelectValue from '@/components/ui/select/SelectValue.vue'
import { ASPECT_RATIOS, useImageCrop } from '@/composables/useImageCrop'
import type { NodeId } from '@/platform/workflow/validation/schemas/workflowSchema'
import type { Bounds } from '@/renderer/core/layout/types'

const props = defineProps<{
  nodeId: NodeId
}>()

const modelValue = defineModel<Bounds>({
  default: () => ({ x: 0, y: 0, width: 512, height: 512 })
})

const imageEl = useTemplateRef<HTMLImageElement>('imageEl')
const containerEl = useTemplateRef<HTMLDivElement>('containerEl')

const ratioKeys = Object.keys(ASPECT_RATIOS)

const {
  imageUrl,
  isLoading,

  selectedRatio,
  isLockEnabled,

  cropBoxStyle,
  resizeHandles,

  handleImageLoad,
  handleImageError,
  handleDragStart,
  handleDragMove,
  handleDragEnd,
  handleResizeStart,
  handleResizeMove,
  handleResizeEnd
} = useImageCrop(props.nodeId, { imageEl, containerEl, modelValue })
</script>
