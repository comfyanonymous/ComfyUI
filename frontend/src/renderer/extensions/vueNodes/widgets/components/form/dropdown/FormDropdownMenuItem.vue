<script setup lang="ts">
import { computed, inject, ref } from 'vue'

import { cn } from '@/utils/tailwindUtil'

import { AssetKindKey } from './types'
import type { LayoutMode } from './types'

interface Props {
  index: number
  selected: boolean
  previewUrl: string
  name: string
  label?: string
  layout?: LayoutMode
}

const props = defineProps<Props>()

const emit = defineEmits<{
  click: [index: number]
  mediaLoad: [event: Event]
}>()

const actualDimensions = ref<string | null>(null)

const assetKind = inject(AssetKindKey)

const isVideo = computed(() => assetKind?.value === 'video')

function handleClick() {
  emit('click', props.index)
}

function handleImageLoad(event: Event) {
  emit('mediaLoad', event)
  if (!event.target || !(event.target instanceof HTMLImageElement)) return
  const img = event.target
  if (img.naturalWidth && img.naturalHeight) {
    actualDimensions.value = `${img.naturalWidth} x ${img.naturalHeight}`
  }
}

function handleVideoLoad(event: Event) {
  emit('mediaLoad', event)
  if (!event.target || !(event.target instanceof HTMLVideoElement)) return
  const video = event.target
  if (video.videoWidth && video.videoHeight) {
    actualDimensions.value = `${video.videoWidth} x ${video.videoHeight}`
  }
}
</script>

<template>
  <div
    :class="
      cn(
        'flex gap-1 select-none group/item cursor-pointer bg-component-node-widget-background',
        'transition-[transform,box-shadow,background-color] duration-150',
        {
          'flex-col text-center': layout === 'grid',
          'flex-row text-left max-h-16 rounded-lg hover:scale-102 active:scale-98':
            layout === 'list',
          'flex-row text-left hover:bg-component-node-widget-background-hovered rounded-lg':
            layout === 'list-small',
          // selection
          'ring-2 ring-component-node-widget-background-highlighted':
            layout === 'list' && selected
        }
      )
    "
    @click="handleClick"
  >
    <!-- Image -->
    <div
      v-if="layout !== 'list-small'"
      :class="
        cn(
          'relative',
          'w-full aspect-square overflow-hidden outline-1 outline-offset-[-1px] outline-interface-stroke',
          'transition-[transform,box-shadow] duration-150',
          {
            'min-w-16 max-w-16 rounded-l-lg': layout === 'list',
            'rounded-sm group-hover/item:scale-108 group-active/item:scale-95':
              layout === 'grid',
            // selection
            'ring-2 ring-component-node-widget-background-highlighted':
              layout === 'grid' && selected
          }
        )
      "
    >
      <!-- Selected Icon -->
      <div
        v-if="selected"
        class="absolute top-1 left-1 size-4 rounded-full border-1 border-base-foreground bg-primary-background"
      >
        <i
          class="icon-[lucide--check] size-3 translate-y-[-0.5px] text-base-foreground bold"
        />
      </div>
      <video
        v-if="previewUrl && isVideo"
        :src="previewUrl"
        class="size-full object-cover"
        preload="metadata"
        muted
        @loadeddata="handleVideoLoad"
      />
      <img
        v-else-if="previewUrl"
        :src="previewUrl"
        :alt="name"
        draggable="false"
        class="size-full object-cover"
        @load="handleImageLoad"
      />
      <div
        v-else
        class="size-full bg-gradient-to-tr from-blue-400 via-teal-500 to-green-400"
      />
    </div>
    <!-- Name -->
    <div
      :class="
        cn('flex gap-1', {
          'flex-col': layout === 'grid',
          'flex-col px-4 py-1 w-full justify-center min-w-0': layout === 'list',
          'flex-row p-2 items-center justify-between w-full':
            layout === 'list-small'
        })
      "
    >
      <span
        v-tooltip="layout === 'grid' ? (label ?? name) : undefined"
        :class="
          cn(
            'block text-xs line-clamp-2 break-words overflow-hidden',
            'transition-colors duration-150',
            // selection
            !!selected && 'text-base-foreground'
          )
        "
      >
        {{ label ?? name }}
      </span>
      <!-- Meta Data -->
      <span v-if="actualDimensions" class="text-secondary block text-xs">
        {{ actualDimensions }}
      </span>
    </div>
  </div>
</template>
