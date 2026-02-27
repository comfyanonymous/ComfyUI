<template>
  <div class="flex flex-col gap-1">
    <Galleria
      v-model:active-index="activeIndex"
      :value="galleryImages"
      v-bind="filteredProps"
      :show-thumbnails="showThumbnails"
      :show-item-navigators="showNavButtons"
      class="max-w-full"
      :pt="{
        thumbnails: {
          class: 'overflow-hidden'
        },
        thumbnailContent: {
          class: 'py-4 px-2'
        },
        thumbnailPrevButton: {
          class: 'm-0'
        },
        thumbnailNextButton: {
          class: 'm-0'
        }
      }"
    >
      <template #item="{ item }">
        <img
          :src="item?.itemImageSrc || item?.src || ''"
          :alt="
            item?.alt ||
            `${t('g.galleryImage')} ${activeIndex + 1} of ${galleryImages.length}`
          "
          class="h-auto max-h-64 w-full object-contain"
        />
      </template>
      <template #thumbnail="{ item }">
        <div class="h-full w-full p-1">
          <img
            :src="item?.thumbnailImageSrc || item?.src || ''"
            :alt="
              item?.alt ||
              `${t('g.galleryThumbnail')} ${galleryImages.findIndex((img) => img === item) + 1} of ${galleryImages.length}`
            "
            class="h-full w-full rounded-lg object-cover"
          />
        </div>
      </template>
    </Galleria>
  </div>
</template>

<script setup lang="ts">
import Galleria from 'primevue/galleria'
import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'

import type { SimplifiedWidget } from '@/types/simplifiedWidget'
import {
  GALLERIA_EXCLUDED_PROPS,
  filterWidgetProps
} from '@/utils/widgetPropFilter'

export interface GalleryImage {
  itemImageSrc?: string
  thumbnailImageSrc?: string
  src?: string
  alt?: string
}

export type GalleryValue = string[] | GalleryImage[]

const value = defineModel<GalleryValue>({ required: true })

const props = defineProps<{
  widget: SimplifiedWidget<GalleryValue>
}>()

const activeIndex = ref(0)

const { t } = useI18n()

const filteredProps = computed(() =>
  filterWidgetProps(props.widget.options, GALLERIA_EXCLUDED_PROPS)
)

const galleryImages = computed(() => {
  if (!value.value || !Array.isArray(value.value)) return []

  return value.value
    .filter((item) => item !== null && item !== undefined) // Filter out null/undefined
    .map((item, index) => {
      if (typeof item === 'string') {
        return {
          itemImageSrc: item,
          thumbnailImageSrc: item,
          alt: `Image ${index}`
        }
      }
      return item ?? {} // Ensure we have at least an empty object
    })
})

const showThumbnails = computed(() => {
  return (
    props.widget.options?.showThumbnails !== false &&
    galleryImages.value.length > 1
  )
})

const showNavButtons = computed(() => {
  return (
    props.widget.options?.showItemNavigators !== false &&
    galleryImages.value.length > 1
  )
})
</script>

<style scoped>
/* Ensure thumbnail container doesn't overflow */
:deep(.p-galleria-thumbnails) {
  overflow: hidden;
}

/* Constrain thumbnail items to prevent overlap */
:deep(.p-galleria-thumbnail-item) {
  flex-shrink: 0;
}

/* Ensure thumbnail wrapper maintains aspect ratio */
:deep(.p-galleria-thumbnail) {
  overflow: hidden;
}
</style>
