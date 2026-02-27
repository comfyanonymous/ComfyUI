<script setup lang="ts">
import {
  getMediaType,
  mediaTypes
} from '@/renderer/extensions/linearMode/mediaTypes'
import type { ResultItemImpl } from '@/stores/queueStore'
import { cn } from '@/utils/tailwindUtil'

const { output } = defineProps<{
  output: ResultItemImpl
}>()
</script>
<template>
  <img
    v-if="getMediaType(output) === 'images'"
    class="block size-10 rounded-sm object-cover bg-secondary-background"
    loading="lazy"
    width="40"
    height="40"
    :src="output.url"
  />
  <i
    v-else
    :class="cn(mediaTypes[getMediaType(output)]?.iconClass, 'block size-10')"
  />
</template>
