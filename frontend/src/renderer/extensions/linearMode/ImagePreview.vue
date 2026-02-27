<script setup lang="ts">
import { ref, useTemplateRef } from 'vue'

import ZoomPane from '@/components/ui/ZoomPane.vue'

const { src } = defineProps<{
  src: string
  mobile?: boolean
}>()

const imageRef = useTemplateRef('imageRef')
const width = ref('')
const height = ref('')
</script>
<template>
  <ZoomPane v-if="!mobile" v-slot="slotProps" class="flex-1 w-full">
    <img
      ref="imageRef"
      :src
      v-bind="slotProps"
      class="h-full object-contain w-full"
      @load="
        () => {
          if (!imageRef) return
          width = `${imageRef.naturalWidth}`
          height = `${imageRef.naturalHeight}`
        }
      "
    />
  </ZoomPane>
  <img
    v-else
    ref="imageRef"
    class="w-full"
    :src
    @load="
      () => {
        if (!imageRef) return
        width = `${imageRef.naturalWidth}`
        height = `${imageRef.naturalHeight}`
      }
    "
  />
  <span class="self-center md:z-10" v-text="`${width} x ${height}`" />
</template>
