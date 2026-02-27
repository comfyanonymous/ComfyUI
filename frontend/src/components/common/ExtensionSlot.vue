<template>
  <component :is="extension.component" v-if="extension.type === 'vue'" />
  <div
    v-else
    :ref="
      (el) => {
        if (el)
          mountCustomExtension(
            props.extension as CustomExtension,
            el as HTMLElement
          )
      }
    "
  />
</template>

<script setup lang="ts">
import { onBeforeUnmount } from 'vue'

import type { CustomExtension, VueExtension } from '@/types/extensionTypes'

const props = defineProps<{
  extension: VueExtension | CustomExtension
}>()

const mountCustomExtension = (extension: CustomExtension, el: HTMLElement) => {
  extension.render(el)
}

onBeforeUnmount(() => {
  if (props.extension.type === 'custom' && props.extension.destroy) {
    props.extension.destroy()
  }
})
</script>
