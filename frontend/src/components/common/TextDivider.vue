<template>
  <div class="flex items-center">
    <span v-if="position === 'left'" class="mr-2 shrink-0">{{ text }}</span>
    <Divider :align="align" :type="type" :layout="layout" class="grow" />
    <span v-if="position === 'right'" class="ml-2 shrink-0">{{ text }}</span>
  </div>
</template>

<script setup lang="ts">
import Divider from 'primevue/divider'

const {
  text,
  position = 'left',
  align = 'center',
  type = 'solid',
  layout = 'horizontal'
} = defineProps<{
  text: string
  position?: 'left' | 'right'
  align?: 'left' | 'center' | 'right' | 'top' | 'bottom'
  type?: 'solid' | 'dashed' | 'dotted'
  layout?: 'horizontal' | 'vertical'
}>()
</script>
