<template>
  <div class="grid grid-cols-[auto_1fr] gap-x-2 gap-y-1">
    <label class="content-center text-xs text-node-component-slot-text">
      {{ $t('boundingBox.x') }}
    </label>
    <ScrubableNumberInput v-model="x" :min="0" :step="1" />
    <label class="content-center text-xs text-node-component-slot-text">
      {{ $t('boundingBox.y') }}
    </label>
    <ScrubableNumberInput v-model="y" :min="0" :step="1" />
    <label class="content-center text-xs text-node-component-slot-text">
      {{ $t('boundingBox.width') }}
    </label>
    <ScrubableNumberInput v-model="width" :min="1" :step="1" />
    <label class="content-center text-xs text-node-component-slot-text">
      {{ $t('boundingBox.height') }}
    </label>
    <ScrubableNumberInput v-model="height" :min="1" :step="1" />
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

import ScrubableNumberInput from '@/components/common/ScrubableNumberInput.vue'
import type { Bounds } from '@/renderer/core/layout/types'

const modelValue = defineModel<Bounds>({
  default: () => ({ x: 0, y: 0, width: 512, height: 512 })
})

const x = computed({
  get: () => modelValue.value.x,
  set: (x) => {
    modelValue.value = { ...modelValue.value, x }
  }
})

const y = computed({
  get: () => modelValue.value.y,
  set: (y) => {
    modelValue.value = { ...modelValue.value, y }
  }
})

const width = computed({
  get: () => modelValue.value.width,
  set: (width) => {
    modelValue.value = { ...modelValue.value, width }
  }
})

const height = computed({
  get: () => modelValue.value.height,
  set: (height) => {
    modelValue.value = { ...modelValue.value, height }
  }
})
</script>
