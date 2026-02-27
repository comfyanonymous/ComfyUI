<script setup lang="ts">
import type { SelectSeparatorProps } from 'reka-ui'
import { SelectSeparator } from 'reka-ui'
import type { HTMLAttributes } from 'vue'

import { cn } from '@/utils/tailwindUtil'

const { class: className, ...restProps } = defineProps<
  SelectSeparatorProps & { class?: HTMLAttributes['class'] }
>()
</script>

<template>
  <SelectSeparator
    v-bind="restProps"
    :class="cn('-mx-1 my-1 h-px bg-border-default', className)"
  />
</template>
