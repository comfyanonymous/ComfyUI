<script setup lang="ts">
import type { SelectGroupProps } from 'reka-ui'
import { SelectGroup } from 'reka-ui'
import type { HTMLAttributes } from 'vue'

import { cn } from '@/utils/tailwindUtil'

const { class: className, ...restProps } = defineProps<
  SelectGroupProps & { class?: HTMLAttributes['class'] }
>()
</script>

<template>
  <SelectGroup :class="cn('w-full', className)" v-bind="restProps">
    <slot />
  </SelectGroup>
</template>
