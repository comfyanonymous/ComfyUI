<script setup lang="ts">
import type { TagsInputItemTextProps } from 'reka-ui'
import { TagsInputItemText, useForwardProps } from 'reka-ui'
import type { HTMLAttributes } from 'vue'

import { cn } from '@/utils/tailwindUtil'

const { class: className, ...restProps } = defineProps<
  TagsInputItemTextProps & { class?: HTMLAttributes['class'] }
>()

const forwardedProps = useForwardProps(restProps)
</script>

<template>
  <TagsInputItemText
    v-bind="forwardedProps"
    :class="cn('bg-transparent text-xs', className)"
  />
</template>
