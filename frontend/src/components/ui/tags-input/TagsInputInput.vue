<script setup lang="ts">
import type { TagsInputInputProps } from 'reka-ui'
import { TagsInputInput, useForwardExpose, useForwardProps } from 'reka-ui'
import { computed, inject, onMounted, onUnmounted, ref } from 'vue'
import type { HTMLAttributes } from 'vue'

import { cn } from '@/utils/tailwindUtil'

import { tagsInputFocusKey, tagsInputIsEditingKey } from './tagsInputContext'

const {
  isEmpty = false,
  class: className,
  ...restProps
} = defineProps<
  TagsInputInputProps & { class?: HTMLAttributes['class']; isEmpty?: boolean }
>()

const forwardedProps = useForwardProps(restProps)
const isEditing = inject(tagsInputIsEditingKey, ref(true))
const showInput = computed(() => isEditing.value || isEmpty)

const { forwardRef, currentElement } = useForwardExpose()
const registerFocus = inject(tagsInputFocusKey, undefined)

function handleEscape() {
  currentElement.value?.blur()
  isEditing.value = false
}

onMounted(() => {
  registerFocus?.(() => currentElement.value?.focus())
})

onUnmounted(() => {
  registerFocus?.(undefined)
})
</script>

<template>
  <TagsInputInput
    v-if="showInput"
    :ref="forwardRef"
    v-bind="forwardedProps"
    :class="
      cn(
        'min-h-6 flex-1 bg-transparent text-xs text-muted-foreground placeholder:text-muted-foreground focus:outline-none appearance-none border-none',
        !isEditing && 'pointer-events-none',
        className
      )
    "
    @keydown.escape.stop="handleEscape"
  />
</template>
