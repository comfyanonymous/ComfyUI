<template>
  <div class="widget-markdown relative w-full" @dblclick="startEditing">
    <!-- Display mode: Rendered markdown -->
    <div
      class="comfy-markdown-content size-full min-h-[60px] overflow-y-auto rounded-lg text-sm"
      :class="isEditing === false ? 'visible' : 'invisible'"
      tabindex="0"
      data-capture-wheel="true"
      v-html="renderedHtml"
    />

    <!-- Edit mode: Textarea -->
    <Textarea
      v-show="isEditing"
      ref="textareaRef"
      v-model="modelValue"
      :aria-label="`${$t('g.edit')} ${widget.name || $t('g.markdown')} ${$t('g.content')}`"
      class="absolute inset-0 min-h-[60px] w-full resize-none"
      :pt="{
        root: {
          class: 'text-sm w-full h-full',
          onBlur: handleBlur
        }
      }"
      data-capture-wheel="true"
      @pointerdown.capture.stop
      @pointermove.capture.stop
      @pointerup.capture.stop
      @click.stop
      @keydown.stop
    />
  </div>
</template>

<script setup lang="ts">
import Textarea from 'primevue/textarea'
import { computed, nextTick, ref } from 'vue'

import type { SimplifiedWidget } from '@/types/simplifiedWidget'
import { renderMarkdownToHtml } from '@/utils/markdownRendererUtil'

const { widget } = defineProps<{
  widget: SimplifiedWidget<string>
}>()

const modelValue = defineModel<string>({ default: '' })

// State
const isEditing = ref(false)
const textareaRef = ref<InstanceType<typeof Textarea> | undefined>()

// Computed
const renderedHtml = computed(() => {
  return renderMarkdownToHtml(modelValue.value || '')
})

// Methods
const startEditing = async () => {
  if (isEditing.value || widget.options?.read_only) return

  isEditing.value = true
  await nextTick()

  // Focus the textarea
  // @ts-expect-error - $el is an internal property of the Textarea component
  textareaRef.value?.$el?.focus()
}

const handleBlur = () => {
  isEditing.value = false
}
</script>
