<template>
  <div class="editable-text">
    <span v-if="!isEditing">
      {{ modelValue }}
    </span>
    <!-- Avoid double triggering finishEditing event when keydown.enter is triggered -->
    <InputText
      v-else
      ref="inputRef"
      v-model:model-value="inputValue"
      v-focus
      type="text"
      size="small"
      fluid
      :pt="{
        root: {
          onBlur: finishEditing,
          ...inputAttrs
        }
      }"
      @keydown.enter.capture.stop="blurInputElement"
      @keydown.escape.capture.stop="cancelEditing"
      @click.stop
      @contextmenu.stop
      @pointerdown.stop.capture
      @pointermove.stop.capture
    />
  </div>
</template>

<script setup lang="ts">
import InputText from 'primevue/inputtext'
import { nextTick, ref, watch } from 'vue'

const {
  modelValue,
  isEditing = false,
  inputAttrs = {}
} = defineProps<{
  modelValue: string
  isEditing?: boolean
  inputAttrs?: Record<string, string>
}>()

const emit = defineEmits(['edit', 'cancel'])
const inputValue = ref<string>(modelValue)
const inputRef = ref<InstanceType<typeof InputText> | undefined>()
const isCanceling = ref(false)

const blurInputElement = () => {
  // @ts-expect-error - $el is an internal property of the InputText component
  inputRef.value?.$el.blur()
}
const finishEditing = () => {
  // Don't save if we're canceling
  if (!isCanceling.value) {
    emit('edit', inputValue.value)
  }
  isCanceling.value = false
}
const cancelEditing = () => {
  // Set canceling flag to prevent blur from saving
  isCanceling.value = true
  // Reset to original value
  inputValue.value = modelValue
  // Emit cancel event
  emit('cancel')
  // Blur the input to exit edit mode
  blurInputElement()
}
watch(
  () => isEditing,
  async (newVal) => {
    if (newVal) {
      inputValue.value = modelValue
      await nextTick(() => {
        if (!inputRef.value) return
        const fileName = inputValue.value.includes('.')
          ? inputValue.value.split('.').slice(0, -1).join('.')
          : inputValue.value
        const start = 0
        const end = fileName.length
        // @ts-expect-error - $el is an internal property of the InputText component
        const inputElement = inputRef.value.$el
        inputElement.setSelectionRange?.(start, end)
      })
    }
  },
  { immediate: true }
)
const vFocus = {
  mounted: (el: HTMLElement) => el.focus()
}
</script>

<style scoped>
.editable-text {
  display: inline;
}
.editable-text input {
  width: 100%;
  box-sizing: border-box;
}
</style>
