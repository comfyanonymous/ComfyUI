<template>
  <div class="color-picker-wrapper flex items-center gap-2">
    <ColorPicker v-model="modelValue" v-bind="$attrs" />
    <InputText v-model="modelValue" class="w-28" :placeholder="label" />
  </div>
</template>

<script setup lang="ts">
import ColorPicker from 'primevue/colorpicker'
import InputText from 'primevue/inputtext'

const modelValue = defineModel<string>('modelValue')
defineProps<{
  label?: string
}>()

defineOptions({
  inheritAttrs: false
})
</script>
