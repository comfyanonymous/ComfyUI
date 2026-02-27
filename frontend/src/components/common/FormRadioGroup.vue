<template>
  <div class="flex flex-row gap-4">
    <div
      v-for="option in normalizedOptions"
      :key="option.value"
      class="flex items-center"
    >
      <RadioButton
        :input-id="`${id}-${option.value}`"
        :name="id"
        :value="option.value"
        :model-value="modelValue"
        :aria-describedby="`${option.text}-label`"
        @update:model-value="$emit('update:modelValue', $event)"
      />
      <label :for="`${id}-${option.value}`" class="ml-2 cursor-pointer">
        {{ option.text }}
      </label>
    </div>
  </div>
</template>

<script setup lang="ts" generic="T extends string | number | boolean | null">
import RadioButton from 'primevue/radiobutton'
import { computed } from 'vue'

import type { SettingOption } from '@/platform/settings/types'

const props = defineProps<{
  modelValue: T
  options?: (string | SettingOption | Record<string, string>)[]
  optionLabel?: string
  optionValue?: string
  id?: string
}>()

defineEmits<{
  'update:modelValue': [value: T]
}>()

const normalizedOptions = computed<SettingOption[]>(() => {
  if (!props.options) return []

  return props.options.map((option) => {
    if (typeof option === 'string') {
      return { text: option, value: option }
    }

    if ('text' in option) {
      return {
        text: option.text,
        value: option.value ?? option.text
      }
    }
    // Handle optionLabel/optionValue
    return {
      text: option[props.optionLabel || 'text'] || 'Unknown',
      value: option[props.optionValue || 'value']
    }
  })
})
</script>
