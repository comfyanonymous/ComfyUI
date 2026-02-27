<template>
  <div class="input-knob flex flex-row items-center gap-2">
    <Knob
      :model-value="modelValue"
      :value-template="displayValue"
      class="knob-part"
      :class="knobClass"
      :min="min"
      :max="max"
      :step="step"
      v-bind="$attrs"
      @update:model-value="updateValue"
    />
    <InputNumber
      :model-value="modelValue"
      class="input-part"
      :max-fraction-digits="3"
      :class="inputClass"
      :min="min"
      :max="max"
      :step="step"
      :allow-empty="false"
      @update:model-value="updateValue"
    />
  </div>
</template>

<script setup lang="ts">
import InputNumber from 'primevue/inputnumber'
import Knob from 'primevue/knob'
import { ref, watch } from 'vue'

const props = defineProps<{
  modelValue: number
  inputClass?: string
  knobClass?: string
  min?: number
  max?: number
  step?: number
  resolution?: number
}>()

const emit = defineEmits<{
  (e: 'update:modelValue', value: number): void
}>()

const localValue = ref(props.modelValue)

watch(
  () => props.modelValue,
  (newValue) => {
    localValue.value = newValue
  }
)

const updateValue = (newValue: number | null) => {
  if (newValue === null) {
    // If the input is cleared, reset to the minimum value or 0
    newValue = Number(props.min) || 0
  }

  const min = Number(props.min ?? Number.NEGATIVE_INFINITY)
  const max = Number(props.max ?? Number.POSITIVE_INFINITY)
  const step = Number(props.step) || 1

  // Ensure the value is within the allowed range
  newValue = Math.max(min, Math.min(max, newValue))

  // Round to the nearest step
  newValue = Math.round(newValue / step) * step

  // Update local value and emit change
  localValue.value = newValue
  emit('update:modelValue', newValue)
}

const displayValue = (value: number): string => {
  updateValue(value)
  const stepString = (props.step ?? 1).toString()
  const resolution = stepString.includes('.')
    ? stepString.split('.')[1].length
    : 0
  return value.toFixed(props.resolution ?? resolution)
}

defineOptions({
  inheritAttrs: false
})
</script>
