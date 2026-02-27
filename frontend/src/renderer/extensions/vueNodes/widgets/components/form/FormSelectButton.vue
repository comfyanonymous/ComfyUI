<template>
  <div
    :class="
      cn(WidgetInputBaseClass, 'w-full p-1 flex min-w-0 items-center gap-1')
    "
  >
    <button
      v-for="(option, index) in options"
      :key="getOptionValue(option, index)"
      :class="
        cn(
          'flex-1 min-w-0 h-6 px-5 py-[5px] rounded flex justify-center items-center gap-1 transition-all duration-150 ease-in-out truncate',
          'bg-transparent border-none',
          'text-center text-xs font-normal',
          {
            'bg-interface-menu-component-surface-selected':
              isSelected(index) && !disabled,
            'hover:bg-interface-menu-component-surface-selected/50':
              !isSelected(index) && !disabled,
            'opacity-50 cursor-not-allowed': disabled,
            'cursor-pointer': !disabled
          },
          isSelected(index) && !disabled
            ? 'text-text-primary'
            : 'text-text-secondary'
        )
      "
      :disabled="disabled"
      @click="handleSelect(index)"
    >
      {{ getOptionLabel(option) }}
    </button>
  </div>
</template>

<script
  setup
  lang="ts"
  generic="
    T extends string | number | { label: string; value: string | number }
  "
>
import { cn } from '@/utils/tailwindUtil'

import { WidgetInputBaseClass } from '../layout'

type ModelValue = T extends object ? T['value'] : T

interface Props {
  modelValue: ModelValue | null | undefined
  options: T[]
  optionLabel?: string // PrimeVue compatible prop
  optionValue?: string // PrimeVue compatible prop
  disabled?: boolean
}

interface Emits {
  'update:modelValue': [value: ModelValue]
}

const {
  modelValue,
  options,
  optionLabel = 'label',
  optionValue = 'value',
  disabled = false
} = defineProps<Props>()

const emit = defineEmits<Emits>()

// handle both string/number arrays and object arrays with PrimeVue compatibility
const getOptionValue = (option: T, index: number): ModelValue => {
  if (typeof option !== 'object') {
    return option as ModelValue
  }

  const valueField = optionValue
  const optionRecord = option as Record<string, unknown>
  const value =
    optionRecord[valueField] ??
    option.value ??
    optionRecord.name ??
    option.label ??
    index
  return value as ModelValue
}

// for display with PrimeVue compatibility
const getOptionLabel = (option: T): string => {
  if (typeof option === 'object' && option !== null) {
    const labelField = optionLabel
    const optionRecord = option as Record<string, unknown>
    return String(
      optionRecord[labelField] ??
        option.label ??
        optionRecord.name ??
        option.value ??
        option
    )
  }
  return String(option)
}

const isSelected = (index: number): boolean => {
  const optVal = getOptionValue(options[index], index)
  return String(optVal) === String(modelValue ?? '')
}

const handleSelect = (index: number) => {
  if (disabled) return

  const optVal = getOptionValue(options[index], index)
  emit('update:modelValue', optVal)
}
</script>
