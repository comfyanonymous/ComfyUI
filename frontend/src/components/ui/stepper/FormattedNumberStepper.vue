<template>
  <label
    :for="inputId"
    :class="
      cn(
        'flex h-10 cursor-text items-center rounded-lg bg-secondary-background text-secondary-foreground hover:bg-secondary-background-hover focus-within:ring-1 focus-within:ring-secondary-foreground',
        disabled && 'opacity-50 pointer-events-none'
      )
    "
  >
    <button
      type="button"
      class="flex h-full w-8 cursor-pointer items-center justify-center rounded-l-lg border-none bg-transparent text-muted-foreground transition-colors hover:text-base-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-inset focus-visible:ring-secondary-foreground disabled:opacity-30"
      :disabled="disabled || modelValue <= min"
      :aria-label="$t('g.decrement')"
      @click="handleStep(-1)"
    >
      <i class="icon-[lucide--minus] size-4" />
    </button>
    <div
      class="flex flex-1 items-center justify-center gap-0.5 overflow-hidden"
    >
      <slot name="prefix" />
      <input
        :id="inputId"
        ref="inputRef"
        v-model="inputValue"
        type="text"
        inputmode="numeric"
        :style="{ width: `${inputWidth}ch` }"
        class="min-w-0 rounded border-none bg-transparent text-center text-base-foreground font-medium text-lg focus-visible:outline-none"
        :disabled="disabled"
        @input="handleInputChange"
        @blur="handleInputBlur"
        @focus="handleInputFocus"
      />
      <slot name="suffix" />
    </div>
    <button
      type="button"
      class="flex h-full w-8 cursor-pointer items-center justify-center rounded-r-lg border-none bg-transparent text-muted-foreground transition-colors hover:text-base-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-inset focus-visible:ring-secondary-foreground disabled:opacity-30"
      :disabled="disabled || modelValue >= max"
      :aria-label="$t('g.increment')"
      @click="handleStep(1)"
    >
      <i class="icon-[lucide--plus] size-4" />
    </button>
  </label>
</template>

<script setup lang="ts">
import { computed, ref, useId, watch } from 'vue'

import { cn } from '@/utils/tailwindUtil'

const {
  min = 0,
  max = Infinity,
  step = 1,
  formatOptions = { useGrouping: true },
  disabled = false
} = defineProps<{
  min?: number
  max?: number
  step?: number | ((value: number) => number)
  formatOptions?: Intl.NumberFormatOptions
  disabled?: boolean
}>()

const emit = defineEmits<{
  'max-reached': []
}>()

const modelValue = defineModel<number>({ required: true })

const inputId = useId()
const inputRef = ref<HTMLInputElement | null>(null)
const inputValue = ref(formatNumber(modelValue.value))

const inputWidth = computed(() =>
  Math.min(Math.max(inputValue.value.length, 1) + 0.5, 9)
)

watch(modelValue, (newValue) => {
  if (document.activeElement !== inputRef.value) {
    inputValue.value = formatNumber(newValue)
  }
})

function formatNumber(num: number): string {
  return num.toLocaleString('en-US', formatOptions)
}

function parseFormattedNumber(str: string): number {
  const cleaned = str.replace(/[^0-9]/g, '')
  return cleaned === '' ? 0 : parseInt(cleaned, 10)
}

function clamp(value: number, minVal: number, maxVal: number): number {
  return Math.min(Math.max(value, minVal), maxVal)
}

function formatWithCursor(
  value: string,
  cursorPos: number
): { formatted: string; newCursor: number } {
  const num = parseFormattedNumber(value)
  const formatted = formatNumber(num)

  const digitsBeforeCursor = value
    .slice(0, cursorPos)
    .replace(/[^0-9]/g, '').length

  let digitCount = 0
  let newCursor = 0
  for (let i = 0; i < formatted.length; i++) {
    if (/[0-9]/.test(formatted[i])) {
      digitCount++
    }
    if (digitCount >= digitsBeforeCursor) {
      newCursor = i + 1
      break
    }
  }

  if (digitCount < digitsBeforeCursor) {
    newCursor = formatted.length
  }

  return { formatted, newCursor }
}

function getStepAmount(): number {
  return typeof step === 'function' ? step(modelValue.value) : step
}

function handleInputChange(e: Event) {
  const input = e.target as HTMLInputElement
  const raw = input.value
  const cursorPos = input.selectionStart ?? raw.length
  const num = parseFormattedNumber(raw)

  const clamped = Math.min(num, max)
  const wasClamped = num > max

  if (wasClamped) {
    emit('max-reached')
  }

  modelValue.value = clamped

  const { formatted, newCursor } = formatWithCursor(
    wasClamped ? formatNumber(clamped) : raw,
    wasClamped ? formatNumber(clamped).length : cursorPos
  )
  inputValue.value = formatted

  requestAnimationFrame(() => {
    inputRef.value?.setSelectionRange(newCursor, newCursor)
  })
}

function handleInputBlur() {
  const clamped = clamp(modelValue.value, min, max)
  modelValue.value = clamped
  inputValue.value = formatNumber(clamped)
}

function handleInputFocus(e: FocusEvent) {
  const input = e.target as HTMLInputElement
  const len = input.value.length
  input.setSelectionRange(len, len)
}

function handleStep(direction: 1 | -1) {
  const stepAmount = getStepAmount()
  const newValue = clamp(modelValue.value + stepAmount * direction, min, max)
  modelValue.value = newValue
  inputValue.value = formatNumber(newValue)
}
</script>
