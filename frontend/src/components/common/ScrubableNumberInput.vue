<template>
  <div
    ref="container"
    class="flex h-7 rounded-lg bg-component-node-widget-background text-xs text-component-node-foreground"
  >
    <slot name="background" />
    <Button
      v-if="!hideButtons"
      :aria-label="t('g.decrement')"
      data-testid="decrement"
      class="h-full w-8 rounded-r-none hover:bg-base-foreground/20 disabled:opacity-30"
      variant="muted-textonly"
      :disabled="!canDecrement"
      tabindex="-1"
      @click="modelValue = clamp(modelValue - step)"
    >
      <i class="pi pi-minus" />
    </Button>
    <div class="relative min-w-[4ch] flex-1 py-1.5 my-0.25">
      <input
        ref="inputField"
        v-bind="inputAttrs"
        :value="displayValue ?? modelValue"
        :disabled
        :class="
          cn(
            'bg-transparent border-0 focus:outline-0 p-1 truncate text-sm absolute inset-0'
          )
        "
        inputmode="decimal"
        autocomplete="off"
        autocorrect="off"
        spellcheck="false"
        @blur="handleBlur"
        @keyup.enter="handleBlur"
        @dragstart.prevent
      />
      <div
        :class="
          cn(
            'absolute inset-0 z-10 cursor-ew-resize',
            textEdit && 'pointer-events-none hidden'
          )
        "
        @pointerdown="handlePointerDown"
        @pointermove="handlePointerMove"
        @pointerup="handlePointerUp"
        @pointercancel="resetDrag"
      />
    </div>
    <slot />
    <Button
      v-if="!hideButtons"
      :aria-label="t('g.increment')"
      data-testid="increment"
      class="h-full w-8 rounded-l-none hover:bg-base-foreground/20 disabled:opacity-30"
      variant="muted-textonly"
      :disabled="!canIncrement"
      tabindex="-1"
      @click="modelValue = clamp(modelValue + step)"
    >
      <i class="pi pi-plus" />
    </Button>
  </div>
</template>

<script setup lang="ts">
import { onClickOutside } from '@vueuse/core'
import { computed, ref, useTemplateRef } from 'vue'
import { useI18n } from 'vue-i18n'

import Button from '@/components/ui/button/Button.vue'
import { cn } from '@/utils/tailwindUtil'

const {
  min,
  max,
  step = 1,
  disabled = false,
  hideButtons = false,
  displayValue,
  parseValue
} = defineProps<{
  min?: number
  max?: number
  step?: number
  disabled?: boolean
  hideButtons?: boolean
  displayValue?: string
  parseValue?: (raw: string) => number | undefined
  inputAttrs?: Record<string, unknown>
}>()

const { t } = useI18n()
const modelValue = defineModel<number>({ default: 0 })

const container = useTemplateRef<HTMLDivElement>('container')
const inputField = useTemplateRef<HTMLInputElement>('inputField')
const textEdit = ref(false)

onClickOutside(container, () => {
  if (textEdit.value) textEdit.value = false
})

function clamp(value: number): number {
  const lo = min ?? -Infinity
  const hi = max ?? Infinity
  return Math.min(hi, Math.max(lo, value))
}

const canDecrement = computed(
  () => modelValue.value > (min ?? -Infinity) && !disabled
)
const canIncrement = computed(
  () => modelValue.value < (max ?? Infinity) && !disabled
)

const dragging = ref(false)
const dragDelta = ref(0)
const hasDragged = ref(false)

function handleBlur(e: Event) {
  const target = e.target as HTMLInputElement
  const raw = target.value.trim()
  const parsed = parseValue
    ? parseValue(raw)
    : raw === ''
      ? undefined
      : Number(raw)
  if (parsed != null && !isNaN(parsed)) {
    modelValue.value = clamp(parsed)
  } else {
    target.value = displayValue ?? String(modelValue.value)
  }
  textEdit.value = false
}

function handlePointerDown(e: PointerEvent) {
  if (e.button !== 0) return
  if (disabled) return
  const target = e.target as HTMLElement
  target.setPointerCapture(e.pointerId)
  dragging.value = true
  dragDelta.value = 0
  hasDragged.value = false
}

function handlePointerMove(e: PointerEvent) {
  if (!dragging.value) return
  dragDelta.value += e.movementX
  const steps = (dragDelta.value / 10) | 0
  if (steps === 0) return
  hasDragged.value = true
  const unclipped = modelValue.value + steps * step
  dragDelta.value %= 10
  modelValue.value = clamp(unclipped)
}

function handlePointerUp() {
  if (!dragging.value) return

  if (!hasDragged.value) {
    textEdit.value = true
    inputField.value?.focus()
    inputField.value?.select()
  }

  resetDrag()
}

function resetDrag() {
  dragging.value = false
  dragDelta.value = 0
}
</script>
