<template>
  <div class="relative show-slider">
    <Button
      v-tooltip.right="{ value: tooltipText, showDelay: 300 }"
      size="icon"
      variant="textonly"
      class="rounded-full"
      :aria-label="tooltipText"
      @click="toggleSlider"
    >
      <i :class="['pi', icon, 'text-lg text-base-foreground']" />
    </Button>
    <div
      v-show="showSlider"
      class="absolute top-0 left-12 rounded-lg bg-interface-menu-surface p-4 shadow-lg w-[150px]"
    >
      <Slider
        v-model="value"
        class="w-full"
        :min="min"
        :max="max"
        :step="step"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import Slider from 'primevue/slider'
import { onMounted, onUnmounted, ref } from 'vue'

import Button from '@/components/ui/button/Button.vue'

const {
  icon = 'pi-expand',
  min = 10,
  max = 150,
  step = 1
} = defineProps<{
  icon?: string
  tooltipText: string
  min?: number
  max?: number
  step?: number
}>()

const value = defineModel<number>()
const showSlider = ref(false)

const toggleSlider = () => {
  showSlider.value = !showSlider.value
}

const closeSlider = (e: MouseEvent) => {
  const target = e.target as HTMLElement

  if (!target.closest('.show-slider')) {
    showSlider.value = false
  }
}

onMounted(() => {
  document.addEventListener('click', closeSlider)
})

onUnmounted(() => {
  document.removeEventListener('click', closeSlider)
})
</script>
