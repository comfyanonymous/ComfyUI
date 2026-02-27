<template>
  <div class="flex flex-col gap-3 pb-3">
    <h3 class="text-center text-[15px] font-sans text-descrip-text mt-2.5">
      {{ t('maskEditor.brushSettings') }}
    </h3>

    <button :class="textButtonClass" @click="resetToDefault">
      {{ t('maskEditor.resetToDefault') }}
    </button>

    <!-- Brush Shape -->
    <div class="flex flex-col gap-3 pb-3">
      <span class="text-left text-xs font-sans text-descrip-text">
        {{ t('maskEditor.brushShape') }}
      </span>

      <div
        class="flex flex-row gap-2.5 items-center h-[50px] w-full rounded-[10px] bg-secondary-background-hover"
      >
        <div
          class="maskEditor_sidePanelBrushShapeCircle hover:bg-comfy-menu-bg"
          :class="
            cn(
              store.brushSettings.type === BrushShape.Arc
                ? 'bg-[var(--p-button-text-primary-color)] active'
                : 'bg-transparent'
            )
          "
          @click="setBrushShape(BrushShape.Arc)"
        ></div>

        <div
          class="maskEditor_sidePanelBrushShapeSquare hover:bg-comfy-menu-bg"
          :class="
            cn(
              store.brushSettings.type === BrushShape.Rect
                ? 'bg-[var(--p-button-text-primary-color)] active'
                : 'bg-transparent'
            )
          "
          @click="setBrushShape(BrushShape.Rect)"
        ></div>
      </div>
    </div>

    <!-- Color -->
    <div class="flex flex-col gap-3 pb-3">
      <span class="text-left text-xs font-sans text-descrip-text">
        {{ t('maskEditor.colorSelector') }}
      </span>
      <input
        ref="colorInputRef"
        v-model="store.rgbColor"
        type="color"
        class="h-10 rounded-md cursor-pointer"
      />
    </div>

    <!-- Thickness -->
    <div class="flex flex-col gap-2">
      <div class="flex items-center justify-between">
        <span class="text-left text-xs font-sans text-descrip-text">
          {{ t('maskEditor.thickness') }}
        </span>
        <input
          v-model.number="brushSize"
          type="number"
          class="w-16 px-2 py-1 text-sm text-center border rounded-md bg-comfy-menu-bg border-p-form-field-border-color text-input-text"
          :min="1"
          :max="250"
          :step="1"
        />
      </div>
      <SliderControl
        v-model="brushSize"
        class="flex-1"
        label=""
        :min="1"
        :max="250"
        :step="1"
      />
    </div>

    <!-- Opacity -->
    <div class="flex flex-col gap-2">
      <div class="flex items-center justify-between">
        <span class="text-left text-xs font-sans text-descrip-text">
          {{ t('maskEditor.opacity') }}
        </span>
        <input
          v-model.number="brushOpacity"
          type="number"
          class="w-16 px-2 py-1 text-sm text-center border rounded-md bg-comfy-menu-bg border-p-form-field-border-color text-input-text"
          :min="0"
          :max="1"
          :step="0.01"
        />
      </div>
      <SliderControl
        v-model="brushOpacity"
        class="flex-1"
        label=""
        :min="0"
        :max="1"
        :step="0.01"
      />
    </div>

    <!-- Hardness -->
    <div class="flex flex-col gap-2">
      <div class="flex items-center justify-between">
        <span class="text-left text-xs font-sans text-descrip-text">
          {{ t('maskEditor.hardness') }}
        </span>
        <input
          v-model.number="brushHardness"
          type="number"
          class="w-16 px-2 py-1 text-sm text-center border rounded-md bg-comfy-menu-bg border-p-form-field-border-color text-input-text"
          :min="0"
          :max="1"
          :step="0.01"
        />
      </div>
      <SliderControl
        v-model="brushHardness"
        class="flex-1"
        label=""
        :min="0"
        :max="1"
        :step="0.01"
      />
    </div>

    <!-- Step Size -->
    <div class="flex flex-col gap-2">
      <div class="flex items-center justify-between">
        <span class="text-left text-xs font-sans text-descrip-text">
          {{ t('maskEditor.stepSize') }}
        </span>
        <input
          v-model.number="brushStepSize"
          type="number"
          class="w-16 px-2 py-1 text-sm text-center border rounded-md bg-comfy-menu-bg border-p-form-field-border-color text-input-text"
          :min="1"
          :max="100"
          :step="1"
        />
      </div>
      <SliderControl
        v-model="brushStepSize"
        class="flex-1"
        label=""
        :min="1"
        :max="100"
        :step="1"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'

import { BrushShape } from '@/extensions/core/maskeditor/types'
import { useMaskEditorStore } from '@/stores/maskEditorStore'
import { cn } from '@/utils/tailwindUtil'

import SliderControl from './controls/SliderControl.vue'

const { t } = useI18n()
const store = useMaskEditorStore()

const colorInputRef = ref<HTMLInputElement>()

const textButtonClass =
  'h-7.5 w-32 rounded-[10px] border border-p-form-field-border-color text-input-text font-sans transition-colors duration-100 bg-comfy-menu-bg hover:bg-secondary-background-hover'

/* Computed properties that use store setters for validation */
const brushSize = computed({
  get: () => store.brushSettings.size,
  set: (value: number) => store.setBrushSize(value)
})

const brushOpacity = computed({
  get: () => store.brushSettings.opacity,
  set: (value: number) => store.setBrushOpacity(value)
})

const brushHardness = computed({
  get: () => store.brushSettings.hardness,
  set: (value: number) => store.setBrushHardness(value)
})

const brushStepSize = computed({
  get: () => store.brushSettings.stepSize,
  set: (value: number) => store.setBrushStepSize(value)
})

/* Brush shape */
const setBrushShape = (shape: BrushShape) => {
  store.brushSettings.type = shape
}

/* Reset */
const resetToDefault = () => {
  store.resetBrushToDefault()
}

onMounted(() => {
  if (colorInputRef.value) {
    store.colorInput = colorInputRef.value
  }
})

onBeforeUnmount(() => {
  store.colorInput = null
})
</script>
