<template>
  <div class="flex flex-col gap-3 pb-3">
    <h3
      class="text-center text-[15px] font-sans text-[var(--descrip-text)] mt-2.5"
    >
      {{ t('maskEditor.layers') }}
    </h3>

    <SliderControl
      :label="t('maskEditor.maskOpacity')"
      :min="0"
      :max="1"
      :step="0.01"
      :model-value="store.maskOpacity"
      @update:model-value="onMaskOpacityChange"
    />

    <span class="text-left text-xs font-sans text-[var(--descrip-text)]">{{
      t('maskEditor.maskBlendingOptions')
    }}</span>

    <div
      class="flex flex-row gap-2.5 items-center min-h-6 relative h-[50px] w-full rounded-[10px] -mt-2 -mb-1.5"
    >
      <select
        class="maskEditor_sidePanelDropdown"
        :value="store.maskBlendMode"
        @change="onBlendModeChange"
      >
        <option value="black">{{ t('maskEditor.black') }}</option>
        <option value="white">{{ t('maskEditor.white') }}</option>
        <option value="negative">{{ t('maskEditor.negative') }}</option>
      </select>
    </div>

    <span class="text-left text-xs font-sans text-[var(--descrip-text)]">{{
      t('maskEditor.maskLayer')
    }}</span>
    <div
      class="flex flex-row gap-2.5 items-center min-h-6 relative h-[50px] w-full rounded-[10px] bg-secondary-background-hover"
      :style="{
        border: store.activeLayer === 'mask' ? '2px solid #007acc' : 'none'
      }"
    >
      <input
        type="checkbox"
        class="maskEditor_sidePanelLayerCheckbox"
        :checked="maskLayerVisible"
        @change="onMaskLayerVisibilityChange"
      />
      <div class="maskEditor_sidePanelLayerPreviewContainer">
        <svg viewBox="0 0 20 20" style="">
          <path
            class="cls-1"
            d="M1.31,5.32v9.36c0,.55.45,1,1,1h15.38c.55,0,1-.45,1-1V5.32c0-.55-.45-1-1-1H2.31c-.55,0-1,.45-1,1ZM11.19,13.44c-2.91.94-5.57-1.72-4.63-4.63.34-1.05,1.19-1.9,2.24-2.24,2.91-.94,5.57,1.72,4.63,4.63-.34,1.05-1.19-1.9-2.24,2.24Z"
          />
        </svg>
      </div>
      <button
        style="font-size: 12px"
        :style="{ opacity: store.activeLayer === 'mask' ? '0.5' : '1' }"
        :disabled="store.activeLayer === 'mask'"
        @click="setActiveLayer('mask')"
      >
        {{ t('maskEditor.activateLayer') }}
      </button>
    </div>

    <span class="text-left text-xs font-sans text-[var(--descrip-text)]">{{
      t('maskEditor.paintLayer')
    }}</span>
    <div
      class="flex flex-row gap-2.5 items-center min-h-6 relative h-[50px] w-full rounded-[10px] bg-secondary-background-hover"
      :style="{
        border: store.activeLayer === 'rgb' ? '2px solid #007acc' : 'none'
      }"
    >
      <input
        type="checkbox"
        class="maskEditor_sidePanelLayerCheckbox"
        :checked="paintLayerVisible"
        @change="onPaintLayerVisibilityChange"
      />
      <div class="maskEditor_sidePanelLayerPreviewContainer">
        <svg viewBox="0 0 20 20">
          <path
            class="cls-1"
            d="M 17 6.965 c 0 0.235 -0.095 0.47 -0.275 0.655 l -6.51 6.52 c -0.045 0.035 -0.09 0.075 -0.135 0.11 c -0.035 -0.695 -0.605 -1.24 -1.305 -1.245 c 0.035 -0.06 0.08 -0.12 0.135 -0.17 l 6.52 -6.52 c 0.36 -0.36 0.945 -0.36 1.3 0 c 0.175 0.175 0.275 0.415 0.275 0.65 Z"
          />
          <path
            class="cls-1"
            d="M 9.82 14.515 c 0 2.23 -3.23 1.59 -4.82 0 c 1.65 -0.235 2.375 -1.29 3.53 -1.29 c 0.715 0 1.29 0.58 1.29 1.29 Z"
          />
        </svg>
      </div>
      <button
        style="font-size: 12px"
        :style="{
          opacity: store.activeLayer === 'rgb' ? '0.5' : '1',
          display: showLayerButtons ? 'block' : 'none'
        }"
        :disabled="store.activeLayer === 'rgb'"
        @click="setActiveLayer('rgb')"
      >
        {{ t('maskEditor.activateLayer') }}
      </button>
    </div>

    <span class="text-left text-xs font-sans text-[var(--descrip-text)]">{{
      t('maskEditor.baseImageLayer')
    }}</span>
    <div
      class="flex flex-row gap-2.5 items-center min-h-6 relative h-[50px] w-full rounded-[10px] bg-secondary-background-hover"
    >
      <input
        type="checkbox"
        class="maskEditor_sidePanelLayerCheckbox"
        :checked="baseImageLayerVisible"
        @change="onBaseImageLayerVisibilityChange"
      />
      <div class="maskEditor_sidePanelLayerPreviewContainer">
        <img
          class="maskEditor_sidePanelImageLayerImage"
          :src="baseImageSrc"
          :alt="t('maskEditor.baseLayerPreview')"
        />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'

import { useCanvasManager } from '@/composables/maskeditor/useCanvasManager'
import type { useToolManager } from '@/composables/maskeditor/useToolManager'
import type { ImageLayer } from '@/extensions/core/maskeditor/types'
import { MaskBlendMode, Tools } from '@/extensions/core/maskeditor/types'
import { useMaskEditorStore } from '@/stores/maskEditorStore'

import SliderControl from './controls/SliderControl.vue'

const { toolManager } = defineProps<{
  toolManager?: ReturnType<typeof useToolManager>
}>()

const { t } = useI18n()
const store = useMaskEditorStore()
const canvasManager = useCanvasManager()

const maskLayerVisible = ref(true)
const paintLayerVisible = ref(true)
const baseImageLayerVisible = ref(true)

const baseImageSrc = computed(() => {
  return store.image?.src ?? ''
})

const showLayerButtons = computed(() => {
  return store.currentTool === Tools.Eraser
})

const onMaskLayerVisibilityChange = (event: Event) => {
  const checked = (event.target as HTMLInputElement).checked
  maskLayerVisible.value = checked

  const maskCanvas = store.maskCanvas
  if (maskCanvas) {
    maskCanvas.style.opacity = checked ? String(store.maskOpacity) : '0'
  }
}

const onPaintLayerVisibilityChange = (event: Event) => {
  const checked = (event.target as HTMLInputElement).checked
  paintLayerVisible.value = checked

  const rgbCanvas = store.rgbCanvas
  if (rgbCanvas) {
    rgbCanvas.style.opacity = checked ? '1' : '0'
  }
}

const onBaseImageLayerVisibilityChange = (event: Event) => {
  const checked = (event.target as HTMLInputElement).checked
  baseImageLayerVisible.value = checked

  const imgCanvas = store.imgCanvas
  if (imgCanvas) {
    imgCanvas.style.opacity = checked ? '1' : '0'
  }
}

const onMaskOpacityChange = (value: number) => {
  store.setMaskOpacity(value)

  const maskCanvas = store.maskCanvas
  if (maskCanvas) {
    maskCanvas.style.opacity = String(value)
  }

  maskLayerVisible.value = value !== 0
}

const onBlendModeChange = async (event: Event) => {
  const value = (event.target as HTMLSelectElement).value
  let blendMode: MaskBlendMode

  switch (value) {
    case 'white':
      blendMode = MaskBlendMode.White
      break
    case 'negative':
      blendMode = MaskBlendMode.Negative
      break
    default:
      blendMode = MaskBlendMode.Black
  }

  store.maskBlendMode = blendMode

  await canvasManager.updateMaskColor()
}

const setActiveLayer = (layer: ImageLayer) => {
  toolManager?.setActiveLayer(layer)
}
</script>
