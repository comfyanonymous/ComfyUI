<template>
  <div class="flex flex-col">
    <div v-if="showLightIntensityButton" class="show-light-intensity relative">
      <Button
        v-tooltip.right="{
          value: $t('load3d.lightIntensity'),
          showDelay: 300
        }"
        size="icon"
        variant="textonly"
        class="rounded-full"
        :aria-label="$t('load3d.lightIntensity')"
        @click="toggleLightIntensity"
      >
        <i class="pi pi-sun text-lg text-base-foreground" />
      </Button>
      <div
        v-show="showLightIntensity"
        class="absolute top-0 left-12 rounded-lg bg-black/50 p-4 shadow-lg"
        style="width: 150px"
      >
        <Slider
          v-model="lightIntensity"
          class="w-full"
          :min="lightIntensityMinimum"
          :max="lightIntensityMaximum"
          :step="lightAdjustmentIncrement"
        />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import Slider from 'primevue/slider'
import { computed, onMounted, onUnmounted, ref } from 'vue'

import Button from '@/components/ui/button/Button.vue'
import type { MaterialMode } from '@/extensions/core/load3d/interfaces'
import { useSettingStore } from '@/platform/settings/settingStore'

const lightIntensity = defineModel<number>('lightIntensity')
const materialMode = defineModel<MaterialMode>('materialMode')

const showLightIntensityButton = computed(
  () => materialMode.value === 'original'
)
const showLightIntensity = ref(false)

const lightIntensityMaximum = useSettingStore().get(
  'Comfy.Load3D.LightIntensityMaximum'
)
const lightIntensityMinimum = useSettingStore().get(
  'Comfy.Load3D.LightIntensityMinimum'
)
const lightAdjustmentIncrement = useSettingStore().get(
  'Comfy.Load3D.LightAdjustmentIncrement'
)

function toggleLightIntensity() {
  showLightIntensity.value = !showLightIntensity.value
}

function closeLightSlider(e: MouseEvent) {
  const target = e.target as HTMLElement

  if (!target.closest('.show-light-intensity')) {
    showLightIntensity.value = false
  }
}

onMounted(() => {
  document.addEventListener('click', closeLightSlider)
})

onUnmounted(() => {
  document.removeEventListener('click', closeLightSlider)
})
</script>
