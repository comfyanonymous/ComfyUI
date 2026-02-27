<template>
  <div class="space-y-4">
    <label>{{ $t('load3d.lightIntensity') }}</label>

    <Slider
      v-model="lightIntensity"
      class="w-full"
      :min="lightIntensityMinimum"
      :max="lightIntensityMaximum"
      :step="lightAdjustmentIncrement"
    />
  </div>
</template>

<script setup lang="ts">
import Slider from 'primevue/slider'

import { useSettingStore } from '@/platform/settings/settingStore'

const lightIntensity = defineModel<number>('lightIntensity')

const lightIntensityMaximum = useSettingStore().get(
  'Comfy.Load3D.LightIntensityMaximum'
)
const lightIntensityMinimum = useSettingStore().get(
  'Comfy.Load3D.LightIntensityMinimum'
)
const lightAdjustmentIncrement = useSettingStore().get(
  'Comfy.Load3D.LightAdjustmentIncrement'
)
</script>
