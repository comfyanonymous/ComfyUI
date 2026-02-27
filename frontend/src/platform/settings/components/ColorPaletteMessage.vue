<template>
  <Message severity="info" icon="pi pi-palette" pt:text="w-full">
    <div class="flex items-center justify-between">
      <div>
        {{ $t('settingsCategories.ColorPalette') }}
      </div>
      <div class="actions">
        <Select
          v-model="activePaletteId"
          class="w-44"
          :options="palettes"
          option-label="name"
          option-value="id"
        />
        <Button
          size="icon"
          variant="textonly"
          :title="$t('g.export')"
          @click="colorPaletteService.exportColorPalette(activePaletteId)"
        >
          <i class="pi pi-file-export" />
        </Button>
        <Button
          size="icon"
          variant="textonly"
          :title="$t('g.import')"
          @click="importCustomPalette"
        >
          <i class="pi pi-file-import" />
        </Button>
        <Button
          size="icon"
          variant="destructive-textonly"
          :title="$t('g.delete')"
          :disabled="!colorPaletteStore.isCustomPalette(activePaletteId)"
          @click="colorPaletteService.deleteCustomColorPalette(activePaletteId)"
        >
          <i class="pi pi-trash" />
        </Button>
      </div>
    </div>
  </Message>
</template>

<script setup lang="ts">
import { storeToRefs } from 'pinia'
import Message from 'primevue/message'
import Select from 'primevue/select'

import Button from '@/components/ui/button/Button.vue'
import { useSettingStore } from '@/platform/settings/settingStore'
import { useColorPaletteService } from '@/services/colorPaletteService'
import { useColorPaletteStore } from '@/stores/workspace/colorPaletteStore'

const settingStore = useSettingStore()
const colorPaletteStore = useColorPaletteStore()
const colorPaletteService = useColorPaletteService()
const { palettes, activePaletteId } = storeToRefs(colorPaletteStore)

const importCustomPalette = async () => {
  const palette = await colorPaletteService.importColorPalette()
  if (palette) {
    await settingStore.set('Comfy.ColorPalette', palette.id)
  }
}
</script>
