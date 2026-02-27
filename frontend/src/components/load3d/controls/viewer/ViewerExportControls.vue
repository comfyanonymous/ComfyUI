<template>
  <div class="space-y-4">
    <Select
      v-model="exportFormat"
      :options="exportFormats"
      option-label="label"
      option-value="value"
    >
    </Select>

    <Button
      variant="muted-textonly"
      class="rounded-full"
      @click="exportModel(exportFormat)"
    >
      {{ $t('load3d.export') }}
    </Button>
  </div>
</template>

<script setup lang="ts">
import Select from 'primevue/select'
import { ref } from 'vue'

import Button from '@/components/ui/button/Button.vue'

const emit = defineEmits<{
  (e: 'exportModel', format: string): void
}>()

const exportFormats = [
  { label: 'GLB', value: 'glb' },
  { label: 'OBJ', value: 'obj' },
  { label: 'STL', value: 'stl' }
]

const exportFormat = ref('obj')

const exportModel = (format: string) => {
  emit('exportModel', format)
}
</script>
