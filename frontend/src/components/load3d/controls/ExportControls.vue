<template>
  <div class="flex flex-col">
    <div class="show-export-formats relative">
      <Button
        v-tooltip.right="{
          value: $t('load3d.exportModel'),
          showDelay: 300
        }"
        size="icon"
        variant="textonly"
        class="rounded-full"
        :aria-label="$t('load3d.exportModel')"
        @click="toggleExportFormats"
      >
        <i class="pi pi-download text-lg text-base-foreground" />
      </Button>
      <div
        v-show="showExportFormats"
        class="absolute top-0 left-12 rounded-lg bg-interface-menu-surface shadow-lg"
      >
        <div class="flex flex-col">
          <Button
            v-for="format in exportFormats"
            :key="format.value"
            variant="textonly"
            class="text-base-foreground"
            @click="exportModel(format.value)"
          >
            {{ format.label }}
          </Button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, onUnmounted, ref } from 'vue'

import Button from '@/components/ui/button/Button.vue'

const emit = defineEmits<{
  (e: 'exportModel', format: string): void
}>()

const showExportFormats = ref(false)

const exportFormats = [
  { label: 'GLB', value: 'glb' },
  { label: 'OBJ', value: 'obj' },
  { label: 'STL', value: 'stl' }
]

function toggleExportFormats() {
  showExportFormats.value = !showExportFormats.value
}

function exportModel(format: string) {
  emit('exportModel', format)

  showExportFormats.value = false
}

function closeExportFormatsList(e: MouseEvent) {
  const target = e.target as HTMLElement

  if (!target.closest('.show-export-formats')) {
    showExportFormats.value = false
  }
}

onMounted(() => {
  document.addEventListener('click', closeExportFormatsList)
})

onUnmounted(() => {
  document.removeEventListener('click', closeExportFormatsList)
})
</script>
