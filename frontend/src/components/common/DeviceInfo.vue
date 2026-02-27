<template>
  <div class="grid grid-cols-2 gap-2">
    <template v-for="col in deviceColumns" :key="col.field">
      <div class="font-medium">
        {{ col.header }}
      </div>
      <div>
        {{ formatValue(props.device[col.field], col.field) }}
      </div>
    </template>
  </div>
</template>

<script setup lang="ts">
import type { DeviceStats } from '@/schemas/apiSchema'
import { formatSize } from '@/utils/formatUtil'

const props = defineProps<{
  device: DeviceStats
}>()

const deviceColumns: { field: keyof DeviceStats; header: string }[] = [
  { field: 'name', header: 'Name' },
  { field: 'type', header: 'Type' },
  { field: 'vram_total', header: 'VRAM Total' },
  { field: 'vram_free', header: 'VRAM Free' },
  { field: 'torch_vram_total', header: 'Torch VRAM Total' },
  { field: 'torch_vram_free', header: 'Torch VRAM Free' }
]

const formatValue = (value: string | number, field: string) => {
  if (
    ['vram_total', 'vram_free', 'torch_vram_total', 'torch_vram_free'].includes(
      field
    )
  ) {
    const num = Number(value)
    if (Number.isFinite(num)) {
      return formatSize(num)
    }
    return value
  }
  return value
}
</script>
