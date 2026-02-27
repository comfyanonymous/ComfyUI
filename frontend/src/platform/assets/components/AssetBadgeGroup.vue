<template>
  <div
    class="absolute left-2 bottom-2 flex flex-wrap justify-start gap-1 select-none"
  >
    <span
      v-for="badge in badges"
      :key="badge.label"
      :class="
        cn(
          'px-2 py-1 rounded text-xs font-bold uppercase tracking-wider text-modal-card-tag-foreground bg-modal-card-tag-background break-all'
        )
      "
    >
      {{ badge.label }}
    </span>
  </div>
</template>

<script setup lang="ts">
import { cn } from '@/utils/tailwindUtil'

type AssetBadge = {
  label: string
  type: 'type' | 'base' | 'size'
}

defineProps<{
  badges: AssetBadge[]
}>()
</script>
