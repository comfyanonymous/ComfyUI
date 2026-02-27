<script setup lang="ts">
import { statusBadgeVariants } from './statusBadge.variants'
import type { StatusBadgeVariants } from './statusBadge.variants'

const {
  label,
  severity = 'default',
  variant
} = defineProps<{
  label?: string | number
  severity?: StatusBadgeVariants['severity']
  variant?: StatusBadgeVariants['variant']
}>()
</script>

<template>
  <span
    :class="
      statusBadgeVariants({
        severity,
        variant: variant ?? (label == null ? 'dot' : 'label')
      })
    "
  >
    {{ label }}
  </span>
</template>
