<template>
  <div :style="gridStyle">
    <PackCardSkeleton v-for="n in skeletonCardCount" :key="n" />
  </div>
</template>

<script setup lang="ts">
import PackCardSkeleton from '@/workbench/extensions/manager/components/manager/skeleton/PackCardSkeleton.vue'

const { skeletonCardCount = 12, gridStyle } = defineProps<{
  skeletonCardCount?: number
  gridStyle: {
    display: string
    gridTemplateColumns: string
    padding: string
    gap: string
  }
}>()
</script>
