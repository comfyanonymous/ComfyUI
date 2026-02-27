<template>
  <div v-if="count > 1" class="flex items-center gap-1">
    <span class="mr-1 text-muted-foreground">
      <slot name="label" />
    </span>
    <Button
      variant="muted-textonly"
      size="icon-sm"
      :disabled="index === 0"
      data-testid="batch-prev"
      @click="index--"
    >
      <i class="icon-[lucide--chevron-left]" />
    </Button>
    <span data-testid="batch-counter">
      {{ $t('batch.index', { current: index + 1, total: count }) }}
    </span>
    <Button
      variant="muted-textonly"
      size="icon-sm"
      :disabled="index === count - 1"
      data-testid="batch-next"
      @click="index++"
    >
      <i class="icon-[lucide--chevron-right]" />
    </Button>
  </div>
</template>

<script setup lang="ts">
import Button from '@/components/ui/button/Button.vue'

const index = defineModel<number>({ required: true })

defineProps<{
  count: number
}>()
</script>
