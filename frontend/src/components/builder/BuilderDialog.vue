<template>
  <div class="flex w-full min-w-96 flex-col rounded-2xl bg-base-background">
    <!-- Header -->
    <div
      class="flex h-12 items-center justify-between border-b border-border-default px-4"
    >
      <div class="flex items-center gap-2">
        <slot name="header-icon" />
        <h2 class="m-0 text-sm font-normal text-base-foreground">
          <slot name="title" />
        </h2>
      </div>
      <Button
        variant="muted-textonly"
        class="-mr-1"
        :aria-label="$t('g.close')"
        @click="$emit('close')"
      >
        <i class="pi pi-times size-4" />
      </Button>
    </div>

    <!-- Body -->
    <div class="flex flex-col gap-4 px-4 py-4">
      <slot />
    </div>

    <!-- Footer -->
    <div class="flex items-center justify-end gap-4 px-4 py-4">
      <slot name="footer" />
    </div>
  </div>
</template>

<script setup lang="ts">
import Button from '@/components/ui/button/Button.vue'

defineEmits<{
  close: []
}>()
</script>
