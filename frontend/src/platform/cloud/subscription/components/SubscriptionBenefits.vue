<template>
  <div class="flex flex-col items-start gap-0 self-stretch">
    <div class="flex items-center gap-2 py-2">
      <i class="pi pi-check text-xs text-text-primary" />
      <span class="text-sm text-text-primary">
        {{
          isFreeTier
            ? $t('subscription.benefits.benefit1FreeTier')
            : $t('subscription.benefits.benefit1')
        }}
      </span>
    </div>

    <div class="flex items-center gap-2 py-2">
      <i class="pi pi-check text-xs text-text-primary" />
      <span class="text-sm text-text-primary">
        {{ $t('subscription.benefits.benefit2') }}
      </span>
    </div>

    <div class="flex items-center gap-2 py-2">
      <i class="pi pi-check text-xs text-text-primary" />
      <span class="text-sm text-text-primary">
        {{ $t('subscription.benefits.benefit3') }}
      </span>
    </div>
  </div>
</template>

<script setup lang="ts">
const { isFreeTier = false } = defineProps<{
  isFreeTier?: boolean
}>()
</script>
