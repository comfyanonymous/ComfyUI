<template>
  <div
    class="flex items-center justify-between p-2 rounded-lg cursor-pointer transition-all duration-200"
    :class="[
      selected
        ? 'bg-secondary-background border-2 border-border-default'
        : 'bg-component-node-disabled hover:bg-secondary-background border-2 border-transparent'
    ]"
    @click="$emit('select')"
  >
    <span class="text-base font-bold text-base-foreground">
      {{ formattedCredits }}
    </span>
    <span class="text-sm font-normal text-muted-foreground">
      {{ description }}
    </span>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'

import { formatCredits } from '@/base/credits/comfyCredits'

const { credits, description, selected } = defineProps<{
  credits: number
  description: string
  selected: boolean
}>()

defineEmits<{
  select: []
}>()

const { locale } = useI18n()

const formattedCredits = computed(() => {
  return formatCredits({
    value: credits,
    locale: locale.value,
    numberOptions: { minimumFractionDigits: 0, maximumFractionDigits: 0 }
  })
})
</script>
