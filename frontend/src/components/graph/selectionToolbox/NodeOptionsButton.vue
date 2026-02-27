<template>
  <Button
    v-tooltip.top="{
      value: $t('g.moreOptions'),
      showDelay: 1000
    }"
    data-testid="more-options-button"
    variant="muted-textonly"
    :aria-label="$t('g.moreOptions')"
    @click="handleClick"
  >
    <i class="icon-[lucide--more-vertical]" />
  </Button>
</template>

<script setup lang="ts">
import Button from '@/components/ui/button/Button.vue'
import { toggleNodeOptions } from '@/composables/graph/useMoreOptionsMenu'

const handleClick = (event: Event) => {
  toggleNodeOptions(event)
}
</script>
