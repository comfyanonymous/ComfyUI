<template>
  <Button
    v-tooltip.top="{
      value: $t('g.info'),
      showDelay: 1000
    }"
    data-testid="info-button"
    variant="muted-textonly"
    :aria-label="$t('g.info')"
    @click="onInfoClick"
  >
    <i class="icon-[lucide--info]" />
  </Button>
</template>

<script setup lang="ts">
import Button from '@/components/ui/button/Button.vue'
import { useTelemetry } from '@/platform/telemetry'
import { useRightSidePanelStore } from '@/stores/workspace/rightSidePanelStore'

const rightSidePanelStore = useRightSidePanelStore()

/**
 * Track node info button click and toggle node help.
 */
const onInfoClick = () => {
  useTelemetry()?.trackUiButtonClicked({
    button_id: 'selection_toolbox_node_info_opened'
  })
  rightSidePanelStore.openPanel('info')
}
</script>
