<template>
  <Button
    v-show="isSingleImageNode"
    v-tooltip.top="{
      value: $t('commands.Comfy_MaskEditor_OpenMaskEditor.label'),
      showDelay: 1000
    }"
    variant="muted-textonly"
    :aria-label="$t('commands.Comfy_MaskEditor_OpenMaskEditor.label')"
    @click="openMaskEditor"
  >
    <i class="icon-[comfy--mask]" />
  </Button>
</template>

<script setup lang="ts">
import Button from '@/components/ui/button/Button.vue'
import { useSelectionState } from '@/composables/graph/useSelectionState'
import { useCommandStore } from '@/stores/commandStore'

const commandStore = useCommandStore()
const { isSingleImageNode } = useSelectionState()

const openMaskEditor = () => {
  void commandStore.execute('Comfy.MaskEditor.OpenMaskEditor')
}
</script>
