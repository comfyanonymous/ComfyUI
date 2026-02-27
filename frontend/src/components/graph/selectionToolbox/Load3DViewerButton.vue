<template>
  <Button
    v-tooltip.top="{
      value: $t('commands.Comfy_3DViewer_Open3DViewer.label'),
      showDelay: 1000
    }"
    variant="muted-textonly"
    :aria-label="$t('commands.Comfy_3DViewer_Open3DViewer.label')"
    @click="open3DViewer"
  >
    <i class="icon-[lucide--pencil]" />
  </Button>
</template>

<script setup lang="ts">
import Button from '@/components/ui/button/Button.vue'
import { useCommandStore } from '@/stores/commandStore'

const commandStore = useCommandStore()

const open3DViewer = () => {
  void commandStore.execute('Comfy.3DViewer.Open3DViewer')
}
</script>
