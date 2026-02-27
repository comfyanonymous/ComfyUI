<template>
  <div class="relative rounded-lg bg-backdrop/30">
    <div class="flex flex-col gap-2">
      <Button
        v-tooltip.right="{
          value: t('load3d.openIn3DViewer'),
          showDelay: 300
        }"
        size="icon"
        variant="textonly"
        class="rounded-full"
        :aria-label="t('load3d.openIn3DViewer')"
        @click="openIn3DViewer"
      >
        <i class="pi pi-expand text-lg text-base-foreground" />
      </Button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { useI18n } from 'vue-i18n'

import Load3DViewerContent from '@/components/load3d/Load3dViewerContent.vue'
import Button from '@/components/ui/button/Button.vue'
import type { LGraphNode } from '@/lib/litegraph/src/LGraphNode'
import { useLoad3dService } from '@/services/load3dService'
import { useDialogStore } from '@/stores/dialogStore'

const { t } = useI18n()
const { node } = defineProps<{
  node: LGraphNode
}>()

const openIn3DViewer = () => {
  const props = { node: node }

  useDialogStore().showDialog({
    key: 'global-load3d-viewer',
    title: t('load3d.viewer.title'),
    component: Load3DViewerContent,
    props: props,
    dialogComponentProps: {
      style: 'width: 80vw; height: 80vh;',
      maximizable: true,
      onClose: async () => {
        await useLoad3dService().handleViewerClose(props.node)
      }
    }
  })
}
</script>

<style scoped></style>
