<script setup lang="ts">
import {
  DropdownMenuContent,
  DropdownMenuPortal,
  DropdownMenuRoot,
  DropdownMenuTrigger
} from 'reka-ui'
import { useI18n } from 'vue-i18n'

import WorkflowActionsList from '@/components/common/WorkflowActionsList.vue'
import Button from '@/components/ui/button/Button.vue'
import { useWorkflowActionsMenu } from '@/composables/useWorkflowActionsMenu'
import { useTelemetry } from '@/platform/telemetry'
import { useCanvasStore } from '@/renderer/core/canvas/canvasStore'
import { useCommandStore } from '@/stores/commandStore'

const { source, align = 'start' } = defineProps<{
  source: string
  align?: 'start' | 'center' | 'end'
}>()

const { t } = useI18n()
const canvasStore = useCanvasStore()

const { menuItems } = useWorkflowActionsMenu(
  () => useCommandStore().execute('Comfy.RenameWorkflow'),
  { isRoot: true }
)

function handleOpen(open: boolean) {
  if (open) {
    useTelemetry()?.trackUiButtonClicked({
      button_id: source
    })
  }
}
</script>

<template>
  <DropdownMenuRoot @update:open="handleOpen">
    <DropdownMenuTrigger as-child>
      <slot name="button">
        <Button
          v-tooltip="{
            value: t('breadcrumbsMenu.workflowActions'),
            showDelay: 300,
            hideDelay: 300
          }"
          variant="secondary"
          size="unset"
          :aria-label="t('breadcrumbsMenu.workflowActions')"
          class="h-10 rounded-lg pl-3 pr-2 pointer-events-auto gap-1 data-[state=open]:bg-secondary-background-hover data-[state=open]:shadow-interface"
        >
          <i
            class="size-4"
            :class="
              canvasStore.linearMode
                ? 'icon-[lucide--panels-top-left]'
                : 'icon-[comfy--workflow]'
            "
          />
          <i class="icon-[lucide--chevron-down] size-4 text-muted-foreground" />
        </Button>
      </slot>
    </DropdownMenuTrigger>
    <DropdownMenuPortal>
      <DropdownMenuContent
        :align
        :side-offset="5"
        :collision-padding="10"
        class="z-1000 rounded-lg px-2 py-3 min-w-56 bg-base-background shadow-interface border border-border-subtle"
      >
        <WorkflowActionsList :items="menuItems" />
      </DropdownMenuContent>
    </DropdownMenuPortal>
  </DropdownMenuRoot>
</template>
