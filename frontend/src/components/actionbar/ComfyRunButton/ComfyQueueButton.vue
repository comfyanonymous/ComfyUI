<template>
  <div class="queue-button-group flex">
    <SplitButton
      v-tooltip.bottom="{
        value: queueButtonTooltip,
        showDelay: 600
      }"
      class="comfyui-queue-button"
      :label="queueButtonLabel"
      :severity="queueButtonSeverity"
      size="small"
      :model="queueModeMenuItems"
      data-testid="queue-button"
      @click="queuePrompt"
    >
      <template #icon>
        <i :class="iconClass" />
      </template>
      <template #item="{ item }">
        <Button
          v-tooltip="{
            value: item.tooltip,
            showDelay: 600
          }"
          :variant="item.key === selectedQueueMode ? 'primary' : 'secondary'"
          size="sm"
          class="w-full justify-start"
        >
          <i v-if="item.icon" :class="item.icon" />
          {{ String(item.label ?? '') }}
        </Button>
      </template>
    </SplitButton>
    <BatchCountEdit />
  </div>
</template>

<script setup lang="ts">
import { storeToRefs } from 'pinia'
import type { MenuItem } from 'primevue/menuitem'
import SplitButton from 'primevue/splitbutton'
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'

import Button from '@/components/ui/button/Button.vue'
import { isCloud } from '@/platform/distribution/types'
import { useTelemetry } from '@/platform/telemetry'
import { app } from '@/scripts/app'
import { useCommandStore } from '@/stores/commandStore'
import { useNodeDefStore } from '@/stores/nodeDefStore'
import {
  isInstantMode,
  isInstantRunningMode,
  useQueueSettingsStore
} from '@/stores/queueStore'
import { useWorkspaceStore } from '@/stores/workspaceStore'
import { graphHasMissingNodes } from '@/workbench/extensions/manager/utils/graphHasMissingNodes'

import BatchCountEdit from '../BatchCountEdit.vue'

const workspaceStore = useWorkspaceStore()
const { mode: queueMode, batchCount } = storeToRefs(useQueueSettingsStore())

const nodeDefStore = useNodeDefStore()
const hasMissingNodes = computed(() =>
  graphHasMissingNodes(app.rootGraph, nodeDefStore.nodeDefsByName)
)

const { t } = useI18n()
type QueueModeMenuKey = 'disabled' | 'change' | 'instant-idle'

const selectedQueueMode = computed<QueueModeMenuKey>(() =>
  isInstantMode(queueMode.value) ? 'instant-idle' : queueMode.value
)

const queueModeMenuItemLookup = computed(() => {
  const items: Record<string, MenuItem> = {
    disabled: {
      key: 'disabled',
      label: t('menu.run'),
      tooltip: t('menu.disabledTooltip'),
      command: () => {
        queueMode.value = 'disabled'
      }
    },
    change: {
      key: 'change',
      label: `${t('menu.run')} (${t('menu.onChange')})`,
      tooltip: t('menu.onChangeTooltip'),
      command: () => {
        useTelemetry()?.trackUiButtonClicked({
          button_id: 'queue_mode_option_run_on_change_selected'
        })
        queueMode.value = 'change'
      }
    }
  }
  if (!isCloud) {
    items['instant-idle'] = {
      key: 'instant-idle',
      label: `${t('menu.run')} (${t('menu.instant')})`,
      tooltip: t('menu.instantTooltip'),
      command: () => {
        useTelemetry()?.trackUiButtonClicked({
          button_id: 'queue_mode_option_run_instant_selected'
        })
        queueMode.value = 'instant-idle'
      }
    }
  }
  return items
})

const activeQueueModeMenuItem = computed(() => {
  // Fallback to disabled mode if current mode is not available (e.g., instant mode in cloud)
  return (
    queueModeMenuItemLookup.value[selectedQueueMode.value] ||
    queueModeMenuItemLookup.value.disabled
  )
})
const queueModeMenuItems = computed(() =>
  Object.values(queueModeMenuItemLookup.value)
)

const isStopInstantAction = computed(() =>
  isInstantRunningMode(queueMode.value)
)

const queueButtonLabel = computed(() =>
  isStopInstantAction.value
    ? t('menu.stopRunInstant')
    : String(activeQueueModeMenuItem.value?.label ?? '')
)

const queueButtonSeverity = computed(() =>
  isStopInstantAction.value ? 'danger' : 'primary'
)

const iconClass = computed(() => {
  if (isStopInstantAction.value) {
    return 'icon-[lucide--square]'
  }
  if (hasMissingNodes.value) {
    return 'icon-[lucide--triangle-alert]'
  }
  if (workspaceStore.shiftDown) {
    return 'icon-[lucide--list-start]'
  }
  if (queueMode.value === 'disabled') {
    return 'icon-[lucide--play]'
  }
  if (isInstantMode(queueMode.value)) {
    return 'icon-[lucide--fast-forward]'
  }
  if (queueMode.value === 'change') {
    return 'icon-[lucide--step-forward]'
  }
  return 'icon-[lucide--play]'
})

const queueButtonTooltip = computed(() => {
  if (isStopInstantAction.value) {
    return t('menu.stopRunInstantTooltip')
  }
  if (hasMissingNodes.value) {
    return t('menu.runWorkflowDisabled')
  }
  if (workspaceStore.shiftDown) {
    return t('menu.runWorkflowFront')
  }
  return t('menu.runWorkflow')
})

const commandStore = useCommandStore()
const queuePrompt = async (e: Event) => {
  if (isStopInstantAction.value) {
    queueMode.value = 'instant-idle'
    return
  }

  const isShiftPressed = 'shiftKey' in e && e.shiftKey
  const commandId = isShiftPressed
    ? 'Comfy.QueuePromptFront'
    : 'Comfy.QueuePrompt'

  if (isInstantMode(queueMode.value)) {
    queueMode.value = 'instant-running'
  }

  if (batchCount.value > 1) {
    useTelemetry()?.trackUiButtonClicked({
      button_id: 'queue_run_multiple_batches_submitted'
    })
  }

  await commandStore.execute(commandId, {
    metadata: {
      subscribe_to_run: false,
      trigger_source: 'button'
    }
  })
}
</script>

<style scoped>
.comfyui-queue-button :deep(.p-splitbutton-dropdown) {
  border-top-right-radius: 0;
  border-bottom-right-radius: 0;
}
</style>
