<template>
  <ContextMenuRoot>
    <ContextMenuTrigger as-child>
      <div
        ref="workflowTabRef"
        class="workflow-tab group flex gap-2 p-2"
        v-bind="$attrs"
        @mouseenter="handleMouseEnter"
        @mouseleave="handleMouseLeave"
        @click="handleClick"
      >
        <i
          v-if="workflowOption.workflow.initialMode === 'app'"
          class="icon-[lucide--panels-top-left] bg-primary-background"
        />
        <span
          class="workflow-label inline-block max-w-[150px] truncate text-sm"
        >
          {{ workflowOption.workflow.filename }}
        </span>
        <div class="relative">
          <span
            v-if="shouldShowStatusIndicator"
            class="absolute top-1/2 left-1/2 z-10 w-4 -translate-1/2 bg-(--comfy-menu-bg) text-2xl font-bold group-hover:hidden"
            >•</span
          >
          <Button
            class="close-button invisible w-auto p-0"
            variant="muted-textonly"
            size="icon-sm"
            :aria-label="t('g.close')"
            @click.stop="onCloseWorkflow(workflowOption)"
          >
            <i class="pi pi-times" />
          </Button>
        </div>
      </div>
    </ContextMenuTrigger>
    <ContextMenuPortal>
      <ContextMenuContent
        class="z-1000 rounded-lg px-2 py-3 min-w-56 bg-base-background shadow-interface border border-border-subtle"
      >
        <WorkflowActionsList
          :items="contextMenuItems"
          :item-component="ContextMenuItem"
          :separator-component="ContextMenuSeparator"
        />
      </ContextMenuContent>
    </ContextMenuPortal>
  </ContextMenuRoot>

  <WorkflowTabPopover
    ref="popoverRef"
    :workflow-filename="workflowOption.workflow.filename"
    :thumbnail-url="thumbnailUrl"
    :is-active-tab="isActiveTab"
  />
</template>

<script setup lang="ts">
import {
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuPortal,
  ContextMenuRoot,
  ContextMenuSeparator,
  ContextMenuTrigger
} from 'reka-ui'
import { computed, onUnmounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'

import WorkflowActionsList from '@/components/common/WorkflowActionsList.vue'
import Button from '@/components/ui/button/Button.vue'
import {
  usePragmaticDraggable,
  usePragmaticDroppable
} from '@/composables/usePragmaticDragAndDrop'
import { useWorkflowActionsMenu } from '@/composables/useWorkflowActionsMenu'
import { useSettingStore } from '@/platform/settings/settingStore'
import { useWorkflowService } from '@/platform/workflow/core/services/workflowService'
import type { ComfyWorkflow } from '@/platform/workflow/management/stores/workflowStore'
import { useWorkflowStore } from '@/platform/workflow/management/stores/workflowStore'
import { useWorkflowThumbnail } from '@/renderer/core/thumbnail/useWorkflowThumbnail'
import { useCommandStore } from '@/stores/commandStore'
import { useWorkspaceStore } from '@/stores/workspaceStore'
import type { WorkflowMenuItem } from '@/types/workflowMenuItem'

import WorkflowTabPopover from './WorkflowTabPopover.vue'

interface WorkflowOption {
  value: string
  workflow: ComfyWorkflow
}

const props = defineProps<{
  workflowOption: WorkflowOption
  isFirst: boolean
  isLast: boolean
}>()

const { t } = useI18n()

const workspaceStore = useWorkspaceStore()
const workflowStore = useWorkflowStore()
const settingStore = useSettingStore()
const workflowTabRef = ref<HTMLElement | null>(null)
const popoverRef = ref<InstanceType<typeof WorkflowTabPopover> | null>(null)
const workflowThumbnail = useWorkflowThumbnail()

// Use computed refs to cache autosave settings
const autoSaveSetting = computed(() =>
  settingStore.get('Comfy.Workflow.AutoSave')
)
const autoSaveDelay = computed(() =>
  settingStore.get('Comfy.Workflow.AutoSaveDelay')
)

const shouldShowStatusIndicator = computed(() => {
  if (workspaceStore.shiftDown) {
    // Branch 1: Shift key is held down, do not show the status indicator.
    return false
  }
  if (!props.workflowOption.workflow.isPersisted) {
    // Branch 2: Workflow is not persisted, show the status indicator.
    return true
  }
  if (props.workflowOption.workflow.isModified) {
    // Branch 3: Workflow is modified.
    if (autoSaveSetting.value === 'off') {
      // Sub-branch 3a: Autosave is off, so show the status indicator.
      return true
    }
    if (autoSaveSetting.value === 'after delay' && autoSaveDelay.value > 3000) {
      // Sub-branch 3b: Autosave delay is too high, so show the status indicator.
      return true
    }
    // Sub-branch 3c: Workflow is modified but no condition applies, do not show the status indicator.
    return false
  }
  // Default: do not show the status indicator. This should not be reachable.
  return false
})

const isActiveTab = computed(() => {
  return workflowStore.activeWorkflow?.key === props.workflowOption.workflow.key
})

const thumbnailUrl = computed(() => {
  return workflowThumbnail.getThumbnail(props.workflowOption.workflow.key)
})

// Event handlers that delegate to the popover component
const handleMouseEnter = (event: Event) => {
  popoverRef.value?.showPopover(event)
}

const handleMouseLeave = () => {
  popoverRef.value?.hidePopover()
}

const handleClick = (event: Event) => {
  popoverRef.value?.togglePopover(event)
}

const closeWorkflows = async (options: WorkflowOption[]) => {
  for (const opt of options) {
    if (
      !(await useWorkflowService().closeWorkflow(opt.workflow, {
        warnIfUnsaved: !workspaceStore.shiftDown,
        hint: t('sideToolbar.workflowTab.dirtyCloseHint')
      }))
    ) {
      // User clicked cancel
      break
    }
  }
}

const onCloseWorkflow = async (option: WorkflowOption) => {
  await closeWorkflows([option])
}

const emit = defineEmits<{
  closeToLeft: []
  closeToRight: []
  closeOthers: []
}>()

const commandStore = useCommandStore()
const workflow = computed(() => props.workflowOption.workflow)

const { menuItems: baseMenuItems } = useWorkflowActionsMenu(
  () => commandStore.execute('Comfy.RenameWorkflow'),
  { includeDelete: false, workflow }
)

const contextMenuItems = computed<WorkflowMenuItem[]>(() => [
  ...baseMenuItems.value,
  { separator: true },
  {
    label: t('tabMenu.closeTab'),
    icon: 'pi pi-times',
    command: () => onCloseWorkflow(props.workflowOption)
  },
  {
    label: t('tabMenu.closeTabsToLeft'),
    overlayIcon: {
      mainIcon: 'pi pi-times',
      subIcon: 'pi pi-arrow-left',
      positionX: 'right',
      positionY: 'bottom',
      subIconScale: 0.5
    },
    command: () => emit('closeToLeft'),
    disabled: props.isFirst
  },
  {
    label: t('tabMenu.closeTabsToRight'),
    overlayIcon: {
      mainIcon: 'pi pi-times',
      subIcon: 'pi pi-arrow-right',
      positionX: 'right',
      positionY: 'bottom',
      subIconScale: 0.5
    },
    command: () => emit('closeToRight'),
    disabled: props.isLast
  },
  {
    label: t('tabMenu.closeOtherTabs'),
    overlayIcon: {
      mainIcon: 'pi pi-times',
      subIcon: 'pi pi-arrows-h',
      positionX: 'right',
      positionY: 'bottom',
      subIconScale: 0.5
    },
    command: () => emit('closeOthers'),
    disabled: props.isFirst && props.isLast
  }
])

const tabGetter = () => workflowTabRef.value as HTMLElement

usePragmaticDraggable(tabGetter, {
  getInitialData: () => {
    return {
      workflowKey: props.workflowOption.workflow.key
    }
  }
})

usePragmaticDroppable(tabGetter, {
  getData: () => {
    return {
      workflowKey: props.workflowOption.workflow.key
    }
  },
  onDrop: (e) => {
    const fromIndex = workflowStore.openWorkflows.findIndex(
      (wf) => wf.key === e.source.data.workflowKey
    )
    const toIndex = workflowStore.openWorkflows.findIndex(
      (wf) => wf.key === e.location.current.dropTargets[0]?.data.workflowKey
    )
    if (fromIndex !== toIndex) {
      workflowStore.reorderWorkflows(fromIndex, toIndex)
    }
  }
})

onUnmounted(() => {
  popoverRef.value?.hidePopover()
})
</script>

<style>
.p-tooltip.workflow-tab-tooltip {
  z-index: 1200 !important;
}
</style>
