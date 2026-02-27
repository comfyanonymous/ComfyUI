<template>
  <a
    ref="wrapperRef"
    v-tooltip.bottom="{
      value: tooltipText,
      showDelay: 512
    }"
    draggable="false"
    href="#"
    class="p-breadcrumb-item-link h-8 cursor-pointer px-2"
    :class="{
      'flex items-center gap-1': isActive,
      'p-breadcrumb-item-link-menu-visible': menu?.overlayVisible,
      'p-breadcrumb-item-link-icon-visible': isActive,
      'active-breadcrumb-item': isActive
    }"
    @click="handleClick"
  >
    <i
      v-if="hasMissingNodes && isRoot"
      class="icon-[lucide--triangle-alert] text-warning-background"
    />
    <span class="p-breadcrumb-item-label px-2">{{ item.label }}</span>
    <Tag v-if="item.isBlueprint" value="Blueprint" severity="primary" />
    <i v-if="isActive" class="pi pi-angle-down text-[10px]"></i>
  </a>
  <Menu
    v-if="isActive || isRoot"
    ref="menu"
    :model="menuItems"
    :popup="true"
    :pt="{
      root: {
        style: 'background-color: var(--comfy-menu-bg)'
      },
      itemLink: {
        class: 'py-2'
      }
    }"
  />
  <InputText
    v-if="isEditing"
    ref="itemInputRef"
    v-model="itemLabel"
    class="fixed z-10000 px-2 py-2 text-[.8rem]"
    @blur="inputBlur(false)"
    @click.stop
    @keydown.enter="inputBlur(true)"
    @keydown.esc="inputBlur(false)"
  />
</template>

<script setup lang="ts">
import InputText from 'primevue/inputtext'
import type { MenuState } from 'primevue/menu'
import Menu from 'primevue/menu'
import type { MenuItem } from 'primevue/menuitem'
import Tag from 'primevue/tag'
import { computed, nextTick, ref } from 'vue'
import { useI18n } from 'vue-i18n'

import { useWorkflowActionsMenu } from '@/composables/useWorkflowActionsMenu'
import { useWorkflowService } from '@/platform/workflow/core/services/workflowService'
import {
  ComfyWorkflow,
  useWorkflowStore
} from '@/platform/workflow/management/stores/workflowStore'
import { app } from '@/scripts/app'
import { useDialogService } from '@/services/dialogService'
import { useCommandStore } from '@/stores/commandStore'
import { useNodeDefStore } from '@/stores/nodeDefStore'
import { useSubgraphNavigationStore } from '@/stores/subgraphNavigationStore'
import { appendJsonExt } from '@/utils/formatUtil'
import { graphHasMissingNodes } from '@/workbench/extensions/manager/utils/graphHasMissingNodes'

interface Props {
  item: MenuItem
  isActive?: boolean
}

const { item, isActive = false } = defineProps<Props>()

const nodeDefStore = useNodeDefStore()
const hasMissingNodes = computed(() =>
  graphHasMissingNodes(app.rootGraph, nodeDefStore.nodeDefsByName)
)

const { t } = useI18n()
const menu = ref<InstanceType<typeof Menu> & MenuState>()
const dialogService = useDialogService()
const workflowStore = useWorkflowStore()
const workflowService = useWorkflowService()
const isEditing = ref(false)
const itemLabel = ref<string>()
const itemInputRef = ref<{ $el?: HTMLInputElement }>()
const wrapperRef = ref<HTMLAnchorElement>()

const rename = async (
  newName: string | null | undefined,
  initialName: string
) => {
  if (newName && newName !== initialName) {
    // Synchronize the node titles with the new name
    item.updateTitle?.(newName)

    if (workflowStore.activeSubgraph) {
      workflowStore.activeSubgraph.name = newName
    } else if (workflowStore.activeWorkflow) {
      try {
        await workflowService.renameWorkflow(
          workflowStore.activeWorkflow,
          ComfyWorkflow.basePath + appendJsonExt(newName)
        )
      } catch (error) {
        console.error(error)
        dialogService.showErrorDialog(error)
        return
      }
    }

    // Force the navigation stack to recompute the labels
    // TODO: investigate if there is a better way to do this
    const navigationStore = useSubgraphNavigationStore()
    navigationStore.restoreState(navigationStore.exportState())
  }
}

const isRoot = item.key === 'root'

const tooltipText = computed(() => {
  if (hasMissingNodes.value && isRoot) {
    return t('breadcrumbsMenu.missingNodesWarning')
  }
  return item.label
})

const startRename = async () => {
  // Check if element is hidden (collapsed breadcrumb)
  // When collapsed, root item is hidden via CSS display:none, so use rename command
  if (isRoot && wrapperRef.value?.offsetParent === null) {
    await useCommandStore().execute('Comfy.RenameWorkflow')
    return
  }

  isEditing.value = true
  itemLabel.value = item.label as string
  void nextTick(() => {
    if (itemInputRef.value?.$el) {
      itemInputRef.value.$el.focus()
      itemInputRef.value.$el.select()
      if (wrapperRef.value) {
        itemInputRef.value.$el.style.width = `${Math.max(200, wrapperRef.value.offsetWidth)}px`
      }
    }
  })
}

const { menuItems } = useWorkflowActionsMenu(startRename, { isRoot })

const handleClick = (event: MouseEvent) => {
  if (isEditing.value) {
    return
  }

  if (event.detail === 1) {
    if (isActive) {
      menu.value?.toggle(event)
    } else {
      item.command?.({ item: item, originalEvent: event })
    }
  } else if (isActive && event.detail === 2) {
    menu.value?.hide()
    event.stopPropagation()
    event.preventDefault()
    startRename()
  }
}

const inputBlur = async (doRename: boolean) => {
  if (doRename) {
    await rename(itemLabel.value, item.label as string)
  }

  isEditing.value = false
}
</script>

<style scoped>
.p-breadcrumb-item-link,
.p-breadcrumb-item-icon {
  user-select: none;
}

.p-breadcrumb-item-link {
  overflow: hidden;
}

.p-breadcrumb-item-label {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.active-breadcrumb-item {
  color: var(--text-primary);
}
</style>
