<script setup lang="ts">
import { storeToRefs } from 'pinia'
import { computed, provide, ref, watchEffect } from 'vue'
import { useI18n } from 'vue-i18n'

import EditableText from '@/components/common/EditableText.vue'
import Tab from '@/components/tab/Tab.vue'
import TabList from '@/components/tab/TabList.vue'
import Button from '@/components/ui/button/Button.vue'
import { useGraphHierarchy } from '@/composables/graph/useGraphHierarchy'
import { st } from '@/i18n'
import { SubgraphNode } from '@/lib/litegraph/src/litegraph'
import type { LGraphNode } from '@/lib/litegraph/src/litegraph'
import { useSettingStore } from '@/platform/settings/settingStore'
import { useCanvasStore } from '@/renderer/core/canvas/canvasStore'
import { useExecutionErrorStore } from '@/stores/executionErrorStore'
import { useRightSidePanelStore } from '@/stores/workspace/rightSidePanelStore'
import type { RightSidePanelTab } from '@/stores/workspace/rightSidePanelStore'
import { resolveNodeDisplayName } from '@/utils/nodeTitleUtil'
import { cn } from '@/utils/tailwindUtil'
import { isGroupNode } from '@/utils/executableGroupNodeDto'

import TabInfo from './info/TabInfo.vue'
import TabGlobalParameters from './parameters/TabGlobalParameters.vue'
import TabNodes from './parameters/TabNodes.vue'
import TabNormalInputs from './parameters/TabNormalInputs.vue'
import TabSubgraphInputs from './parameters/TabSubgraphInputs.vue'
import TabGlobalSettings from './settings/TabGlobalSettings.vue'
import TabSettings from './settings/TabSettings.vue'
import {
  GetNodeParentGroupKey,
  useFlatAndCategorizeSelectedItems
} from './shared'
import SubgraphEditor from './subgraph/SubgraphEditor.vue'
import TabErrors from './errors/TabErrors.vue'

const canvasStore = useCanvasStore()
const executionErrorStore = useExecutionErrorStore()
const rightSidePanelStore = useRightSidePanelStore()
const settingStore = useSettingStore()
const { t } = useI18n()

const { hasAnyError, allErrorExecutionIds, activeMissingNodeGraphIds } =
  storeToRefs(executionErrorStore)

const { findParentGroup } = useGraphHierarchy()

const { selectedItems: directlySelectedItems } = storeToRefs(canvasStore)
const { activeTab, isEditingSubgraph } = storeToRefs(rightSidePanelStore)

const sidebarLocation = computed<'left' | 'right'>(() =>
  settingStore.get('Comfy.Sidebar.Location')
)

// Panel is on the left when sidebar is on the right, and vice versa
const panelIcon = computed(() =>
  sidebarLocation.value === 'right'
    ? 'icon-[lucide--panel-left]'
    : 'icon-[lucide--panel-right]'
)

const { flattedItems, selectedNodes, selectedGroups, nodeToParentGroup } =
  useFlatAndCategorizeSelectedItems(directlySelectedItems)

const shouldShowGroupNames = computed(() => {
  return !(
    directlySelectedItems.value.length === 1 &&
    (selectedGroups.value.length === 1 || selectedNodes.value.length === 1)
  )
})

provide(GetNodeParentGroupKey, (node: LGraphNode) => {
  if (!shouldShowGroupNames.value) return null
  return nodeToParentGroup.value.get(node) ?? findParentGroup(node)
})

const hasSelection = computed(() => flattedItems.value.length > 0)

const selectedSingleNode = computed(() => {
  return selectedNodes.value.length === 1 && flattedItems.value.length === 1
    ? selectedNodes.value[0]
    : null
})

const isSingleSubgraphNode = computed(() => {
  return selectedSingleNode.value instanceof SubgraphNode
})

function closePanel() {
  rightSidePanelStore.closePanel()
}

type RightSidePanelTabList = Array<{
  label: () => string
  value: RightSidePanelTab
  icon?: string
}>

const hasDirectNodeError = computed(() =>
  selectedNodes.value.some((node) =>
    executionErrorStore.activeGraphErrorNodeIds.has(String(node.id))
  )
)

const hasContainerInternalError = computed(() => {
  if (allErrorExecutionIds.value.length === 0) return false
  return selectedNodes.value.some((node) => {
    if (!(node instanceof SubgraphNode || isGroupNode(node))) return false
    return executionErrorStore.isContainerWithInternalError(node)
  })
})

const hasMissingNodeSelected = computed(
  () =>
    hasSelection.value &&
    selectedNodes.value.some((node) =>
      activeMissingNodeGraphIds.value.has(String(node.id))
    )
)

const hasRelevantErrors = computed(() => {
  if (!hasSelection.value) return hasAnyError.value
  return (
    hasDirectNodeError.value ||
    hasContainerInternalError.value ||
    hasMissingNodeSelected.value
  )
})

const tabs = computed<RightSidePanelTabList>(() => {
  const list: RightSidePanelTabList = []

  if (
    settingStore.get('Comfy.RightSidePanel.ShowErrorsTab') &&
    hasRelevantErrors.value
  ) {
    list.push({
      label: () => t('rightSidePanel.errors'),
      value: 'errors',
      icon: 'icon-[lucide--octagon-alert] bg-node-stroke-error ml-1'
    })
  }

  list.push({
    label: () =>
      flattedItems.value.length > 1
        ? t('rightSidePanel.nodes')
        : t('rightSidePanel.parameters'),
    value: 'parameters'
  })

  if (!hasSelection.value) {
    list.push({
      label: () => t('rightSidePanel.nodes'),
      value: 'nodes'
    })
  }

  if (hasSelection.value) {
    if (selectedSingleNode.value && !isSingleSubgraphNode.value) {
      list.push({
        label: () => t('rightSidePanel.info'),
        value: 'info'
      })
    }
  }

  list.push({
    label: () =>
      hasSelection.value
        ? t('g.settings')
        : t('rightSidePanel.globalSettings.title'),
    value: 'settings'
  })

  return list
})

// Use global state for activeTab and ensure it's valid
watchEffect(() => {
  if (
    !tabs.value.some((tab) => tab.value === activeTab.value) &&
    !(activeTab.value === 'subgraph' && isSingleSubgraphNode.value)
  ) {
    rightSidePanelStore.openPanel(tabs.value[0].value)
  }
})

function resolveTitle() {
  const items = flattedItems.value
  const nodes = selectedNodes.value
  const groups = selectedGroups.value

  if (items.length === 0) {
    return t('rightSidePanel.workflowOverview')
  }
  if (directlySelectedItems.value.length === 1) {
    if (groups.length === 1) {
      return groups[0].title || t('rightSidePanel.fallbackGroupTitle')
    }
    if (nodes.length === 1) {
      const fallbackNodeTitle = t('rightSidePanel.fallbackNodeTitle')
      return resolveNodeDisplayName(nodes[0], {
        emptyLabel: fallbackNodeTitle,
        untitledLabel: fallbackNodeTitle,
        st
      })
    }
  }
  return t('rightSidePanel.title', { count: items.length })
}

const panelTitle = ref(resolveTitle())
watchEffect(() => (panelTitle.value = resolveTitle()))

const isEditing = ref(false)

const allowTitleEdit = computed(() => {
  return (
    directlySelectedItems.value.length === 1 &&
    (selectedGroups.value.length === 1 || selectedNodes.value.length === 1)
  )
})

function handleTitleEdit(newTitle: string) {
  isEditing.value = false

  const trimmedTitle = newTitle.trim()
  if (!trimmedTitle) return

  const node = selectedGroups.value[0] || selectedNodes.value[0]
  if (!node) return

  if (trimmedTitle === node.title) return

  node.title = trimmedTitle
  panelTitle.value = trimmedTitle
  canvasStore.canvas?.setDirty(true, true)
}

function handleTitleCancel() {
  isEditing.value = false
}
</script>

<template>
  <div
    data-testid="properties-panel"
    class="flex size-full flex-col bg-comfy-menu-bg"
  >
    <!-- Panel Header -->
    <section class="pt-1">
      <div class="flex items-center justify-between pl-4 pr-3">
        <h3 class="my-3.5 text-sm font-semibold line-clamp-2 cursor-default">
          <template v-if="allowTitleEdit">
            <EditableText
              :model-value="panelTitle"
              :is-editing="isEditing"
              :input-attrs="{ 'data-testid': 'node-title-input' }"
              class="cursor-text"
              @edit="handleTitleEdit"
              @cancel="handleTitleCancel"
              @click="isEditing = true"
            />
            <i
              v-if="!isEditing"
              class="icon-[lucide--pencil] size-4 text-muted-foreground ml-2 content-center relative top-[2px] hover:text-base-foreground cursor-pointer shrink-0"
              @click="isEditing = true"
            />
          </template>
          <template v-else>
            {{ panelTitle }}
          </template>
        </h3>

        <div class="flex gap-2">
          <Button
            v-if="isSingleSubgraphNode"
            variant="secondary"
            size="icon"
            :class="cn(isEditingSubgraph && 'bg-secondary-background-selected')"
            @click="
              rightSidePanelStore.openPanel(
                isEditingSubgraph ? 'parameters' : 'subgraph'
              )
            "
          >
            <i class="icon-[lucide--settings-2]" />
          </Button>
          <Button
            variant="secondary"
            size="icon"
            :aria-pressed="rightSidePanelStore.isOpen"
            :aria-label="t('rightSidePanel.togglePanel')"
            @click="closePanel"
          >
            <i :class="cn(panelIcon, 'size-4')" />
          </Button>
        </div>
      </div>
      <nav class="px-4 pb-2 pt-1 overflow-x-auto">
        <TabList
          :model-value="activeTab"
          @update:model-value="
            (newTab: RightSidePanelTab) => {
              rightSidePanelStore.openPanel(newTab)
            }
          "
        >
          <Tab
            v-for="tab in tabs"
            :key="tab.value"
            class="text-sm py-1 px-2 font-inter transition-all active:scale-95"
            :value="tab.value"
          >
            {{ tab.label() }}
            <i v-if="tab.icon" :class="cn(tab.icon, 'size-4')" />
          </Tab>
        </TabList>
      </nav>
    </section>

    <!-- Panel Content -->
    <div class="scrollbar-thin flex-1 overflow-y-auto">
      <TabErrors v-if="activeTab === 'errors'" />
      <template v-else-if="!hasSelection">
        <TabGlobalParameters v-if="activeTab === 'parameters'" />
        <TabNodes v-else-if="activeTab === 'nodes'" />
        <TabGlobalSettings v-else-if="activeTab === 'settings'" />
      </template>
      <SubgraphEditor
        v-else-if="isSingleSubgraphNode && isEditingSubgraph"
        :node="selectedSingleNode"
      />
      <template v-else>
        <TabSubgraphInputs
          v-if="activeTab === 'parameters' && isSingleSubgraphNode"
          :node="selectedSingleNode as SubgraphNode"
        />
        <TabNormalInputs
          v-else-if="activeTab === 'parameters'"
          :nodes="selectedNodes"
          :must-show-node-title="selectedGroups.length > 0"
        />
        <TabInfo v-else-if="activeTab === 'info'" :nodes="selectedNodes" />
        <TabSettings
          v-else-if="activeTab === 'settings'"
          :nodes="flattedItems"
        />
      </template>
    </div>
  </div>
</template>
