<template>
  <SidebarTabTemplate :title="$t('sideToolbar.nodes')">
    <template #header>
      <TabsRoot v-model="selectedTab" class="flex flex-col">
        <div class="flex items-center justify-between gap-2 px-2 pb-2 2xl:px-4">
          <SearchBox
            ref="searchBoxRef"
            v-model="searchQuery"
            :placeholder="$t('g.search') + '...'"
            @search="handleSearch"
          />
          <DropdownMenuRoot>
            <DropdownMenuTrigger as-child>
              <button
                :aria-label="$t('g.sort')"
                class="flex size-10 shrink-0 cursor-pointer items-center justify-center rounded-lg bg-comfy-input hover:bg-comfy-input-hover border-none"
              >
                <i class="icon-[lucide--arrow-up-down] size-4" />
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuPortal>
              <DropdownMenuContent
                class="z-[9999] min-w-32 rounded-lg border border-border-default bg-comfy-menu-bg p-1 shadow-lg"
                align="end"
                :side-offset="4"
              >
                <DropdownMenuRadioGroup v-model="sortOrder">
                  <DropdownMenuRadioItem
                    v-for="option in sortingOptions"
                    :key="option.id"
                    :value="option.id"
                    class="flex cursor-pointer items-center justify-end gap-2 rounded-md px-2 py-1.5 text-sm outline-none hover:bg-comfy-input"
                  >
                    <DropdownMenuItemIndicator class="w-4">
                      <i class="icon-[lucide--check] size-4" />
                    </DropdownMenuItemIndicator>
                    <span>{{ $t(option.label) }}</span>
                  </DropdownMenuRadioItem>
                </DropdownMenuRadioGroup>
              </DropdownMenuContent>
            </DropdownMenuPortal>
          </DropdownMenuRoot>
        </div>
        <Separator decorative class="border border-dashed border-comfy-input" />
        <!-- Tab list in header (fixed) -->
        <TabsList
          class="flex gap-4 border-b border-comfy-input bg-background p-4 justify-between"
        >
          <TabsTrigger
            v-for="tab in tabs"
            :key="tab.value"
            :value="tab.value"
            :class="
              cn(
                'flex-1 text-center select-none border-none outline-none px-3 py-2 rounded-lg cursor-pointer',
                'text-sm text-foreground transition-colors',
                selectedTab === tab.value
                  ? 'bg-comfy-input font-bold'
                  : 'bg-transparent font-normal'
              )
            "
          >
            {{ tab.label }}
          </TabsTrigger>
        </TabsList>
      </TabsRoot>
    </template>
    <template #body>
      <NodeDragPreview />
      <!-- Tab content (scrollable) -->
      <TabsRoot v-model="selectedTab" class="h-full">
        <EssentialNodesPanel
          v-if="
            flags.nodeLibraryEssentialsEnabled && selectedTab === 'essentials'
          "
          v-model:expanded-keys="expandedKeys"
          :root="renderedEssentialRoot"
          @node-click="handleNodeClick"
        />
        <AllNodesPanel
          v-if="selectedTab === 'all'"
          v-model:expanded-keys="expandedKeys"
          :sections="renderedSections"
          :fill-node-info="fillNodeInfo"
          @node-click="handleNodeClick"
        />
        <CustomNodesPanel
          v-if="selectedTab === 'custom'"
          v-model:expanded-keys="expandedKeys"
          :sections="renderedCustomSections"
          @node-click="handleNodeClick"
        />
      </TabsRoot>
    </template>
  </SidebarTabTemplate>
</template>

<script setup lang="ts">
import { cn } from '@/utils/tailwindUtil'
import { useLocalStorage } from '@vueuse/core'
import {
  DropdownMenuContent,
  DropdownMenuItemIndicator,
  DropdownMenuPortal,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuRoot,
  DropdownMenuTrigger,
  Separator,
  TabsList,
  TabsRoot,
  TabsTrigger
} from 'reka-ui'
import { computed, nextTick, onMounted, ref, watchEffect } from 'vue'
import { useI18n } from 'vue-i18n'

import { resolveEssentialsDisplayName } from '@/constants/essentialsDisplayNames'
import SearchBox from '@/components/common/SearchBoxV2.vue'
import { useFeatureFlags } from '@/composables/useFeatureFlags'
import { useNodeDragToCanvas } from '@/composables/node/useNodeDragToCanvas'
import { usePerTabState } from '@/composables/usePerTabState'
import {
  DEFAULT_SORTING_ID,
  DEFAULT_TAB_ID,
  nodeOrganizationService
} from '@/services/nodeOrganizationService'
import { getProviderIcon } from '@/utils/categoryUtil'
import { sortedTree } from '@/utils/treeUtil'
import type { ComfyNodeDefImpl } from '@/stores/nodeDefStore'
import { useNodeDefStore } from '@/stores/nodeDefStore'
import type { SortingStrategyId, TabId } from '@/types/nodeOrganizationTypes'
import type {
  RenderedTreeExplorerNode,
  TreeNode
} from '@/types/treeExplorerTypes'

import AllNodesPanel from './nodeLibrary/AllNodesPanel.vue'
import CustomNodesPanel from './nodeLibrary/CustomNodesPanel.vue'
import EssentialNodesPanel from './nodeLibrary/EssentialNodesPanel.vue'
import NodeDragPreview from './nodeLibrary/NodeDragPreview.vue'
import SidebarTabTemplate from './SidebarTabTemplate.vue'

const { flags } = useFeatureFlags()

const selectedTab = useLocalStorage<TabId>(
  'Comfy.NodeLibrary.Tab',
  DEFAULT_TAB_ID
)

watchEffect(() => {
  if (
    !flags.nodeLibraryEssentialsEnabled &&
    selectedTab.value === 'essentials'
  ) {
    selectedTab.value = DEFAULT_TAB_ID
  }
})

const sortOrderByTab = useLocalStorage<Record<TabId, SortingStrategyId>>(
  'Comfy.NodeLibrary.SortByTab',
  {
    essentials: DEFAULT_SORTING_ID,
    all: DEFAULT_SORTING_ID,
    custom: 'alphabetical'
  }
)
const sortOrder = usePerTabState(selectedTab, sortOrderByTab)

const sortingOptions = computed(() =>
  nodeOrganizationService.getSortingStrategies().map((strategy) => ({
    id: strategy.id,
    label: strategy.label
  }))
)

const { t } = useI18n()

const searchBoxRef = ref()
const searchQuery = ref('')
const expandedKeysByTab = ref<Record<TabId, string[]>>({
  essentials: [],
  all: [],
  custom: []
})
const expandedKeys = usePerTabState(selectedTab, expandedKeysByTab)

const nodeDefStore = useNodeDefStore()
const { startDrag } = useNodeDragToCanvas()

const filteredNodeDefs = computed(() => {
  if (searchQuery.value.length === 0) {
    return []
  }
  return nodeDefStore.nodeSearchService.searchNode(
    searchQuery.value,
    [],
    { limit: 64 },
    { matchWildcards: false }
  )
})

const activeNodes = computed(() =>
  filteredNodeDefs.value.length > 0
    ? filteredNodeDefs.value
    : nodeDefStore.visibleNodeDefs
)

const sections = computed(() => {
  if (selectedTab.value !== 'all') return []
  return nodeOrganizationService.organizeNodesByTab(activeNodes.value, 'all')
})

function getFolderIcon(node: TreeNode): string {
  const firstLeaf = findFirstLeaf(node)
  if (
    firstLeaf?.key?.startsWith('root/api node') &&
    firstLeaf.key.replace(`${node.key}/`, '') === firstLeaf.label
  ) {
    return getProviderIcon(node.label ?? '')
  }
  return 'icon-[lucide--folder]'
}

function findFirstLeaf(node: TreeNode): TreeNode | undefined {
  if (node.leaf) return node
  for (const child of node.children ?? []) {
    const leaf = findFirstLeaf(child)
    if (leaf) return leaf
  }
  return undefined
}

function fillNodeInfo(
  node: TreeNode,
  { useEssentialsLabels = false }: { useEssentialsLabels?: boolean } = {}
): RenderedTreeExplorerNode<ComfyNodeDefImpl> {
  const children = node.children?.map((child) =>
    fillNodeInfo(child, { useEssentialsLabels })
  )
  const totalLeaves = node.leaf
    ? 1
    : (children?.reduce((acc, child) => acc + child.totalLeaves, 0) ?? 0)

  return {
    key: node.key,
    label: node.leaf
      ? useEssentialsLabels
        ? (resolveEssentialsDisplayName(node.data) ?? node.data?.display_name)
        : node.data?.display_name
      : node.label,
    leaf: node.leaf,
    data: node.data,
    icon: node.leaf ? 'icon-[comfy--node]' : getFolderIcon(node),
    type: node.leaf ? 'node' : 'folder',
    totalLeaves,
    children
  }
}

function applySorting(tree: TreeNode): TreeNode {
  if (sortOrder.value === 'alphabetical') {
    return sortedTree(tree, { groupLeaf: true })
  }
  return tree
}

const renderedSections = computed(() => {
  return sections.value.map((section) => ({
    title: section.title,
    root: fillNodeInfo(applySorting(section.tree))
  }))
})

const essentialSections = computed(() => {
  if (selectedTab.value !== 'essentials') return []
  return nodeOrganizationService.organizeNodesByTab(
    activeNodes.value,
    'essentials'
  )
})

const renderedEssentialRoot = computed(() => {
  const section = essentialSections.value[0]
  return section
    ? fillNodeInfo(applySorting(section.tree), { useEssentialsLabels: true })
    : fillNodeInfo({ key: 'root', label: '', children: [] })
})

const customSections = computed(() => {
  if (selectedTab.value !== 'custom') return []
  return nodeOrganizationService.organizeNodesByTab(activeNodes.value, 'custom')
})

const renderedCustomSections = computed(() => {
  return customSections.value.map((section) => ({
    title: section.title,
    root: fillNodeInfo(applySorting(section.tree))
  }))
})

function collectFolderKeys(node: TreeNode): string[] {
  if (node.leaf) return []
  const keys = [node.key]
  for (const child of node.children ?? []) {
    keys.push(...collectFolderKeys(child))
  }
  return keys
}

function handleNodeClick(node: RenderedTreeExplorerNode<ComfyNodeDefImpl>) {
  if (node.type === 'node' && node.data) {
    startDrag(node.data)
  }
  if (node.type === 'folder') {
    const index = expandedKeys.value.indexOf(node.key)
    if (index === -1) {
      expandedKeys.value = [...expandedKeys.value, node.key]
    } else {
      expandedKeys.value = expandedKeys.value.filter((k) => k !== node.key)
    }
  }
}

async function handleSearch() {
  await nextTick()

  if (filteredNodeDefs.value.length === 0) {
    expandedKeys.value = []
    return
  }

  const allKeys: string[] = []
  if (selectedTab.value === 'essentials') {
    for (const section of essentialSections.value) {
      allKeys.push(...collectFolderKeys(section.tree))
    }
  } else if (selectedTab.value === 'custom') {
    for (const section of customSections.value) {
      allKeys.push(...collectFolderKeys(section.tree))
    }
  } else {
    for (const section of sections.value) {
      allKeys.push(...collectFolderKeys(section.tree))
    }
  }
  expandedKeys.value = allKeys
}

const tabs = computed(() => {
  const baseTabs: Array<{ value: TabId; label: string }> = [
    { value: 'all', label: t('sideToolbar.nodeLibraryTab.allNodes') },
    { value: 'custom', label: t('sideToolbar.nodeLibraryTab.custom') }
  ]
  return flags.nodeLibraryEssentialsEnabled
    ? [
        {
          value: 'essentials' as TabId,
          label: t('sideToolbar.nodeLibraryTab.essentials')
        },
        ...baseTabs
      ]
    : baseTabs
})

onMounted(() => {
  searchBoxRef.value?.focus()
})
</script>
