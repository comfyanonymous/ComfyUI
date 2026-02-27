<template>
  <div class="h-full">
    <SidebarTabTemplate
      v-if="!isHelpOpen"
      :title="$t('sideToolbar.nodeLibrary')"
    >
      <template #tool-buttons>
        <Button
          v-tooltip.bottom="$t('g.newFolder')"
          class="new-folder-button"
          variant="muted-textonly"
          size="icon"
          :aria-label="$t('g.newFolder')"
          @click="nodeBookmarkTreeExplorerRef?.addNewBookmarkFolder()"
        >
          <i class="icon-[lucide--folder-plus] size-4" />
        </Button>
        <Button
          v-tooltip.bottom="$t('sideToolbar.nodeLibraryTab.groupBy')"
          variant="muted-textonly"
          size="icon"
          :aria-label="$t('sideToolbar.nodeLibraryTab.groupBy')"
          @click="groupingPopover?.toggle($event)"
        >
          <i :class="[selectedGroupingIcon, 'size-4']" />
        </Button>
        <Button
          v-tooltip.bottom="$t('sideToolbar.nodeLibraryTab.sortMode')"
          variant="muted-textonly"
          size="icon"
          :aria-label="$t('sideToolbar.nodeLibraryTab.sortMode')"
          @click="sortingPopover?.toggle($event)"
        >
          <i :class="[selectedSortingIcon, 'size-4']" />
        </Button>
        <Button
          v-tooltip.bottom="$t('sideToolbar.nodeLibraryTab.resetView')"
          variant="muted-textonly"
          size="icon"
          :aria-label="$t('sideToolbar.nodeLibraryTab.resetView')"
          @click="resetOrganization"
        >
          <i class="icon-[lucide--filter-x] size-4" />
        </Button>
        <Button
          v-tooltip.bottom="$t('menu.refresh')"
          variant="muted-textonly"
          size="icon"
          :aria-label="$t('menu.refresh')"
          @click="() => commandStore.execute('Comfy.RefreshNodeDefinitions')"
        >
          <i class="icon-[lucide--refresh-cw] size-4" />
        </Button>
        <Popover ref="groupingPopover">
          <div class="flex flex-col gap-1 p-2">
            <Button
              v-for="option in groupingOptions"
              :key="option.id"
              :variant="
                selectedGroupingId === option.id ? 'primary' : 'textonly'
              "
              class="justify-start"
              @click="selectGrouping(option.id)"
            >
              <i :class="[option.icon, 'size-4']" />
              {{ $t(option.label) }}
            </Button>
          </div>
        </Popover>
        <Popover ref="sortingPopover">
          <div class="flex flex-col gap-1 p-2">
            <Button
              v-for="option in sortingOptions"
              :key="option.id"
              :variant="
                selectedSortingId === option.id ? 'primary' : 'textonly'
              "
              class="justify-start"
              @click="selectSorting(option.id)"
            >
              <i :class="[option.icon, 'size-4']" />
              {{ $t(option.label) }}
            </Button>
          </div>
        </Popover>
      </template>
      <template #header>
        <div class="px-2 2xl:px-4">
          <SearchBox
            ref="searchBoxRef"
            v-model:model-value="searchQuery"
            data-testid="node-library-search"
            class="node-lib-search-box"
            :placeholder="$t('g.searchPlaceholder', { subject: $t('g.nodes') })"
            filter-icon="pi pi-filter"
            :filters
            @search="handleSearch"
            @show-filter="($event) => searchFilter?.toggle($event)"
            @remove-filter="onRemoveFilter"
          />

          <Popover ref="searchFilter" class="ml-[-13px]">
            <NodeSearchFilter @add-filter="onAddFilter" />
          </Popover>
        </div>
      </template>
      <template #body>
        <div>
          <NodeBookmarkTreeExplorer
            ref="nodeBookmarkTreeExplorerRef"
            :filtered-node-defs="filteredNodeDefs"
            :open-node-help="openHelp"
          />
          <Divider
            v-show="nodeBookmarkStore.bookmarks.length > 0"
            type="dashed"
            class="m-2"
          />
          <TreeExplorer
            v-model:expanded-keys="expandedKeys"
            data-testid="node-library-tree"
            class="node-lib-tree-explorer"
            :root="renderedRoot"
          >
            <template #folder="{ node }">
              <NodeTreeFolder :node="node" />
            </template>
            <template #node="{ node }">
              <NodeTreeLeaf :node="node" :open-node-help="openHelp" />
            </template>
          </TreeExplorer>
        </div>
      </template>
    </SidebarTabTemplate>

    <NodeHelpPage v-else :node="currentHelpNode!" @close="closeHelp" />
  </div>
  <div id="node-library-node-preview-container" />
</template>

<script setup lang="ts">
import { useLocalStorage } from '@vueuse/core'
import { storeToRefs } from 'pinia'
import Divider from 'primevue/divider'
import Popover from 'primevue/popover'
import type { Ref } from 'vue'
import {
  computed,
  getCurrentInstance,
  h,
  nextTick,
  onMounted,
  ref,
  render
} from 'vue'

import { resolveEssentialsDisplayName } from '@/constants/essentialsDisplayNames'
import SearchBox from '@/components/common/SearchBox.vue'
import type { SearchFilter } from '@/components/common/SearchFilterChip.vue'
import TreeExplorer from '@/components/common/TreeExplorer.vue'
import NodePreview from '@/components/node/NodePreview.vue'
import NodeSearchFilter from '@/components/searchbox/NodeSearchFilter.vue'
import SidebarTabTemplate from '@/components/sidebar/tabs/SidebarTabTemplate.vue'
import NodeHelpPage from '@/components/sidebar/tabs/nodeLibrary/NodeHelpPage.vue'
import NodeTreeFolder from '@/components/sidebar/tabs/nodeLibrary/NodeTreeFolder.vue'
import NodeTreeLeaf from '@/components/sidebar/tabs/nodeLibrary/NodeTreeLeaf.vue'
import Button from '@/components/ui/button/Button.vue'
import { useTreeExpansion } from '@/composables/useTreeExpansion'
import { useLitegraphService } from '@/services/litegraphService'
import {
  DEFAULT_GROUPING_ID,
  DEFAULT_SORTING_ID,
  nodeOrganizationService
} from '@/services/nodeOrganizationService'
import { useCommandStore } from '@/stores/commandStore'
import { useNodeBookmarkStore } from '@/stores/nodeBookmarkStore'
import type { ComfyNodeDefImpl } from '@/stores/nodeDefStore'
import { useNodeDefStore } from '@/stores/nodeDefStore'
import { useNodeHelpStore } from '@/stores/workspace/nodeHelpStore'
import type {
  GroupingStrategyId,
  SortingStrategyId
} from '@/types/nodeOrganizationTypes'
import type { TreeExplorerNode, TreeNode } from '@/types/treeExplorerTypes'
import type { FuseFilterWithValue } from '@/utils/fuseUtil'

import NodeBookmarkTreeExplorer from './nodeLibrary/NodeBookmarkTreeExplorer.vue'

const instance = getCurrentInstance()!
const appContext = instance.appContext
const nodeDefStore = useNodeDefStore()
const nodeBookmarkStore = useNodeBookmarkStore()
const nodeHelpStore = useNodeHelpStore()
const commandStore = useCommandStore()
const expandedKeys = ref<Record<string, boolean>>({})
const { expandNode, toggleNodeOnEvent } = useTreeExpansion(expandedKeys)

const nodeBookmarkTreeExplorerRef = ref<InstanceType<
  typeof NodeBookmarkTreeExplorer
> | null>(null)
const searchBoxRef = ref()

onMounted(() => {
  searchBoxRef.value?.focus()
})

const searchFilter = ref<InstanceType<typeof Popover> | null>(null)
const groupingPopover = ref<InstanceType<typeof Popover> | null>(null)
const sortingPopover = ref<InstanceType<typeof Popover> | null>(null)
const selectedGroupingId = useLocalStorage<GroupingStrategyId>(
  'Comfy.NodeLibrary.GroupBy',
  DEFAULT_GROUPING_ID
)
const selectedSortingId = useLocalStorage<SortingStrategyId>(
  'Comfy.NodeLibrary.SortBy',
  DEFAULT_SORTING_ID
)

const searchQuery = ref<string>('')

const { currentHelpNode, isHelpOpen } = storeToRefs(nodeHelpStore)
const { openHelp, closeHelp } = nodeHelpStore

const groupingOptions = computed(() =>
  nodeOrganizationService.getGroupingStrategies().map((strategy) => ({
    id: strategy.id,
    label: strategy.label,
    icon: strategy.icon
  }))
)
const sortingOptions = computed(() =>
  nodeOrganizationService.getSortingStrategies().map((strategy) => ({
    id: strategy.id,
    label: strategy.label,
    icon: strategy.icon
  }))
)

const selectedGroupingIcon = computed(() =>
  nodeOrganizationService.getGroupingIcon(selectedGroupingId.value)
)
const selectedSortingIcon = computed(() =>
  nodeOrganizationService.getSortingIcon(selectedSortingId.value)
)

const selectGrouping = (groupingId: string) => {
  selectedGroupingId.value = groupingId as GroupingStrategyId
  groupingPopover.value?.hide()
}
const selectSorting = (sortingId: string) => {
  selectedSortingId.value = sortingId as SortingStrategyId
  sortingPopover.value?.hide()
}

const resetOrganization = () => {
  selectedGroupingId.value = DEFAULT_GROUPING_ID
  selectedSortingId.value = DEFAULT_SORTING_ID
}

const root = computed(() => {
  // Determine which nodes to use
  const nodes =
    filteredNodeDefs.value.length > 0
      ? filteredNodeDefs.value
      : nodeDefStore.visibleNodeDefs

  // Use the service to organize nodes
  return nodeOrganizationService.organizeNodes(nodes, {
    groupBy: selectedGroupingId.value,
    sortBy: selectedSortingId.value
  })
})

const renderedRoot = computed<TreeExplorerNode<ComfyNodeDefImpl>>(() => {
  const fillNodeInfo = (node: TreeNode): TreeExplorerNode<ComfyNodeDefImpl> => {
    const children = node.children?.map(fillNodeInfo)

    return {
      key: node.key,
      label: node.leaf
        ? (resolveEssentialsDisplayName(node.data) ?? node.data.display_name)
        : node.label,
      leaf: node.leaf,
      data: node.data,
      getIcon() {
        if (this.leaf) {
          return 'pi pi-circle-fill'
        }
      },
      children,
      draggable: node.leaf,
      renderDragPreview(container) {
        const vnode = h(NodePreview, { nodeDef: node.data })
        vnode.appContext = appContext
        render(vnode, container)
        return () => {
          render(null, container)
        }
      },
      handleClick(e: MouseEvent) {
        if (this.leaf) {
          // @ts-expect-error fixme ts strict error
          useLitegraphService().addNodeOnGraph(this.data)
        } else {
          toggleNodeOnEvent(e, this)
        }
      }
    }
  }
  return fillNodeInfo(root.value)
})

const filteredNodeDefs = ref<ComfyNodeDefImpl[]>([])
const filters: Ref<
  (SearchFilter & { filter: FuseFilterWithValue<ComfyNodeDefImpl, string> })[]
> = ref([])
const handleSearch = async (query: string) => {
  // Don't apply a min length filter because it does not make sense in
  // multi-byte languages like Chinese, Japanese, Korean, etc.
  if (query.length === 0 && !filters.value.length) {
    filteredNodeDefs.value = []
    expandedKeys.value = {}
    return
  }

  const f = filters.value.map((f) => f.filter)
  filteredNodeDefs.value = nodeDefStore.nodeSearchService.searchNode(
    query,
    f,
    {
      limit: 64
    },
    {
      matchWildcards: false
    }
  )

  await nextTick()
  // Expand the search results tree
  if (filteredNodeDefs.value.length > 0) {
    expandNode(root.value)
  }
}

const onAddFilter = async (
  filterAndValue: FuseFilterWithValue<ComfyNodeDefImpl, string>
) => {
  filters.value.push({
    filter: filterAndValue,
    badge: filterAndValue.filterDef.invokeSequence.toUpperCase(),
    badgeClass: filterAndValue.filterDef.invokeSequence + '-badge',
    text: filterAndValue.value,
    id: +new Date()
  })

  await handleSearch(searchQuery.value)
}

// @ts-expect-error fixme ts strict error
const onRemoveFilter = async (filterAndValue) => {
  const index = filters.value.findIndex((f) => f === filterAndValue)
  if (index !== -1) {
    filters.value.splice(index, 1)
  }
  await handleSearch(searchQuery.value)
}
</script>
