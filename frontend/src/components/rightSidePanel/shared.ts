import type { InjectionKey, MaybeRefOrGetter } from 'vue'
import { computed, toValue } from 'vue'
import Fuse from 'fuse.js'
import type { IFuseOptions } from 'fuse.js'

import type { Positionable } from '@/lib/litegraph/src/interfaces'
import type { LGraphGroup } from '@/lib/litegraph/src/LGraphGroup'
import type { LGraphNode, NodeId } from '@/lib/litegraph/src/LGraphNode'
import type { IBaseWidget } from '@/lib/litegraph/src/types/widgets'
import { isLGraphGroup, isLGraphNode } from '@/utils/litegraphUtil'
import { useSettingStore } from '@/platform/settings/settingStore'

export const GetNodeParentGroupKey: InjectionKey<
  (node: LGraphNode) => LGraphGroup | null
> = Symbol('getNodeParentGroup')

export type NodeWidgetsList = Array<{ node: LGraphNode; widget: IBaseWidget }>
export type NodeWidgetsListList = Array<{
  node: LGraphNode
  widgets: NodeWidgetsList
}>

interface WidgetSearchItem {
  index: number
  searchableLabel: string
  searchableName: string
  searchableType: string
  searchableValue: string
}

/**
 * Searches widgets in a list using fuzzy search and returns search results.
 * Uses Fuse.js for better matching with typo tolerance and relevance ranking.
 * Filters by name, localized label, type, and user-input value.
 */
export function searchWidgets<T extends { widget: IBaseWidget }[]>(
  list: T,
  query: string
): T {
  if (query.trim() === '') {
    return list
  }

  const searchableList: WidgetSearchItem[] = list.map((item, index) => {
    const searchableItem = {
      index,
      searchableLabel: item.widget.label?.toLowerCase() || '',
      searchableName: item.widget.name.toLowerCase(),
      searchableType: item.widget.type.toLowerCase(),
      searchableValue: item.widget.value?.toString().toLowerCase() || ''
    }
    return searchableItem
  })

  const fuseOptions: IFuseOptions<WidgetSearchItem> = {
    keys: [
      { name: 'searchableName', weight: 0.4 },
      { name: 'searchableLabel', weight: 0.3 },
      { name: 'searchableValue', weight: 0.3 },
      { name: 'searchableType', weight: 0.2 }
    ],
    threshold: 0.3
  }

  const fuse = new Fuse(searchableList, fuseOptions)
  const results = fuse.search(query.trim())

  const matchedItems = new Set(
    results.map((result) => list[result.item.index]!)
  )

  return list.filter((item) => matchedItems.has(item)) as T
}

type NodeSearchItem = {
  nodeId: NodeId
  searchableTitle: string
}

/**
 * Searches widgets and nodes in a list using fuzzy search and returns search results.
 * Uses Fuse.js for node title matching with typo tolerance and relevance ranking.
 * First checks if the node title matches the query (if so, keeps entire node).
 * Otherwise, filters widgets using searchWidgets.
 */
export function searchWidgetsAndNodes(
  list: NodeWidgetsListList,
  query: string
): NodeWidgetsListList {
  if (query.trim() === '') {
    return list
  }

  const searchableList: NodeSearchItem[] = list.map((item) => ({
    nodeId: item.node.id,
    searchableTitle: (item.node.getTitle() ?? '').toLowerCase()
  }))

  const fuseOptions: IFuseOptions<NodeSearchItem> = {
    keys: [{ name: 'searchableTitle', weight: 1.0 }],
    threshold: 0.3
  }

  const fuse = new Fuse(searchableList, fuseOptions)
  const nodeMatches = fuse.search(query.trim())
  const matchedNodeIds = new Set(
    nodeMatches.map((result) => result.item.nodeId)
  )

  return list
    .map((item) => {
      if (matchedNodeIds.has(item.node.id)) {
        return { ...item, keep: true }
      }
      return {
        ...item,
        keep: false,
        widgets: searchWidgets(item.widgets, query)
      }
    })
    .filter((item) => item.keep || item.widgets.length > 0)
}

type MixedSelectionItem = LGraphGroup | LGraphNode
type FlatAndCategorizeSelectedItemsResult = {
  all: MixedSelectionItem[]
  nodes: LGraphNode[]
  groups: LGraphGroup[]
  others: Positionable[]
  nodeToParentGroup: Map<LGraphNode, LGraphGroup>
}

type FlatItemsContext = {
  nodeToParentGroup: Map<LGraphNode, LGraphGroup>
  depth: number
  parentGroup?: LGraphGroup
}

/**
 * The selected items may contain "Group" nodes, which can include child nodes.
 * This function flattens such structures and categorizes items into:
 * - all: all categorizable nodes (does not include nodes in "others")
 * - nodes: node items
 * - groups: group items
 * - others: items not currently supported
 * - nodeToParentGroup: a map from each node to its direct parent group (if any)
 * @param items The selected items to flatten and categorize
 * @returns An object containing arrays: all, nodes, groups, others, and nodeToParentGroup map
 */
export function flatAndCategorizeSelectedItems(
  items: Positionable[]
): FlatAndCategorizeSelectedItemsResult {
  const ctx: FlatItemsContext = {
    nodeToParentGroup: new Map<LGraphNode, LGraphGroup>(),
    depth: 0
  }
  const { all, nodes, groups, others } = flatItems(items, ctx)
  return {
    all: repeatItems(all),
    nodes: repeatItems(nodes),
    groups: repeatItems(groups),
    others: repeatItems(others),
    nodeToParentGroup: ctx.nodeToParentGroup
  }
}

export function useFlatAndCategorizeSelectedItems(
  items: MaybeRefOrGetter<Positionable[]>
) {
  const result = computed(() => flatAndCategorizeSelectedItems(toValue(items)))

  return {
    flattedItems: computed(() => result.value.all),
    selectedNodes: computed(() => result.value.nodes),
    selectedGroups: computed(() => result.value.groups),
    selectedOthers: computed(() => result.value.others),
    nodeToParentGroup: computed(() => result.value.nodeToParentGroup)
  }
}

function flatItems(
  items: Positionable[],
  ctx: FlatItemsContext
): Omit<FlatAndCategorizeSelectedItemsResult, 'nodeToParentGroup'> {
  const result: MixedSelectionItem[] = []
  const nodes: LGraphNode[] = []
  const groups: LGraphGroup[] = []
  const others: Positionable[] = []

  if (ctx.depth > 1000) {
    return {
      all: [],
      nodes: [],
      groups: [],
      others: []
    }
  }

  for (let i = 0; i < items.length; i++) {
    const item = items[i] as Positionable

    if (isLGraphGroup(item)) {
      result.push(item)
      groups.push(item)

      const children = Array.from(item.children)
      const childCtx: FlatItemsContext = {
        nodeToParentGroup: ctx.nodeToParentGroup,
        depth: ctx.depth + 1,
        parentGroup: item
      }
      const {
        all: childAll,
        nodes: childNodes,
        groups: childGroups,
        others: childOthers
      } = flatItems(children, childCtx)
      result.push(...childAll)
      nodes.push(...childNodes)
      groups.push(...childGroups)
      others.push(...childOthers)
    } else if (isLGraphNode(item)) {
      result.push(item)
      nodes.push(item)
      if (ctx.parentGroup) {
        ctx.nodeToParentGroup.set(item, ctx.parentGroup)
      }
    } else {
      // Other types of items are not supported yet
      // Do not add to all
      others.push(item)
    }
  }
  return {
    all: result,
    nodes,
    groups,
    others
  }
}

function repeatItems<T>(items: T[]): T[] {
  const itemSet = new Set<T>()
  const result: T[] = []
  for (const item of items) {
    if (itemSet.has(item)) continue
    itemSet.add(item)
    result.push(item)
  }
  return result
}

export function computedSectionDataList(nodes: MaybeRefOrGetter<LGraphNode[]>) {
  const settingStore = useSettingStore()

  const includesAdvanced = computed(() =>
    settingStore.get('Comfy.Node.AlwaysShowAdvancedWidgets')
  )

  const widgetsSectionDataList = computed((): NodeWidgetsListList => {
    return toValue(nodes).map((node) => {
      const { widgets = [] } = node
      const shownWidgets = widgets
        .filter(
          (w) =>
            !(
              w.options?.canvasOnly ||
              w.options?.hidden ||
              (w.options?.advanced && !includesAdvanced.value)
            )
        )
        .map((widget) => ({ node, widget }))
      return { widgets: shownWidgets, node }
    })
  })

  return {
    widgetsSectionDataList,
    includesAdvanced
  }
}
