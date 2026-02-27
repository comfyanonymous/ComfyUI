import { isPromotedWidgetView } from '@/core/graph/subgraph/promotedWidgetTypes'
import { t } from '@/i18n'
import type {
  IContextMenuValue,
  LGraphNode
} from '@/lib/litegraph/src/litegraph'
import type { SubgraphNode } from '@/lib/litegraph/src/subgraph/SubgraphNode'
import type { IBaseWidget } from '@/lib/litegraph/src/types/widgets.ts'
import { useToastStore } from '@/platform/updates/common/toastStore'
import {
  CANVAS_IMAGE_PREVIEW_WIDGET,
  supportsVirtualCanvasImagePreview
} from '@/composables/node/useNodeCanvasImagePreview'
import { useCanvasStore } from '@/renderer/core/canvas/canvasStore'
import { useLitegraphService } from '@/services/litegraphService'
import { usePromotionStore } from '@/stores/promotionStore'
import { useSubgraphNavigationStore } from '@/stores/subgraphNavigationStore'

type PartialNode = Pick<LGraphNode, 'title' | 'id' | 'type'>

export type WidgetItem = [PartialNode, IBaseWidget]
export { CANVAS_IMAGE_PREVIEW_WIDGET }

export function getWidgetName(w: IBaseWidget): string {
  return isPromotedWidgetView(w) ? w.sourceWidgetName : w.name
}

/** Known non-$$ preview widget types added by core or popular extensions. */
const PREVIEW_WIDGET_TYPES = new Set(['preview', 'video', 'audioUI'])

/**
 * Returns true for pseudo-widgets that display media previews and should
 * be auto-promoted when their node is inside a subgraph.
 * Matches the core `$$` convention as well as custom-node patterns
 * (e.g. VHS `videopreview` with type `"preview"`).
 */
export function isPreviewPseudoWidget(widget: IBaseWidget): boolean {
  if (widget.name.startsWith('$$')) return true
  // Custom nodes may set serialize on the widget or in options
  if (widget.serialize !== false && widget.options?.serialize !== false)
    return false
  if (typeof widget.type === 'string' && PREVIEW_WIDGET_TYPES.has(widget.type))
    return true
  return false
}

export function promoteWidget(
  node: PartialNode,
  widget: IBaseWidget,
  parents: SubgraphNode[]
) {
  const store = usePromotionStore()
  const nodeId = String(
    isPromotedWidgetView(widget) ? widget.sourceNodeId : node.id
  )
  const widgetName = getWidgetName(widget)
  for (const parent of parents) {
    store.promote(parent.rootGraph.id, parent.id, nodeId, widgetName)
  }
}

export function demoteWidget(
  node: PartialNode,
  widget: IBaseWidget,
  parents: SubgraphNode[]
) {
  const store = usePromotionStore()
  const nodeId = String(
    isPromotedWidgetView(widget) ? widget.sourceNodeId : node.id
  )
  const widgetName = getWidgetName(widget)
  for (const parent of parents) {
    store.demote(parent.rootGraph.id, parent.id, nodeId, widgetName)
  }
}

function getParentNodes(): SubgraphNode[] {
  const { navigationStack } = useSubgraphNavigationStore()
  const subgraph = navigationStack.at(-1)
  if (!subgraph) {
    useToastStore().add({
      severity: 'error',
      summary: t('g.error'),
      detail: t('subgraphStore.promoteOutsideSubgraph'),
      life: 2000
    })
    return []
  }
  const parentGraph = navigationStack.at(-2) ?? subgraph.rootGraph
  return parentGraph.nodes.filter(
    (node): node is SubgraphNode =>
      node.type === subgraph.id && node.isSubgraphNode()
  )
}

export function addWidgetPromotionOptions(
  options: (IContextMenuValue<unknown> | null)[],
  widget: IBaseWidget,
  node: LGraphNode
) {
  const store = usePromotionStore()
  const parents = getParentNodes()
  const nodeId = String(node.id)
  const widgetName = getWidgetName(widget)
  const promotableParents = parents.filter(
    (s) => !store.isPromoted(s.rootGraph.id, s.id, nodeId, widgetName)
  )
  if (promotableParents.length > 0)
    options.unshift({
      content: t('subgraphStore.promoteWidget', {
        name: widget.label ?? widget.name
      }),
      callback: () => {
        promoteWidget(node, widget, promotableParents)
        widget.callback?.(widget.value)
      }
    })
  else {
    options.unshift({
      content: t('subgraphStore.unpromoteWidget', {
        name: widget.label ?? widget.name
      }),
      callback: () => {
        demoteWidget(node, widget, parents)
        widget.callback?.(widget.value)
      }
    })
  }
}

export function tryToggleWidgetPromotion() {
  const canvas = useCanvasStore().getCanvas()
  const [x, y] = canvas.graph_mouse
  const node = canvas.graph?.getNodeOnPos(x, y, canvas.visible_nodes)
  if (!node) return
  const widget = node.getWidgetOnPos(x, y, true)
  const parents = getParentNodes()
  if (!parents.length || !widget) return
  const store = usePromotionStore()
  const nodeId = String(node.id)
  const widgetName = getWidgetName(widget)
  const promotableParents = parents.filter(
    (s) => !store.isPromoted(s.rootGraph.id, s.id, nodeId, widgetName)
  )
  if (promotableParents.length > 0)
    promoteWidget(node, widget, promotableParents)
  else demoteWidget(node, widget, parents)
}

const recommendedNodes = [
  'CLIPTextEncode',
  'LoadImage',
  'SaveImage',
  'PreviewImage'
]
const recommendedWidgetNames = ['seed']
export function isRecommendedWidget([node, widget]: WidgetItem) {
  return (
    !widget.computedDisabled &&
    (recommendedNodes.includes(node.type) ||
      recommendedWidgetNames.includes(widget.name))
  )
}

function supportsVirtualPreviewWidget(node: LGraphNode): boolean {
  return supportsVirtualCanvasImagePreview(node)
}

function createVirtualCanvasImagePreviewWidget(): IBaseWidget {
  return {
    name: CANVAS_IMAGE_PREVIEW_WIDGET,
    type: 'IMAGE_PREVIEW',
    options: { serialize: false },
    serialize: false,
    y: 0,
    computedDisabled: false
  }
}

export function getPromotableWidgets(node: LGraphNode): IBaseWidget[] {
  const widgets = [...(node.widgets ?? [])]

  const hasCanvasPreviewWidget = widgets.some(
    (widget) => widget.name === CANVAS_IMAGE_PREVIEW_WIDGET
  )
  const supportsVirtualPreview = supportsVirtualPreviewWidget(node)
  if (!hasCanvasPreviewWidget && supportsVirtualPreview) {
    widgets.push(createVirtualCanvasImagePreviewWidget())
  }

  return widgets
}

function nodeWidgets(n: LGraphNode): WidgetItem[] {
  return getPromotableWidgets(n).map((w: IBaseWidget) => [n, w])
}

export function promoteRecommendedWidgets(subgraphNode: SubgraphNode) {
  const store = usePromotionStore()
  const { updatePreviews } = useLitegraphService()
  const interiorNodes = subgraphNode.subgraph.nodes
  for (const node of interiorNodes) {
    node.updateComputedDisabled()

    const hasPreviewWidget = () =>
      node.widgets?.some(isPreviewPseudoWidget) ?? false

    function promotePreviewWidget() {
      const widget = node.widgets?.find(isPreviewPseudoWidget)
      if (!widget) return
      if (
        store.isPromoted(
          subgraphNode.rootGraph.id,
          subgraphNode.id,
          String(node.id),
          widget.name
        )
      )
        return
      promoteWidget(node, widget, [subgraphNode])
    }
    // Promote preview widgets that already exist (e.g. custom node DOM widgets
    // like VHS videopreview that are created in onNodeCreated).
    promotePreviewWidget()

    // If a preview widget already exists in this frame, there's nothing to
    // defer. Core $$ preview widgets are the lazy path that needs updatePreviews.
    if (hasPreviewWidget()) continue

    // Nodes in CANVAS_IMAGE_PREVIEW_NODE_TYPES support a virtual $$
    // preview widget. Eagerly promote it so getPseudoWidgetPreviewTargets
    // includes this node and onDrawBackground can call updatePreviews on it
    // once execution outputs arrive.
    if (supportsVirtualCanvasImagePreview(node)) {
      if (
        !store.isPromoted(
          subgraphNode.rootGraph.id,
          subgraphNode.id,
          String(node.id),
          CANVAS_IMAGE_PREVIEW_WIDGET
        )
      ) {
        store.promote(
          subgraphNode.rootGraph.id,
          subgraphNode.id,
          String(node.id),
          CANVAS_IMAGE_PREVIEW_WIDGET
        )
      }
      continue
    }

    // Also schedule a deferred check: core $$ widgets are created lazily by
    // updatePreviews when node outputs are first loaded.
    requestAnimationFrame(() => updatePreviews(node, promotePreviewWidget))
  }
  const filteredWidgets: WidgetItem[] = interiorNodes
    .flatMap(nodeWidgets)
    .filter(isRecommendedWidget)
  for (const [n, w] of filteredWidgets) {
    store.promote(
      subgraphNode.rootGraph.id,
      subgraphNode.id,
      String(n.id),
      getWidgetName(w)
    )
  }
  subgraphNode.computeSize(subgraphNode.size)
}

export function pruneDisconnected(subgraphNode: SubgraphNode) {
  const store = usePromotionStore()
  const subgraph = subgraphNode.subgraph
  const entries = store.getPromotions(
    subgraphNode.rootGraph.id,
    subgraphNode.id
  )
  const removedEntries: Array<{ interiorNodeId: string; widgetName: string }> =
    []

  const validEntries = entries.filter((entry) => {
    const node = subgraph.getNodeById(entry.interiorNodeId)
    if (!node) {
      removedEntries.push(entry)
      return false
    }
    const hasWidget = getPromotableWidgets(node).some(
      (iw) => iw.name === entry.widgetName
    )
    if (!hasWidget) {
      removedEntries.push(entry)
    }
    return hasWidget
  })

  if (removedEntries.length > 0 && import.meta.env.DEV) {
    console.warn(
      '[proxyWidgetUtils] Pruned disconnected promotions',
      removedEntries,
      {
        graphId: subgraphNode.rootGraph.id,
        subgraphNodeId: subgraphNode.id
      }
    )
  }

  store.setPromotions(subgraphNode.rootGraph.id, subgraphNode.id, validEntries)
}
