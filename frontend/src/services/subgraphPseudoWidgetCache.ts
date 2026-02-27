export interface SubgraphPromotionEntry {
  interiorNodeId: string
  widgetName: string
}

export interface SubgraphPseudoWidget {
  name: string
}

export interface SubgraphPseudoWidgetNode<
  TWidget extends SubgraphPseudoWidget
> {
  id: string | number
  widgets?: TWidget[]
}

interface SubgraphPseudoWidgetCacheEntry<
  TNode extends SubgraphPseudoWidgetNode<TWidget>,
  TWidget extends SubgraphPseudoWidget
> {
  interiorNodeId: string
  widgetName: string
  node: TNode
}

export interface SubgraphPseudoWidgetCache<
  TNode extends SubgraphPseudoWidgetNode<TWidget>,
  TWidget extends SubgraphPseudoWidget
> {
  promotions: readonly SubgraphPromotionEntry[]
  entries: SubgraphPseudoWidgetCacheEntry<TNode, TWidget>[]
  nodes: TNode[]
}

interface ResolveSubgraphPseudoWidgetCacheArgs<
  TNode extends SubgraphPseudoWidgetNode<TWidget>,
  TWidget extends SubgraphPseudoWidget
> {
  cache: SubgraphPseudoWidgetCache<TNode, TWidget> | null
  promotions: readonly SubgraphPromotionEntry[]
  getNodeById: (nodeId: string) => TNode | undefined
  isPreviewPseudoWidget: (widget: TWidget) => boolean
}

interface ResolveSubgraphPseudoWidgetCacheResult<
  TNode extends SubgraphPseudoWidgetNode<TWidget>,
  TWidget extends SubgraphPseudoWidget
> {
  cache: SubgraphPseudoWidgetCache<TNode, TWidget>
  nodes: TNode[]
}

function isPseudoPromotion<TWidget extends SubgraphPseudoWidget>(
  widgetName: string,
  widgets: TWidget[] | undefined,
  isPreviewPseudoWidget: (widget: TWidget) => boolean
): boolean {
  const widget = widgets?.find((candidate) => candidate.name === widgetName)
  if (widget) return isPreviewPseudoWidget(widget)
  return widgetName.startsWith('$$')
}

function isCacheStillValid<
  TNode extends SubgraphPseudoWidgetNode<TWidget>,
  TWidget extends SubgraphPseudoWidget
>(
  cache: SubgraphPseudoWidgetCache<TNode, TWidget>,
  getNodeById: (nodeId: string) => TNode | undefined,
  isPreviewPseudoWidget: (widget: TWidget) => boolean
): boolean {
  return cache.entries.every((entry) => {
    const currentNode = getNodeById(entry.interiorNodeId)
    if (!currentNode || currentNode !== entry.node) return false
    return isPseudoPromotion(
      entry.widgetName,
      currentNode.widgets,
      isPreviewPseudoWidget
    )
  })
}

export function resolveSubgraphPseudoWidgetCache<
  TNode extends SubgraphPseudoWidgetNode<TWidget>,
  TWidget extends SubgraphPseudoWidget
>(
  args: ResolveSubgraphPseudoWidgetCacheArgs<TNode, TWidget>
): ResolveSubgraphPseudoWidgetCacheResult<TNode, TWidget> {
  const { cache, promotions, getNodeById, isPreviewPseudoWidget } = args

  if (
    cache &&
    cache.promotions === promotions &&
    isCacheStillValid(cache, getNodeById, isPreviewPseudoWidget)
  )
    return { cache, nodes: cache.nodes }

  const entries = promotions.flatMap((promotion) => {
    const node = getNodeById(promotion.interiorNodeId)
    if (!node) return []
    if (
      !isPseudoPromotion(
        promotion.widgetName,
        node.widgets,
        isPreviewPseudoWidget
      )
    )
      return []
    return [{ ...promotion, node }]
  })

  const nextCache: SubgraphPseudoWidgetCache<TNode, TWidget> = {
    promotions,
    entries,
    nodes: entries.map((entry) => entry.node)
  }
  return {
    cache: nextCache,
    nodes: nextCache.nodes
  }
}
