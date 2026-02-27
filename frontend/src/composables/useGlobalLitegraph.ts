import {
  ContextMenu,
  DragAndScale,
  LGraph,
  LGraphBadge,
  LGraphCanvas,
  LGraphGroup,
  LGraphNode,
  LLink,
  LiteGraph
} from '@/lib/litegraph/src/litegraph'

/**
 * Assign all properties of LiteGraph to window to make it backward compatible.
 */
export const useGlobalLitegraph = () => {
  // @ts-expect-error fixme ts strict error
  window['LiteGraph'] = LiteGraph
  // @ts-expect-error fixme ts strict error
  window['LGraph'] = LGraph
  // @ts-expect-error fixme ts strict error
  window['LLink'] = LLink
  // @ts-expect-error fixme ts strict error
  window['LGraphNode'] = LGraphNode
  // @ts-expect-error fixme ts strict error
  window['LGraphGroup'] = LGraphGroup
  // @ts-expect-error fixme ts strict error
  window['DragAndScale'] = DragAndScale
  // @ts-expect-error fixme ts strict error
  window['LGraphCanvas'] = LGraphCanvas
  // @ts-expect-error fixme ts strict error
  window['ContextMenu'] = ContextMenu
  // @ts-expect-error fixme ts strict error
  window['LGraphBadge'] = LGraphBadge
}
