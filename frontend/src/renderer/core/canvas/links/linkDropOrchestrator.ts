import type { LGraph } from '@/lib/litegraph/src/LGraph'
import type { NodeId } from '@/lib/litegraph/src/LGraphNode'
import type { LinkConnectorAdapter } from '@/renderer/core/canvas/links/linkConnectorAdapter'
import { useSlotLinkDragUIState } from '@/renderer/core/canvas/links/slotLinkDragUIState'
import type { SlotDropCandidate } from '@/renderer/core/canvas/links/slotLinkDragUIState'
import { getSlotKey } from '@/renderer/core/layout/slots/slotIdentifier'
import { layoutStore } from '@/renderer/core/layout/store/layoutStore'
import type { SlotLinkDragContext } from '@/renderer/extensions/vueNodes/composables/slotLinkDragContext'

interface DropResolutionContext {
  adapter: LinkConnectorAdapter | null
  graph: LGraph | null
  session: SlotLinkDragContext
}

export const resolveSlotTargetCandidate = (
  target: EventTarget | null,
  { adapter, graph }: DropResolutionContext
): SlotDropCandidate | null => {
  const { state: dragState, setCompatibleForKey } = useSlotLinkDragUIState()
  if (!(target instanceof HTMLElement)) return null

  const elWithKey = target
    .closest('.lg-slot, .lg-node-widget')
    ?.querySelector<HTMLElement>('[data-slot-key]')
  const key = elWithKey?.dataset['slotKey']
  if (!key) return null

  const layout = layoutStore.getSlotLayout(key)
  if (!layout) return null

  const candidate: SlotDropCandidate = { layout, compatible: false }

  if (adapter && graph) {
    const cached = dragState.compatible.get(key)
    if (cached != null) {
      candidate.compatible = cached
    } else {
      const nodeId: NodeId = layout.nodeId
      const compatible =
        layout.type === 'input'
          ? adapter.isInputValidDrop(nodeId, layout.index)
          : adapter.isOutputValidDrop(nodeId, layout.index)

      setCompatibleForKey(key, compatible)
      candidate.compatible = compatible
    }
  }

  return candidate
}

export const resolveNodeSurfaceSlotCandidate = (
  target: EventTarget | null,
  { adapter, graph, session }: DropResolutionContext
): SlotDropCandidate | null => {
  const { setCompatibleForKey } = useSlotLinkDragUIState()
  if (!(target instanceof HTMLElement)) return null

  const elWithNode = target.closest<HTMLElement>('[data-node-id]')
  const nodeIdAttr = elWithNode?.dataset['nodeId']
  if (!nodeIdAttr) return null

  if (!adapter || !graph) return null

  const nodeId: NodeId = nodeIdAttr

  const cachedPreferredSlotForNode = session.preferredSlotForNode.get(nodeId)
  if (cachedPreferredSlotForNode !== undefined) {
    return cachedPreferredSlotForNode
      ? { layout: cachedPreferredSlotForNode.layout, compatible: true }
      : null
  }

  const node = graph.getNodeById(nodeId)
  if (!node) return null

  const firstLink = adapter.renderLinks[0]
  if (!firstLink) return null

  const connectingTo = adapter.linkConnector.state.connectingTo
  if (connectingTo !== 'input' && connectingTo !== 'output') return null

  const isInput = connectingTo === 'input'
  const slotType = firstLink.fromSlot.type

  const result = isInput
    ? node.findInputByType(slotType)
    : node.findOutputByType(slotType)

  const index = result?.index
  if (index == null) {
    session.preferredSlotForNode.set(nodeId, null)
    return null
  }

  const key = getSlotKey(String(nodeId), index, isInput)
  const layout = layoutStore.getSlotLayout(key)
  if (!layout) {
    session.preferredSlotForNode.set(nodeId, null)
    return null
  }

  const compatible = isInput
    ? adapter.isInputValidDrop(nodeId, index)
    : adapter.isOutputValidDrop(nodeId, index)

  setCompatibleForKey(key, compatible)

  if (!compatible) {
    session.preferredSlotForNode.set(nodeId, null)
    return null
  }

  const preferred = { index, key, layout }
  session.preferredSlotForNode.set(nodeId, preferred)

  return { layout, compatible: true }
}
