import { tryOnScopeDispose, useEventListener } from '@vueuse/core'
import type { Fn } from '@vueuse/core'

import { useSharedCanvasPositionConversion } from '@/composables/element/useCanvasPositionConversion'
import type { LGraph } from '@/lib/litegraph/src/LGraph'
import type { LGraphNode, NodeId } from '@/lib/litegraph/src/LGraphNode'
import { LLink } from '@/lib/litegraph/src/LLink'
import type { Reroute } from '@/lib/litegraph/src/Reroute'
import type { RenderLink } from '@/lib/litegraph/src/canvas/RenderLink'
import type {
  INodeInputSlot,
  INodeOutputSlot
} from '@/lib/litegraph/src/interfaces'
import { LinkDirection } from '@/lib/litegraph/src/types/globalEnums'
import {
  clearCanvasPointerHistory,
  toCanvasPointerEvent
} from '@/renderer/core/canvas/interaction/canvasPointerEvent'
import { createLinkConnectorAdapter } from '@/renderer/core/canvas/links/linkConnectorAdapter'
import type { LinkConnectorAdapter } from '@/renderer/core/canvas/links/linkConnectorAdapter'
import {
  resolveNodeSurfaceSlotCandidate,
  resolveSlotTargetCandidate
} from '@/renderer/core/canvas/links/linkDropOrchestrator'
import { useSlotLinkDragUIState } from '@/renderer/core/canvas/links/slotLinkDragUIState'
import type { SlotDropCandidate } from '@/renderer/core/canvas/links/slotLinkDragUIState'
import { getSlotKey } from '@/renderer/core/layout/slots/slotIdentifier'
import { layoutStore } from '@/renderer/core/layout/store/layoutStore'
import type { Point } from '@/renderer/core/layout/types'
import { toPoint } from '@/renderer/core/layout/utils/geometry'
import { createSlotLinkDragContext } from '@/renderer/extensions/vueNodes/composables/slotLinkDragContext'
import { augmentToCanvasPointerEvent } from '@/renderer/extensions/vueNodes/utils/eventUtils'
import { app } from '@/scripts/app'
import { createRafBatch } from '@/utils/rafBatch'

interface SlotInteractionOptions {
  nodeId: string
  index: number
  type: 'input' | 'output'
}

interface SlotInteractionHandlers {
  onClick: (event: PointerEvent) => void
  onDoubleClick: (event: PointerEvent) => void
  onPointerDown: (event: PointerEvent) => void
}

interface PointerSession {
  begin: (pointerId: number) => void
  register: (...stops: Array<Fn | null | undefined>) => void
  matches: (event: PointerEvent) => boolean
  isActive: () => boolean
  clear: () => void
}

function createPointerSession(): PointerSession {
  let pointerId: number | null = null
  let stops: Fn[] = []

  const begin = (id: number) => {
    pointerId = id
  }

  const register = (...newStops: Array<Fn | null | undefined>) => {
    for (const stop of newStops) {
      if (typeof stop === 'function') {
        stops.push(stop)
      }
    }
  }

  const matches = (event: PointerEvent) =>
    pointerId !== null && event.pointerId === pointerId

  const isActive = () => pointerId !== null

  const clear = () => {
    for (const stop of stops) {
      stop()
    }
    stops = []
    pointerId = null
  }

  return { begin, register, matches, isActive, clear }
}

/**
 * Resolves the actual DOM element under the pointer position.
 *
 * On touch/mobile devices, pointer events have "implicit pointer capture" -
 * event.target stays as the element where the touch started, not the element
 * currently under the pointer. This helper uses document.elementFromPoint()
 * to get the actual element under the pointer, falling back to the provided
 * fallback target if elementFromPoint returns null.
 *
 * @param clientX - The client X coordinate of the pointer
 * @param clientY - The client Y coordinate of the pointer
 * @param fallback - Fallback target to use if elementFromPoint returns null
 * @returns The resolved target element
 */
export function resolvePointerTarget(
  clientX: number,
  clientY: number,
  fallback: EventTarget | null
): EventTarget | null {
  return document.elementFromPoint(clientX, clientY) ?? fallback
}

export function useSlotLinkInteraction({
  nodeId,
  index,
  type
}: SlotInteractionOptions): SlotInteractionHandlers {
  const {
    state,
    beginDrag,
    endDrag,
    updatePointerPosition,
    setCandidate,
    setCompatibleForKey,
    clearCompatible
  } = useSlotLinkDragUIState()
  const conversion = useSharedCanvasPositionConversion()
  const pointerSession = createPointerSession()
  let activeAdapter: LinkConnectorAdapter | null = null

  // Per-drag drag-state context (non-reactive caches + RAF batching)
  const dragContext = createSlotLinkDragContext()

  const resolveRenderLinkSource = (link: RenderLink): Point | null => {
    if (link.fromReroute) {
      const rerouteLayout = layoutStore.getRerouteLayout(link.fromReroute.id)
      if (rerouteLayout) return rerouteLayout.position
      const [x, y] = link.fromReroute.pos
      return toPoint(x, y)
    }

    const nodeId = link.node.id
    if (nodeId != null) {
      const isInputFrom = link.toType === 'output'
      const key = getSlotKey(String(nodeId), link.fromSlotIndex, isInputFrom)
      const layout = layoutStore.getSlotLayout(key)
      if (layout) return layout.position
    }

    const pos = link.fromPos
    return toPoint(pos[0], pos[1])
  }

  const syncRenderLinkOrigins = () => {
    if (!activeAdapter) return
    for (const link of activeAdapter.renderLinks) {
      const origin = resolveRenderLinkSource(link)
      if (!origin) continue
      const x = origin.x
      const y = origin.y
      if (link.fromPos[0] !== x || link.fromPos[1] !== y) {
        link.fromPos[0] = x
        link.fromPos[1] = y
      }
    }
  }

  function hasCanConnectToReroute(
    link: RenderLink
  ): link is RenderLink & { canConnectToReroute: (r: Reroute) => boolean } {
    return 'canConnectToReroute' in link
  }

  type ToInputLink = RenderLink & { toType: 'input' }
  type ToOutputLink = RenderLink & { toType: 'output' }
  const isToInputLink = (link: RenderLink): link is ToInputLink =>
    link.toType === 'input'
  const isToOutputLink = (link: RenderLink): link is ToOutputLink =>
    link.toType === 'output'

  function connectLinksToInput(
    links: ReadonlyArray<RenderLink>,
    node: LGraphNode,
    inputSlot: INodeInputSlot
  ): boolean {
    const validCandidates = links
      .filter(isToInputLink)
      .filter((link) => link.canConnectToInput(node, inputSlot))

    for (const link of validCandidates) {
      link.connectToInput(node, inputSlot, activeAdapter?.linkConnector.events)
    }

    return validCandidates.length > 0
  }

  function connectLinksToOutput(
    links: ReadonlyArray<RenderLink>,
    node: LGraphNode,
    outputSlot: INodeOutputSlot
  ): boolean {
    const validCandidates = links
      .filter(isToOutputLink)
      .filter((link) => link.canConnectToOutput(node, outputSlot))

    for (const link of validCandidates) {
      link.connectToOutput(
        node,
        outputSlot,
        activeAdapter?.linkConnector.events
      )
    }

    return validCandidates.length > 0
  }

  const resolveLinkOrigin = (
    link: LLink | undefined
  ): { position: Point; direction: LinkDirection } | null => {
    if (!link) return null

    const slotKey = getSlotKey(String(link.origin_id), link.origin_slot, false)
    const layout = layoutStore.getSlotLayout(slotKey)
    if (!layout) return null

    return { position: { ...layout.position }, direction: LinkDirection.NONE }
  }

  const resolveExistingInputLinkAnchor = (
    graph: LGraph,
    inputSlot: INodeInputSlot | undefined
  ): { position: Point; direction: LinkDirection } | null => {
    if (!inputSlot) return null

    const directLink = graph.getLink(inputSlot.link)
    if (directLink) {
      const reroutes = LLink.getReroutes(graph, directLink)
      const lastReroute = reroutes.at(-1)
      if (lastReroute) {
        const rerouteLayout = layoutStore.getRerouteLayout(lastReroute.id)
        if (rerouteLayout) {
          return {
            position: { ...rerouteLayout.position },
            direction: LinkDirection.NONE
          }
        }

        const pos = lastReroute.pos
        if (pos) {
          return {
            position: toPoint(pos[0], pos[1]),
            direction: LinkDirection.NONE
          }
        }
      }

      const directAnchor = resolveLinkOrigin(directLink)
      if (directAnchor) return directAnchor
    }

    const floatingLinkIterator = inputSlot._floatingLinks?.values()
    const floatingLink = floatingLinkIterator
      ? floatingLinkIterator.next().value
      : undefined
    if (!floatingLink) return null

    if (floatingLink.parentId != null) {
      const rerouteLayout = layoutStore.getRerouteLayout(floatingLink.parentId)
      if (rerouteLayout) {
        return {
          position: { ...rerouteLayout.position },
          direction: LinkDirection.NONE
        }
      }

      const reroute = graph.getReroute(floatingLink.parentId)
      if (reroute) {
        return {
          position: toPoint(reroute.pos[0], reroute.pos[1]),
          direction: LinkDirection.NONE
        }
      }
    }

    return null
  }

  const cleanupInteraction = () => {
    if (state.pointerId != null) {
      clearCanvasPointerHistory(state.pointerId)
    }
    activeAdapter?.reset()
    pointerSession.clear()
    endDrag()
    activeAdapter = null
    raf.cancel()
    dragContext.dispose()
    clearCompatible()
  }

  const updatePointerState = (event: PointerEvent) => {
    const clientX = event.clientX
    const clientY = event.clientY
    const [canvasX, canvasY] = conversion.clientPosToCanvasPos([
      clientX,
      clientY
    ])

    updatePointerPosition(clientX, clientY, canvasX, canvasY)
  }

  const processPointerMoveFrame = () => {
    const data = dragContext.pendingPointerMove
    if (!data) return
    dragContext.pendingPointerMove = null

    const [canvasX, canvasY] = conversion.clientPosToCanvasPos([
      data.clientX,
      data.clientY
    ])
    updatePointerPosition(data.clientX, data.clientY, canvasX, canvasY)

    syncRenderLinkOrigins()

    let hoveredSlotKey: string | null = null
    let hoveredNodeId: NodeId | null = null
    const target = resolvePointerTarget(data.clientX, data.clientY, data.target)
    if (target === dragContext.lastPointerEventTarget) {
      hoveredSlotKey = dragContext.lastPointerTargetSlotKey
      hoveredNodeId = dragContext.lastPointerTargetNodeId
    } else if (target instanceof HTMLElement) {
      const elWithSlot = target
        .closest('.lg-slot, .lg-node-widget')
        ?.querySelector<HTMLElement>('[data-slot-key]')
      const elWithNode = target.closest<HTMLElement>('[data-node-id]')
      hoveredSlotKey = elWithSlot?.dataset['slotKey'] ?? null
      hoveredNodeId = elWithNode?.dataset['nodeId'] ?? null
      dragContext.lastPointerEventTarget = target
      dragContext.lastPointerTargetSlotKey = hoveredSlotKey
      dragContext.lastPointerTargetNodeId = hoveredNodeId
    }

    const hoverChanged =
      hoveredSlotKey !== dragContext.lastHoverSlotKey ||
      hoveredNodeId !== dragContext.lastHoverNodeId

    let candidate: SlotDropCandidate | null = state.candidate

    if (hoverChanged) {
      const adapter = activeAdapter
      const graph = app.canvas?.graph ?? null
      const context = { adapter, graph, session: dragContext }
      const slotCandidate = resolveSlotTargetCandidate(target, context)
      const nodeCandidate = resolveNodeSurfaceSlotCandidate(target, context)
      candidate = slotCandidate?.compatible ? slotCandidate : nodeCandidate
      dragContext.lastHoverSlotKey = hoveredSlotKey
      dragContext.lastHoverNodeId = hoveredNodeId

      if (slotCandidate) {
        const key = getSlotKey(
          slotCandidate.layout.nodeId,
          slotCandidate.layout.index,
          slotCandidate.layout.type === 'input'
        )
        setCompatibleForKey(key, !!slotCandidate.compatible)
      }
      if (nodeCandidate && !slotCandidate?.compatible) {
        const key = getSlotKey(
          nodeCandidate.layout.nodeId,
          nodeCandidate.layout.index,
          nodeCandidate.layout.type === 'input'
        )
        setCompatibleForKey(key, !!nodeCandidate.compatible)
      }
    }

    const newCandidate = candidate?.compatible ? candidate : null
    const newCandidateKey = newCandidate
      ? getSlotKey(
          newCandidate.layout.nodeId,
          newCandidate.layout.index,
          newCandidate.layout.type === 'input'
        )
      : null

    const candidateChanged = newCandidateKey !== dragContext.lastCandidateKey
    if (candidateChanged) {
      setCandidate(newCandidate)
      dragContext.lastCandidateKey = newCandidateKey
    }

    let snapPosChanged = false
    if (activeAdapter) {
      const snapX = newCandidate
        ? newCandidate.layout.position.x
        : state.pointer.canvas.x
      const snapY = newCandidate
        ? newCandidate.layout.position.y
        : state.pointer.canvas.y
      const currentSnap = activeAdapter.linkConnector.state.snapLinksPos
      snapPosChanged =
        !currentSnap || currentSnap[0] !== snapX || currentSnap[1] !== snapY
      if (snapPosChanged) {
        activeAdapter.linkConnector.state.snapLinksPos = [snapX, snapY]
      }
    }

    const shouldRedraw = candidateChanged || snapPosChanged
    if (shouldRedraw) app.canvas?.setDirty(true, true)
  }
  const raf = createRafBatch(processPointerMoveFrame)

  const handlePointerMove = (event: PointerEvent) => {
    if (!pointerSession.matches(event)) return
    event.stopPropagation()

    dragContext.pendingPointerMove = {
      clientX: event.clientX,
      clientY: event.clientY,
      target: event.target
    }
    raf.schedule()
  }

  // Attempt to finalize by connecting to a DOM slot candidate
  const tryConnectToCandidate = (
    candidate: SlotDropCandidate | null
  ): boolean => {
    if (!candidate?.compatible) return false
    const graph = app.canvas?.graph
    const adapter = activeAdapter
    if (!graph || !adapter) return false

    const nodeId: NodeId = candidate.layout.nodeId
    const targetNode = graph.getNodeById(nodeId)
    if (!targetNode) return false

    if (candidate.layout.type === 'input') {
      const inputSlot = targetNode.inputs?.[candidate.layout.index]
      return (
        !!inputSlot &&
        connectLinksToInput(adapter.renderLinks, targetNode, inputSlot)
      )
    }

    if (candidate.layout.type === 'output') {
      const outputSlot = targetNode.outputs?.[candidate.layout.index]
      return (
        !!outputSlot &&
        connectLinksToOutput(adapter.renderLinks, targetNode, outputSlot)
      )
    }

    return false
  }

  // Attempt to finalize by dropping on a reroute under the pointer
  const tryConnectViaRerouteAtPointer = (): boolean => {
    const rerouteLayout = layoutStore.queryRerouteAtPoint({
      x: state.pointer.canvas.x,
      y: state.pointer.canvas.y
    })
    const graph = app.canvas?.graph
    const adapter = activeAdapter
    if (!rerouteLayout || !graph || !adapter) return false

    const reroute = graph.getReroute(rerouteLayout.id)
    if (!reroute || !adapter.isRerouteValidDrop(reroute.id)) return false

    let didConnect = false

    const results = reroute.findTargetInputs() ?? []
    const maybeReroutes = reroute.getReroutes()
    if (results.length && maybeReroutes !== null) {
      const originalReroutes = maybeReroutes.slice(0, -1).reverse()
      for (const link of adapter.renderLinks) {
        if (!isToInputLink(link)) continue
        for (const result of results) {
          link.connectToRerouteInput(
            reroute,
            result,
            adapter.linkConnector.events,
            originalReroutes
          )
          didConnect = true
        }
      }
    }

    const sourceOutput = reroute.findSourceOutput()
    if (sourceOutput) {
      const { node, output } = sourceOutput
      for (const link of adapter.renderLinks) {
        if (!isToOutputLink(link)) continue
        if (hasCanConnectToReroute(link) && !link.canConnectToReroute(reroute))
          continue
        link.connectToRerouteOutput(
          reroute,
          node,
          output,
          adapter.linkConnector.events
        )
        didConnect = true
      }
    }

    return didConnect
  }

  const finishInteraction = (event: PointerEvent) => {
    if (!pointerSession.matches(event)) return
    const canvasEvent = toCanvasPointerEvent(event)
    canvasEvent.preventDefault()

    raf.flush()

    raf.flush()

    if (!state.source) {
      cleanupInteraction()
      app.canvas?.setDirty(true, true)
      return
    }

    const snappedCandidate = state.candidate?.compatible
      ? state.candidate
      : null

    const dropTarget = resolvePointerTarget(
      event.clientX,
      event.clientY,
      canvasEvent.target
    )
    const hasConnected = connectByPriority(dropTarget, snappedCandidate)

    if (!hasConnected && dropTarget === app.canvas?.canvas) {
      activeAdapter?.dropOnCanvas(canvasEvent)
    }

    cleanupInteraction()
    app.canvas?.setDirty(true, true)
  }

  const handlePointerUp = (event: PointerEvent) => {
    event.stopPropagation()
    finishInteraction(event)
  }

  const handlePointerCancel = (event: PointerEvent) => {
    if (!pointerSession.matches(event)) return

    raf.flush()
    toCanvasPointerEvent(event)
    cleanupInteraction()
    app.canvas?.setDirty(true, true)
  }

  function connectByPriority(
    target: EventTarget | null,
    snappedCandidate: SlotDropCandidate | null
  ): boolean {
    const adapter = activeAdapter
    const graph = app.canvas?.graph ?? null
    const context = { adapter, graph, session: dragContext }

    const attemptSnapped = () => tryConnectToCandidate(snappedCandidate)

    const domSlotCandidate = resolveSlotTargetCandidate(target, context)
    const attemptDomSlot = () => tryConnectToCandidate(domSlotCandidate)

    const nodeSurfaceSlotCandidate = resolveNodeSurfaceSlotCandidate(
      target,
      context
    )
    const attemptNodeSurface = () =>
      tryConnectToCandidate(nodeSurfaceSlotCandidate)
    const attemptReroute = () => tryConnectViaRerouteAtPointer()

    if (attemptSnapped()) return true
    if (attemptDomSlot()) return true
    if (attemptNodeSurface()) return true
    if (attemptReroute()) return true
    return false
  }

  const onPointerDown = (event: PointerEvent) => {
    if (event.button !== 0) return
    if (!nodeId) return
    if (pointerSession.isActive()) return
    event.preventDefault()
    event.stopPropagation()

    const canvas = app.canvas
    const graph = canvas?.graph
    if (!canvas || !graph) return

    activeAdapter = createLinkConnectorAdapter()
    if (!activeAdapter) return
    raf.cancel()
    dragContext.reset()

    const layout = layoutStore.getSlotLayout(
      getSlotKey(nodeId, index, type === 'input')
    )
    if (!layout) return

    const localNodeId: NodeId = nodeId
    const isInputSlot = type === 'input'
    const isOutputSlot = type === 'output'

    const resolvedNode = graph.getNodeById(localNodeId)
    const inputSlot = isInputSlot ? resolvedNode?.inputs?.[index] : undefined
    const outputSlot = isOutputSlot ? resolvedNode?.outputs?.[index] : undefined

    const ctrlOrMeta = event.ctrlKey || event.metaKey

    const inputLinkId = inputSlot?.link ?? null
    const inputFloatingCount = inputSlot?._floatingLinks?.size ?? 0
    const hasExistingInputLink = inputLinkId != null || inputFloatingCount > 0

    const outputLinkCount = outputSlot?.links?.length ?? 0
    const outputFloatingCount = outputSlot?._floatingLinks?.size ?? 0
    const hasExistingOutputLink = outputLinkCount > 0 || outputFloatingCount > 0

    const shouldBreakExistingInputLink =
      isInputSlot &&
      hasExistingInputLink &&
      ctrlOrMeta &&
      event.altKey &&
      !event.shiftKey

    const shouldBatchDisconnectOutputLinks =
      isOutputSlot &&
      hasExistingOutputLink &&
      ctrlOrMeta &&
      event.altKey &&
      !event.shiftKey

    const existingInputLink =
      isInputSlot && inputLinkId != null
        ? graph.getLink(inputLinkId)
        : undefined

    if (shouldBreakExistingInputLink && resolvedNode) {
      resolvedNode.disconnectInput(index, true)
    }

    if (shouldBatchDisconnectOutputLinks && resolvedNode) {
      resolvedNode.disconnectOutput(index)
      canvas.setDirty(true, true)
      event.preventDefault()
      event.stopPropagation()
      return
    }

    const baseDirection = isInputSlot
      ? (inputSlot?.dir ?? LinkDirection.LEFT)
      : (outputSlot?.dir ?? LinkDirection.RIGHT)

    const existingAnchor =
      isInputSlot && !shouldBreakExistingInputLink
        ? resolveExistingInputLinkAnchor(graph, inputSlot)
        : null

    const shouldMoveExistingOutput =
      isOutputSlot && event.shiftKey && hasExistingOutputLink

    const shouldMoveExistingInput =
      isInputSlot && !shouldBreakExistingInputLink && hasExistingInputLink

    if (isOutputSlot) {
      activeAdapter.beginFromOutput(localNodeId, index, {
        moveExisting: shouldMoveExistingOutput
      })
    } else {
      activeAdapter.beginFromInput(localNodeId, index, {
        moveExisting: shouldMoveExistingInput,
        layout
      })
    }

    if (shouldMoveExistingInput && existingInputLink) {
      existingInputLink._dragging = true
    }

    syncRenderLinkOrigins()

    const direction = existingAnchor?.direction ?? baseDirection
    const startPosition = existingAnchor?.position ?? {
      x: layout.position.x,
      y: layout.position.y
    }

    beginDrag(
      {
        nodeId,
        slotIndex: index,
        type,
        direction,
        position: startPosition,
        linkId: !shouldBreakExistingInputLink
          ? existingInputLink?.id
          : undefined,
        movingExistingOutput: shouldMoveExistingOutput
      },
      event.pointerId
    )

    pointerSession.begin(event.pointerId)

    toCanvasPointerEvent(event)
    updatePointerState(event)

    activeAdapter.linkConnector.state.snapLinksPos = [
      state.pointer.canvas.x,
      state.pointer.canvas.y
    ]

    pointerSession.register(
      useEventListener('pointermove', handlePointerMove, {
        capture: true
      }),
      useEventListener('pointerup', handlePointerUp, {
        capture: true
      }),
      useEventListener('pointercancel', handlePointerCancel, {
        capture: true
      })
    )
    const targetType: 'input' | 'output' = type === 'input' ? 'output' : 'input'
    const allKeys = layoutStore.getAllSlotKeys()
    clearCompatible()
    for (const key of allKeys) {
      const slotLayout = layoutStore.getSlotLayout(key)
      if (!slotLayout) continue
      if (slotLayout.type !== targetType) continue
      const idx = slotLayout.index
      const ok =
        targetType === 'input'
          ? activeAdapter.isInputValidDrop(slotLayout.nodeId, idx)
          : activeAdapter.isOutputValidDrop(slotLayout.nodeId, idx)
      setCompatibleForKey(key, ok)
    }
    canvas.setDirty(true, true)
  }

  tryOnScopeDispose(() => {
    if (pointerSession.isActive()) {
      cleanupInteraction()
    }
  })

  function onDoubleClick(e: PointerEvent) {
    const { graph } = app.canvas
    if (!graph) return
    const node = graph.getNodeById(nodeId)
    if (!node) return
    augmentToCanvasPointerEvent(e, node, app.canvas)
    node.onInputDblClick?.(index, e)
  }
  function onClick(e: PointerEvent) {
    const { graph } = app.canvas
    if (!graph) return
    const node = graph.getNodeById(nodeId)
    if (!node) return
    augmentToCanvasPointerEvent(e, node, app.canvas)
    node.onInputClick?.(index, e)
  }

  return {
    onClick,
    onDoubleClick,
    onPointerDown
  }
}
