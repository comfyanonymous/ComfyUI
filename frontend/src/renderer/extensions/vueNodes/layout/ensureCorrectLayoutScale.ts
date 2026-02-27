import { useVueFeatureFlags } from '@/composables/useVueFeatureFlags'
import type { LGraph, RendererType } from '@/lib/litegraph/src/LGraph'
import { createBounds } from '@/lib/litegraph/src/measure'
import { useSettingStore } from '@/platform/settings/settingStore'
import { layoutStore } from '@/renderer/core/layout/store/layoutStore'
import { useLayoutMutations } from '@/renderer/core/layout/operations/layoutMutations'
import { LayoutSource } from '@/renderer/core/layout/types'
import type { NodeBoundsUpdate } from '@/renderer/core/layout/types'
import { app as comfyApp } from '@/scripts/app'
import type { SubgraphInputNode } from '@/lib/litegraph/src/subgraph/SubgraphInputNode'
import type { SubgraphOutputNode } from '@/lib/litegraph/src/subgraph/SubgraphOutputNode'

const SCALE_FACTOR = 1.2

export function ensureCorrectLayoutScale(
  renderer: RendererType = 'LG',
  targetGraph?: LGraph
) {
  const autoScaleLayoutSetting = useSettingStore().get(
    'Comfy.VueNodes.AutoScaleLayout'
  )

  if (!autoScaleLayoutSetting) return

  const canvas = comfyApp.canvas
  const graph = targetGraph ?? canvas?.graph

  if (!graph?.nodes) return

  const { shouldRenderVueNodes } = useVueFeatureFlags()

  const needsUpscale = renderer === 'LG' && shouldRenderVueNodes.value
  const needsDownscale = renderer === 'Vue' && !shouldRenderVueNodes.value

  if (!needsUpscale && !needsDownscale) {
    // Don't scale, but ensure workflowRendererVersion is set for future checks
    graph.extra.workflowRendererVersion ??= renderer
    return
  }

  const lgBounds = createBounds(graph.nodes)

  if (!lgBounds) return

  const [originX, originY] = lgBounds

  const lgNodesById = new Map(graph.nodes.map((node) => [node.id, node]))

  const yjsMoveNodeUpdates: NodeBoundsUpdate[] = []

  const scaleFactor = needsUpscale ? SCALE_FACTOR : 1 / SCALE_FACTOR

  const onActiveGraph = !targetGraph || targetGraph === canvas?.graph

  //TODO: once we remove the need for LiteGraph.NODE_TITLE_HEIGHT in vue nodes we nned to remove everything here.
  for (const node of graph.nodes) {
    const lgNode = lgNodesById.get(node.id)
    if (!lgNode) continue

    const [oldX, oldY] = lgNode.pos

    const relativeX = oldX - originX
    const relativeY = oldY - originY

    const scaledX = originX + relativeX * scaleFactor
    const scaledY = originY + relativeY * scaleFactor

    const scaledWidth = lgNode.width * scaleFactor

    const scaledHeight = lgNode.size[1] * scaleFactor

    // Directly update LiteGraph node to ensure immediate consistency
    // Dont need to reference vue directly because the pos and dims are already in yjs
    lgNode.pos[0] = scaledX
    lgNode.pos[1] = scaledY
    lgNode.size[0] = scaledWidth
    lgNode.size[1] = scaledHeight

    // Track updates for layout store (only if this is the active graph)
    if (onActiveGraph) {
      yjsMoveNodeUpdates.push({
        nodeId: String(lgNode.id),
        bounds: {
          x: scaledX,
          y: scaledY,
          width: scaledWidth,
          height: scaledHeight
        }
      })
    }
  }

  if (onActiveGraph && yjsMoveNodeUpdates.length > 0) {
    layoutStore.setSource(LayoutSource.Canvas)
    layoutStore.batchUpdateNodeBounds(yjsMoveNodeUpdates)
  }

  for (const reroute of graph.reroutes.values()) {
    const [oldX, oldY] = reroute.pos

    const relativeX = oldX - originX
    const relativeY = oldY - originY

    const scaledX = originX + relativeX * scaleFactor
    const scaledY = originY + relativeY * scaleFactor

    reroute.pos = [scaledX, scaledY]

    if (onActiveGraph && shouldRenderVueNodes.value) {
      const layoutMutations = useLayoutMutations()
      layoutMutations.moveReroute(
        reroute.id,
        { x: scaledX, y: scaledY },
        { x: oldX, y: oldY }
      )
    }
  }

  if ('inputNode' in graph && 'outputNode' in graph) {
    const ioNodes = [
      graph.inputNode as SubgraphInputNode,
      graph.outputNode as SubgraphOutputNode
    ]
    for (const ioNode of ioNodes) {
      const [oldX, oldY] = ioNode.pos
      const [oldWidth, oldHeight] = ioNode.size

      const relativeX = oldX - originX
      const relativeY = oldY - originY

      const scaledX = originX + relativeX * scaleFactor
      const scaledY = originY + relativeY * scaleFactor

      const scaledWidth = oldWidth * scaleFactor
      const scaledHeight = oldHeight * scaleFactor

      ioNode.pos = [scaledX, scaledY]
      ioNode.size = [scaledWidth, scaledHeight]
    }
  }

  graph.groups.forEach((group) => {
    const [oldX, oldY] = group.pos
    const [oldWidth, oldHeight] = group.size

    const relativeX = oldX - originX
    const relativeY = oldY - originY

    const scaledX = originX + relativeX * scaleFactor
    const scaledY = originY + relativeY * scaleFactor

    const scaledWidth = oldWidth * scaleFactor
    const scaledHeight = oldHeight * scaleFactor

    group.pos = [scaledX, scaledY]
    group.size = [scaledWidth, scaledHeight]
  })

  if (onActiveGraph && canvas) {
    const originScreen = canvas.ds.convertOffsetToCanvas([originX, originY])
    canvas.ds.changeScale(canvas.ds.scale / scaleFactor, originScreen)
  }

  graph.extra.workflowRendererVersion = needsUpscale ? 'Vue' : 'LG'
}
