import type { LGraph } from '@/lib/litegraph/src/litegraph'
import { LGraphEventMode } from '@/lib/litegraph/src/litegraph'
import { useColorPaletteStore } from '@/stores/workspace/colorPaletteStore'
import { adjustColor } from '@/utils/colorUtil'

import { MinimapDataSourceFactory } from './data/MinimapDataSourceFactory'
import type {
  IMinimapDataSource,
  MinimapNodeData,
  MinimapRenderContext
} from './types'

/**
 * Get theme-aware colors for the minimap
 */
function getMinimapColors() {
  const colorPaletteStore = useColorPaletteStore()
  const isLightTheme = colorPaletteStore.completedActivePalette.light_theme

  return {
    nodeColor: isLightTheme ? '#3DA8E099' : '#0B8CE999',
    nodeColorDefault: isLightTheme ? '#D9D9D9' : '#353535',
    linkColor: isLightTheme ? '#616161' : '#B3B3B3',
    slotColor: isLightTheme ? '#616161' : '#B3B3B3',
    groupColor: isLightTheme ? '#A2D3EC' : '#1F547A',
    groupColorDefault: isLightTheme ? '#283640' : '#B3C1CB',
    bypassColor: isLightTheme ? '#DBDBDB' : '#4B184B',
    errorColor: '#FF0000',
    isLightTheme
  }
}

/**
 * Get node color based on settings and node properties (Single Responsibility)
 */
function getNodeColor(
  node: MinimapNodeData,
  settings: MinimapRenderContext['settings'],
  colors: ReturnType<typeof getMinimapColors>
): string {
  if (settings.renderBypass && node.mode === LGraphEventMode.BYPASS) {
    return colors.bypassColor
  }

  if (settings.nodeColors) {
    if (node.bgcolor) {
      return colors.isLightTheme
        ? adjustColor(node.bgcolor, { lightness: 0.5 })
        : node.bgcolor
    }
    return colors.nodeColorDefault
  }

  return colors.nodeColor
}

/**
 * Render groups on the minimap
 */
function renderGroups(
  ctx: CanvasRenderingContext2D,
  dataSource: IMinimapDataSource,
  offsetX: number,
  offsetY: number,
  context: MinimapRenderContext,
  colors: ReturnType<typeof getMinimapColors>
) {
  const groups = dataSource.getGroups()
  if (groups.length === 0) return

  for (const group of groups) {
    const x = (group.x - context.bounds.minX) * context.scale + offsetX
    const y = (group.y - context.bounds.minY) * context.scale + offsetY
    const w = group.width * context.scale
    const h = group.height * context.scale

    let color = colors.groupColor

    if (context.settings.nodeColors) {
      color = group.color ?? colors.groupColorDefault

      if (colors.isLightTheme) {
        color = adjustColor(color, { opacity: 0.5 })
      }
    }

    ctx.fillStyle = color
    ctx.fillRect(x, y, w, h)
  }
}

/**
 * Render nodes on the minimap with performance optimizations
 */
function renderNodes(
  ctx: CanvasRenderingContext2D,
  dataSource: IMinimapDataSource,
  offsetX: number,
  offsetY: number,
  context: MinimapRenderContext,
  colors: ReturnType<typeof getMinimapColors>
) {
  const nodes = dataSource.getNodes()
  if (nodes.length === 0) return

  // Group nodes by color for batch rendering (performance optimization)
  const nodesByColor = new Map<
    string,
    Array<{ x: number; y: number; w: number; h: number; hasErrors?: boolean }>
  >()

  for (const node of nodes) {
    const x = (node.x - context.bounds.minX) * context.scale + offsetX
    const y = (node.y - context.bounds.minY) * context.scale + offsetY
    const w = node.width * context.scale
    const h = node.height * context.scale

    const color = getNodeColor(node, context.settings, colors)

    if (!nodesByColor.has(color)) {
      nodesByColor.set(color, [])
    }

    nodesByColor.get(color)!.push({ x, y, w, h, hasErrors: node.hasErrors })
  }

  // Batch render nodes by color
  for (const [color, nodes] of nodesByColor) {
    ctx.fillStyle = color
    for (const node of nodes) {
      ctx.fillRect(node.x, node.y, node.w, node.h)
    }
  }

  // Render error outlines if needed
  if (context.settings.renderError) {
    ctx.strokeStyle = colors.errorColor
    ctx.lineWidth = 0.3
    for (const nodes of nodesByColor.values()) {
      for (const node of nodes) {
        if (node.hasErrors) {
          ctx.strokeRect(node.x, node.y, node.w, node.h)
        }
      }
    }
  }
}

/**
 * Render connections on the minimap
 */
function renderConnections(
  ctx: CanvasRenderingContext2D,
  dataSource: IMinimapDataSource,
  offsetX: number,
  offsetY: number,
  context: MinimapRenderContext,
  colors: ReturnType<typeof getMinimapColors>
) {
  const links = dataSource.getLinks()
  if (links.length === 0) return

  ctx.strokeStyle = colors.linkColor
  ctx.lineWidth = 0.3

  const slotRadius = Math.max(context.scale, 0.5)
  const connections: Array<{
    x1: number
    y1: number
    x2: number
    y2: number
  }> = []

  for (const link of links) {
    const x1 =
      (link.sourceNode.x - context.bounds.minX) * context.scale + offsetX
    const y1 =
      (link.sourceNode.y - context.bounds.minY) * context.scale + offsetY
    const x2 =
      (link.targetNode.x - context.bounds.minX) * context.scale + offsetX
    const y2 =
      (link.targetNode.y - context.bounds.minY) * context.scale + offsetY

    const outputX = x1 + link.sourceNode.width * context.scale
    const outputY = y1 + link.sourceNode.height * context.scale * 0.2
    const inputX = x2
    const inputY = y2 + link.targetNode.height * context.scale * 0.2

    // Draw connection line
    ctx.beginPath()
    ctx.moveTo(outputX, outputY)
    ctx.lineTo(inputX, inputY)
    ctx.stroke()

    connections.push({ x1: outputX, y1: outputY, x2: inputX, y2: inputY })
  }

  // Render connection slots on top
  ctx.fillStyle = colors.slotColor
  for (const conn of connections) {
    // Output slot
    ctx.beginPath()
    ctx.arc(conn.x1, conn.y1, slotRadius, 0, Math.PI * 2)
    ctx.fill()

    // Input slot
    ctx.beginPath()
    ctx.arc(conn.x2, conn.y2, slotRadius, 0, Math.PI * 2)
    ctx.fill()
  }
}

/**
 * Render a graph to a minimap canvas
 */
export function renderMinimapToCanvas(
  canvas: HTMLCanvasElement,
  graph: LGraph,
  context: MinimapRenderContext
) {
  const ctx = canvas.getContext('2d')
  if (!ctx) return

  // Clear canvas
  ctx.clearRect(0, 0, context.width, context.height)

  // Create unified data source (Dependency Inversion)
  const dataSource = MinimapDataSourceFactory.create(graph)

  // Fast path for empty graph
  if (!dataSource.hasData()) {
    return
  }

  const colors = getMinimapColors()
  const offsetX = (context.width - context.bounds.width * context.scale) / 2
  const offsetY = (context.height - context.bounds.height * context.scale) / 2

  // Render in correct order: groups -> links -> nodes
  if (context.settings.showGroups) {
    renderGroups(ctx, dataSource, offsetX, offsetY, context, colors)
  }

  if (context.settings.showLinks) {
    renderConnections(ctx, dataSource, offsetX, offsetY, context, colors)
  }

  renderNodes(ctx, dataSource, offsetX, offsetY, context, colors)
}
