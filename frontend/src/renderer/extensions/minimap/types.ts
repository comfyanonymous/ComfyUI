/**
 * Minimap-specific type definitions
 */
import type { LGraph } from '@/lib/litegraph/src/litegraph'
import type { NodeId } from '@/platform/workflow/validation/schemas/workflowSchema'

/**
 * Minimal interface for what the minimap needs from the canvas
 */
export interface MinimapCanvas {
  canvas: HTMLCanvasElement
  ds: {
    scale: number
    offset: [number, number]
  }
  graph?: LGraph | null
  setDirty: (fg?: boolean, bg?: boolean) => void
}

export interface MinimapRenderContext {
  bounds: {
    minX: number
    minY: number
    width: number
    height: number
  }
  scale: number
  settings: MinimapRenderSettings
  width: number
  height: number
}

interface MinimapRenderSettings {
  nodeColors: boolean
  showLinks: boolean
  showGroups: boolean
  renderBypass: boolean
  renderError: boolean
}

export interface MinimapBounds {
  minX: number
  minY: number
  maxX: number
  maxY: number
  width: number
  height: number
}

export interface ViewportTransform {
  x: number
  y: number
  width: number
  height: number
}

export interface UpdateFlags {
  bounds: boolean
  nodes: boolean
  connections: boolean
  viewport: boolean
}

export type MinimapSettingsKey =
  | 'Comfy.Minimap.NodeColors'
  | 'Comfy.Minimap.ShowLinks'
  | 'Comfy.Minimap.ShowGroups'
  | 'Comfy.Minimap.RenderBypassState'
  | 'Comfy.Minimap.RenderErrorState'

/**
 * Node data required for minimap rendering
 */
export interface MinimapNodeData {
  id: NodeId
  x: number
  y: number
  width: number
  height: number
  bgcolor?: string
  mode?: number
  hasErrors?: boolean
}

/**
 * Link data required for minimap rendering
 */
export interface MinimapLinkData {
  sourceNode: MinimapNodeData
  targetNode: MinimapNodeData
  sourceSlot: number
  targetSlot: number
}

/**
 * Group data required for minimap rendering
 */
export interface MinimapGroupData {
  x: number
  y: number
  width: number
  height: number
  color?: string
}

/**
 * Interface for minimap data sources (Dependency Inversion Principle)
 */
export interface IMinimapDataSource {
  getNodes(): MinimapNodeData[]
  getLinks(): MinimapLinkData[]
  getGroups(): MinimapGroupData[]
  getBounds(): MinimapBounds
  getNodeCount(): number
  hasData(): boolean
}
