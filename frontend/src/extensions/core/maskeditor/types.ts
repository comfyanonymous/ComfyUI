export enum BrushShape {
  Arc = 'arc',
  Rect = 'rect'
}

export enum Tools {
  MaskPen = 'pen',
  PaintPen = 'rgbPaint',
  Eraser = 'eraser',
  MaskBucket = 'paintBucket',
  MaskColorFill = 'colorSelect'
}

export const allTools = [
  Tools.MaskPen,
  Tools.PaintPen,
  Tools.Eraser,
  Tools.MaskBucket,
  Tools.MaskColorFill
]

const allImageLayers = ['mask', 'rgb'] as const

export type ImageLayer = (typeof allImageLayers)[number]

export interface ToolInternalSettings {
  container: HTMLElement
  cursor?: string
  newActiveLayerOnSet?: ImageLayer
}

export enum CompositionOperation {
  SourceOver = 'source-over',
  DestinationOut = 'destination-out'
}

export enum MaskBlendMode {
  Black = 'black',
  White = 'white',
  Negative = 'negative'
}

export enum ColorComparisonMethod {
  Simple = 'simple',
  HSL = 'hsl',
  LAB = 'lab'
}

export interface Point {
  x: number
  y: number
}

export interface Offset {
  x: number
  y: number
}

export interface Brush {
  type: BrushShape
  size: number
  opacity: number
  hardness: number
  stepSize: number
}
