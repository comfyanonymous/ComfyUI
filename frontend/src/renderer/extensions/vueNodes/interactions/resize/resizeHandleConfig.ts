import type { CompassCorners } from '@/lib/litegraph/src/interfaces'

interface ResizeHandle {
  readonly corner: CompassCorners
  readonly positionClasses: string
  readonly cursorClass: string
  readonly i18nKey: string
  readonly svgPositionClasses: string
  readonly svgTransform: string
}

export const RESIZE_HANDLES: ResizeHandle[] = [
  {
    corner: 'SE',
    positionClasses: '-right-1 -bottom-1',
    cursorClass: 'cursor-se-resize',
    i18nKey: 'g.resizeFromBottomRight',
    svgPositionClasses: 'top-1 left-1',
    svgTransform: ''
  },
  {
    corner: 'NE',
    positionClasses: '-right-1 -top-1',
    cursorClass: 'cursor-ne-resize',
    i18nKey: 'g.resizeFromTopRight',
    svgPositionClasses: 'bottom-1 left-1',
    svgTransform: 'scaleY(-1)'
  },
  {
    corner: 'SW',
    positionClasses: '-left-1 -bottom-1',
    cursorClass: 'cursor-sw-resize',
    i18nKey: 'g.resizeFromBottomLeft',
    svgPositionClasses: 'top-1 right-1',
    svgTransform: 'scaleX(-1)'
  },
  {
    corner: 'NW',
    positionClasses: '-left-1 -top-1',
    cursorClass: 'cursor-nw-resize',
    i18nKey: 'g.resizeFromTopLeft',
    svgPositionClasses: 'bottom-1 right-1',
    svgTransform: 'scale(-1, -1)'
  }
] as const
