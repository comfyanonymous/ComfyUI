import type { LGraphNode, NodeId } from '@/lib/litegraph/src/LGraphNode'
import type { LGraphCanvas } from '@/lib/litegraph/src/LGraphCanvas'
import type { CanvasPointer } from '@/lib/litegraph/src/CanvasPointer'
import type { Point } from '@/lib/litegraph/src/interfaces'
import type { CanvasPointerEvent } from '@/lib/litegraph/src/types/events'
import type { IBaseWidget } from '@/lib/litegraph/src/types/widgets'
import type { SubgraphNode } from '@/lib/litegraph/src/subgraph/SubgraphNode'
import type { BaseWidget } from '@/lib/litegraph/src/widgets/BaseWidget'
import { toConcreteWidget } from '@/lib/litegraph/src/widgets/widgetMap'
import { t } from '@/i18n'
import { useDomWidgetStore } from '@/stores/domWidgetStore'
import {
  stripGraphPrefix,
  useWidgetValueStore
} from '@/stores/widgetValueStore'

import type { PromotedWidgetView as IPromotedWidgetView } from './promotedWidgetTypes'

export type { PromotedWidgetView } from './promotedWidgetTypes'
export { isPromotedWidgetView } from './promotedWidgetTypes'

function resolve(
  subgraphNode: SubgraphNode,
  nodeId: string,
  widgetName: string
): { node: LGraphNode; widget: IBaseWidget } | undefined {
  const node = subgraphNode.subgraph.getNodeById(nodeId)
  if (!node) return undefined
  const widget = node.widgets?.find((w: IBaseWidget) => w.name === widgetName)
  return widget ? { node, widget } : undefined
}

function isWidgetValue(value: unknown): value is IBaseWidget['value'] {
  if (value === undefined) return true
  if (typeof value === 'string') return true
  if (typeof value === 'number') return true
  if (typeof value === 'boolean') return true
  return value !== null && typeof value === 'object'
}

type LegacyMouseWidget = IBaseWidget & {
  mouse: (e: CanvasPointerEvent, pos: Point, node: LGraphNode) => unknown
}

function hasLegacyMouse(widget: IBaseWidget): widget is LegacyMouseWidget {
  return 'mouse' in widget && typeof widget.mouse === 'function'
}

export function createPromotedWidgetView(
  subgraphNode: SubgraphNode,
  nodeId: string,
  widgetName: string,
  displayName?: string
): IPromotedWidgetView {
  return new PromotedWidgetView(subgraphNode, nodeId, widgetName, displayName)
}

class PromotedWidgetView implements IPromotedWidgetView {
  [symbol: symbol]: boolean

  readonly sourceNodeId: string
  readonly sourceWidgetName: string

  readonly serialize = false

  last_y?: number
  computedHeight?: number

  private readonly graphId: string
  private readonly bareNodeId: NodeId
  private yValue = 0

  private projectedSourceNode?: LGraphNode
  private projectedSourceWidget?: IBaseWidget
  private projectedWidget?: BaseWidget

  constructor(
    private readonly subgraphNode: SubgraphNode,
    nodeId: string,
    widgetName: string,
    private readonly displayName?: string
  ) {
    this.sourceNodeId = nodeId
    this.sourceWidgetName = widgetName
    this.graphId = subgraphNode.rootGraph.id
    this.bareNodeId = stripGraphPrefix(nodeId)
  }

  get node(): SubgraphNode {
    return this.subgraphNode
  }

  get name(): string {
    return this.displayName ?? this.sourceWidgetName
  }

  get y(): number {
    return this.yValue
  }

  set y(value: number) {
    this.yValue = value
    this.syncDomOverride()
  }

  get computedDisabled(): false {
    return false
  }

  set computedDisabled(_value: boolean | undefined) {}

  get type(): IBaseWidget['type'] {
    return this.resolve()?.widget.type ?? 'button'
  }

  get options(): IBaseWidget['options'] {
    return this.resolve()?.widget.options ?? {}
  }

  get tooltip(): string | undefined {
    return this.resolve()?.widget.tooltip
  }

  get linkedWidgets(): IBaseWidget[] | undefined {
    return this.resolve()?.widget.linkedWidgets
  }

  get value(): IBaseWidget['value'] {
    const state = this.getWidgetState()
    if (state && isWidgetValue(state.value)) return state.value
    return this.resolve()?.widget.value
  }

  set value(value: IBaseWidget['value']) {
    const state = this.getWidgetState()
    if (state) {
      state.value = value
      return
    }

    const resolved = this.resolve()
    if (resolved && isWidgetValue(value)) {
      resolved.widget.value = value
    }
  }

  get label(): string | undefined {
    const state = this.getWidgetState()
    return state?.label ?? this.displayName ?? this.sourceWidgetName
  }

  set label(value: string | undefined) {
    const state = this.getWidgetState()
    if (state) state.label = value
  }

  get hidden(): boolean {
    return this.resolve()?.widget.hidden ?? false
  }

  get computeLayoutSize(): IBaseWidget['computeLayoutSize'] {
    const resolved = this.resolve()
    const computeLayoutSize = resolved?.widget.computeLayoutSize
    if (!computeLayoutSize) return undefined
    return (node: LGraphNode) => computeLayoutSize.call(resolved.widget, node)
  }

  get computeSize(): IBaseWidget['computeSize'] {
    const resolved = this.resolve()
    const computeSize = resolved?.widget.computeSize
    if (!computeSize) return undefined
    return (width?: number) => computeSize.call(resolved.widget, width)
  }

  draw(
    ctx: CanvasRenderingContext2D,
    _node: LGraphNode,
    widgetWidth: number,
    y: number,
    H: number,
    lowQuality?: boolean
  ): void {
    const resolved = this.resolve()
    if (!resolved) {
      drawDisconnectedPlaceholder(ctx, widgetWidth, y, H)
      return
    }

    if (isBaseDOMWidget(resolved.widget)) return this.syncDomOverride(resolved)

    const projected = this.getProjectedWidget(resolved)
    if (!projected || typeof projected.drawWidget !== 'function') return

    const originalY = projected.y
    const originalComputedHeight = projected.computedHeight

    projected.y = this.y
    projected.computedHeight = this.computedHeight
    projected.value = this.value

    projected.drawWidget(ctx, {
      width: widgetWidth,
      showText: !lowQuality,
      suppressPromotedOutline: true,
      previewImages: resolved.node.imgs
    })

    projected.y = originalY
    projected.computedHeight = originalComputedHeight
  }

  onPointerDown(
    pointer: CanvasPointer,
    _node: LGraphNode,
    canvas: LGraphCanvas
  ): boolean {
    const resolved = this.resolve()
    if (!resolved) return false

    const interior = resolved.widget
    if (typeof interior.onPointerDown === 'function') {
      const handled = interior.onPointerDown(pointer, this.subgraphNode, canvas)
      if (handled) return true
    }

    const concrete = toConcreteWidget(interior, this.subgraphNode, false)
    if (concrete)
      return this.bindConcretePointerHandlers(pointer, canvas, concrete)

    if (hasLegacyMouse(interior))
      return this.handleLegacyMouse(pointer, interior)

    return false
  }

  callback(
    value: unknown,
    canvas?: LGraphCanvas,
    node?: LGraphNode,
    pos?: Point,
    e?: CanvasPointerEvent
  ) {
    this.resolve()?.widget.callback?.(value, canvas, node, pos, e)
  }

  private resolve(): { node: LGraphNode; widget: IBaseWidget } | undefined {
    return resolve(this.subgraphNode, this.sourceNodeId, this.sourceWidgetName)
  }

  private getWidgetState() {
    return useWidgetValueStore().getWidget(
      this.graphId,
      this.bareNodeId,
      this.sourceWidgetName
    )
  }

  private getProjectedWidget(resolved: {
    node: LGraphNode
    widget: IBaseWidget
  }): BaseWidget | undefined {
    const shouldRebuild =
      !this.projectedWidget ||
      this.projectedSourceNode !== resolved.node ||
      this.projectedSourceWidget !== resolved.widget

    if (!shouldRebuild) return this.projectedWidget

    const concrete = toConcreteWidget(resolved.widget, resolved.node, false)
    if (!concrete) {
      this.projectedWidget = undefined
      this.projectedSourceNode = undefined
      this.projectedSourceWidget = undefined
      return undefined
    }

    this.projectedWidget = concrete.createCopyForNode(this.subgraphNode)
    this.projectedSourceNode = resolved.node
    this.projectedSourceWidget = resolved.widget
    return this.projectedWidget
  }

  private bindConcretePointerHandlers(
    pointer: CanvasPointer,
    canvas: LGraphCanvas,
    concrete: BaseWidget
  ): boolean {
    const downEvent = pointer.eDown
    if (!downEvent) return false

    pointer.onClick = () =>
      concrete.onClick({
        e: downEvent,
        node: this.subgraphNode,
        canvas
      })
    pointer.onDrag = (eMove) =>
      concrete.onDrag?.({
        e: eMove,
        node: this.subgraphNode,
        canvas
      })
    return true
  }

  private handleLegacyMouse(
    pointer: CanvasPointer,
    interior: LegacyMouseWidget
  ): boolean {
    const downEvent = pointer.eDown
    if (!downEvent) return false

    const downPosition: Point = [
      downEvent.canvasX - this.subgraphNode.pos[0],
      downEvent.canvasY - this.subgraphNode.pos[1]
    ]
    interior.mouse(downEvent, downPosition, this.subgraphNode)

    pointer.finally = () => {
      const upEvent = pointer.eUp
      if (!upEvent) return

      const upPosition: Point = [
        upEvent.canvasX - this.subgraphNode.pos[0],
        upEvent.canvasY - this.subgraphNode.pos[1]
      ]
      interior.mouse(upEvent, upPosition, this.subgraphNode)
    }

    return true
  }

  private syncDomOverride(
    resolved:
      | { node: LGraphNode; widget: IBaseWidget }
      | undefined = this.resolve()
  ) {
    if (!resolved || !isBaseDOMWidget(resolved.widget)) return
    useDomWidgetStore().setPositionOverride(resolved.widget.id, {
      node: this.subgraphNode,
      widget: this
    })
  }
}

/** Checks if a widget is a BaseDOMWidget (DOMWidget or ComponentWidget). */
function isBaseDOMWidget(
  widget: IBaseWidget
): widget is IBaseWidget & { id: string } {
  return 'id' in widget && ('element' in widget || 'component' in widget)
}

function drawDisconnectedPlaceholder(
  ctx: CanvasRenderingContext2D,
  width: number,
  y: number,
  H: number
) {
  ctx.save()
  ctx.fillStyle = '#333'
  ctx.fillRect(15, y, width - 30, H)
  ctx.fillStyle = '#999'
  ctx.font = '11px monospace'
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.fillText(t('subgraphStore.disconnected'), width / 2, y + H / 2)
  ctx.restore()
}
