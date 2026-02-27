import { t } from '@/i18n'
import { drawTextInArea } from '@/lib/litegraph/src/draw'
import { Rectangle } from '@/lib/litegraph/src/infrastructure/Rectangle'
import type { Point } from '@/lib/litegraph/src/interfaces'
import type { NodeId } from '@/lib/litegraph/src/LGraphNode'
import type {
  CanvasPointer,
  LGraphCanvas,
  LGraphNode,
  Size
} from '@/lib/litegraph/src/litegraph'
import { LiteGraph } from '@/lib/litegraph/src/litegraph'
import type { CanvasPointerEvent } from '@/lib/litegraph/src/types/events'
import type {
  IBaseWidget,
  NodeBindable,
  TWidgetType
} from '@/lib/litegraph/src/types/widgets'
import { usePromotionStore } from '@/stores/promotionStore'
import type { WidgetState } from '@/stores/widgetValueStore'
import { useWidgetValueStore } from '@/stores/widgetValueStore'

export interface DrawWidgetOptions {
  /** The width of the node where this widget will be displayed. */
  width: number
  /** Synonym for "low quality". */
  showText?: boolean
  /** When true, suppresses the promoted outline color (e.g. for projected copies on SubgraphNode). */
  suppressPromotedOutline?: boolean
  /** Transient image source for preview widgets rendered on behalf of another node (e.g. subgraph promotion). */
  previewImages?: HTMLImageElement[]
}

interface DrawTruncatingTextOptions extends DrawWidgetOptions {
  /** The canvas context to draw the text on. */
  ctx: CanvasRenderingContext2D
  /** The amount of padding to add to the left of the text. */
  leftPadding?: number
  /** The amount of padding to add to the right of the text. */
  rightPadding?: number
}

export interface WidgetEventOptions {
  e: CanvasPointerEvent
  node: LGraphNode
  canvas: LGraphCanvas
}

export abstract class BaseWidget<TWidget extends IBaseWidget = IBaseWidget>
  implements IBaseWidget, NodeBindable
{
  /** From node edge to widget edge */
  static margin = 15
  /** From widget edge to tip of arrow button */
  static arrowMargin = 6
  /** Arrow button width */
  static arrowWidth = 10
  /** Absolute minimum display width of widget values */
  static minValueWidth = 42
  /** Minimum gap between label and value */
  static labelValueGap = 5

  declare computedHeight?: number
  declare serialize?: boolean
  computeLayoutSize?(node: LGraphNode): {
    minHeight: number
    maxHeight?: number
    minWidth: number
    maxWidth?: number
  }

  private _node: LGraphNode
  /** The node that this widget belongs to. */
  get node() {
    return this._node
  }

  linkedWidgets?: IBaseWidget[]
  name: string
  options: TWidget['options']
  type: TWidget['type']
  y: number = 0
  last_y?: number
  width?: number
  computedDisabled?: boolean
  tooltip?: string

  private _state: Omit<WidgetState, 'nodeId'> &
    Partial<Pick<WidgetState, 'nodeId'>>

  get label(): string | undefined {
    return this._state.label
  }
  set label(value: string | undefined) {
    this._state.label = value
  }

  hidden?: boolean
  advanced?: boolean

  get disabled(): boolean | undefined {
    return this._state.disabled
  }
  set disabled(value: boolean | undefined) {
    this._state.disabled = value ?? false
  }

  element?: HTMLElement
  callback?(
    value: TWidget['value'],
    canvas?: LGraphCanvas,
    node?: LGraphNode,
    pos?: Point,
    e?: CanvasPointerEvent
  ): void
  mouse?(
    event: CanvasPointerEvent,
    pointerOffset: Point,
    node: LGraphNode
  ): boolean
  computeSize?(width?: number): Size
  onPointerDown?(
    pointer: CanvasPointer,
    node: LGraphNode,
    canvas: LGraphCanvas
  ): boolean

  get value(): TWidget['value'] {
    return this._state.value as TWidget['value']
  }
  set value(value: TWidget['value']) {
    this._state.value = value
  }

  /**
   * Associates this widget with a node ID and registers it in the WidgetValueStore.
   * Once set, value reads/writes will be delegated to the store.
   */
  setNodeId(nodeId: NodeId): void {
    const graphId = this.node.graph?.rootGraph.id
    if (!graphId) return

    this._state = useWidgetValueStore().registerWidget(graphId, {
      ...this._state,
      // BaseWidget: this.value getter returns this._state.value. So value: this.value === value: this._state.value.
      // BaseDOMWidgetImpl: this.value getter returns options.getValue?.() ?? ''. Resolves the correct initial value instead of undefined.
      // I.e., calls overriden getter -> options.getValue() -> correct value (https://github.com/Comfy-Org/ComfyUI_frontend/issues/9194).
      value: this.value,
      nodeId
    })
  }

  constructor(widget: TWidget & { node: LGraphNode })
  constructor(widget: TWidget, node: LGraphNode)
  constructor(widget: TWidget & { node: LGraphNode }, node?: LGraphNode) {
    // Private fields
    this._node = node ?? widget.node

    // The set and get functions for DOM widget values are hacked on to the options object;
    // attempting to set value before options will throw.
    // https://github.com/Comfy-Org/ComfyUI_frontend/blob/df86da3d672628a452baed3df3347a52c0c8d378/src/scripts/domWidget.ts#L125
    this.name = widget.name
    this.options = widget.options
    this.type = widget.type

    // `node` has no setter - Object.assign will throw.
    // TODO: Resolve this workaround. Ref: https://github.com/Comfy-Org/litegraph.js/issues/1022
    const {
      node: _,
      // @ts-expect-error Prevent naming conflicts with custom nodes.
      outline_color,
      // @ts-expect-error Prevent naming conflicts with custom nodes.
      background_color,
      // @ts-expect-error Prevent naming conflicts with custom nodes.
      height,
      // @ts-expect-error Prevent naming conflicts with custom nodes.
      text_color,
      // @ts-expect-error Prevent naming conflicts with custom nodes.
      secondary_text_color,
      // @ts-expect-error Prevent naming conflicts with custom nodes.
      disabledTextColor,
      // @ts-expect-error Prevent naming conflicts with custom nodes.
      displayName,
      // @ts-expect-error Prevent naming conflicts with custom nodes.
      displayValue,
      // @ts-expect-error Prevent naming conflicts with custom nodes.
      labelBaseline,
      label,
      disabled,
      value,
      linkedWidgets,
      ...safeValues
    } = widget

    Object.assign(this, safeValues)

    this._state = {
      name: this.name,
      type: this.type as TWidgetType,
      value,
      label,
      disabled: disabled ?? false,
      serialize: this.serialize,
      options: this.options
    }
  }

  getOutlineColor(suppressPromotedOutline = false) {
    const graphId = this.node.graph?.rootGraph.id
    if (
      graphId &&
      !suppressPromotedOutline &&
      usePromotionStore().isPromotedByAny(
        graphId,
        String(this.node.id),
        this.name
      )
    )
      return LiteGraph.WIDGET_PROMOTED_OUTLINE_COLOR
    return this.advanced
      ? LiteGraph.WIDGET_ADVANCED_OUTLINE_COLOR
      : LiteGraph.WIDGET_OUTLINE_COLOR
  }

  get outline_color() {
    return this.getOutlineColor()
  }

  get background_color() {
    return LiteGraph.WIDGET_BGCOLOR
  }

  get height() {
    return LiteGraph.NODE_WIDGET_HEIGHT
  }

  get text_color() {
    return LiteGraph.WIDGET_TEXT_COLOR
  }

  get secondary_text_color() {
    return LiteGraph.WIDGET_SECONDARY_TEXT_COLOR
  }

  get disabledTextColor() {
    return LiteGraph.WIDGET_DISABLED_TEXT_COLOR
  }

  get displayName() {
    return this.label || this.name
  }

  // TODO: Resolve this workaround. Ref: https://github.com/Comfy-Org/litegraph.js/issues/1022
  get _displayValue(): string {
    return this.computedDisabled ? '' : String(this.value)
  }

  get labelBaseline() {
    return this.y + this.height * 0.7
  }

  /**
   * Draws the widget
   * @param ctx The canvas context
   * @param options The options for drawing the widget
   * @remarks Not naming this `draw` as `draw` conflicts with the `draw` method in
   * custom widgets.
   */
  abstract drawWidget(
    ctx: CanvasRenderingContext2D,
    options: DrawWidgetOptions
  ): void

  /**
   * Draws the standard widget shape - elongated capsule. The path of the widget shape is not
   * cleared, and may be used for further drawing.
   * @param ctx The canvas context
   * @param options The options for drawing the widget
   * @remarks Leaves {@link ctx} dirty.
   */
  protected drawWidgetShape(
    ctx: CanvasRenderingContext2D,
    { width, showText, suppressPromotedOutline }: DrawWidgetOptions
  ): void {
    const { height, y } = this
    const { margin } = BaseWidget

    ctx.textAlign = 'left'
    ctx.strokeStyle = this.getOutlineColor(suppressPromotedOutline)
    ctx.fillStyle = this.background_color
    ctx.beginPath()

    if (showText) {
      ctx.roundRect(margin, y, width - margin * 2, height, [height * 0.5])
    } else {
      ctx.rect(margin, y, width - margin * 2, height)
    }
    ctx.fill()
    if (showText && !this.computedDisabled) ctx.stroke()
  }

  /**
   * Draws a placeholder for widgets that only have a Vue implementation.
   * @param ctx The canvas context
   * @param options The options for drawing the widget
   * @param label The label to display (e.g., "ImageCrop", "BoundingBox")
   */
  protected drawVueOnlyWarning(
    ctx: CanvasRenderingContext2D,
    { width, suppressPromotedOutline }: DrawWidgetOptions,
    label: string
  ): void {
    const { y, height } = this

    ctx.save()

    ctx.fillStyle = this.background_color
    ctx.fillRect(15, y, width - 30, height)

    ctx.strokeStyle = this.getOutlineColor(suppressPromotedOutline)
    ctx.strokeRect(15, y, width - 30, height)

    ctx.fillStyle = this.text_color
    ctx.font = '11px monospace'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'

    ctx.fillText(
      `${label}: ${t('widgets.node2only')}`,
      width / 2,
      y + height / 2
    )

    ctx.restore()
  }

  /**
   * A shared routine for drawing a label and value as text, truncated
   * if they exceed the available width.
   */
  protected drawTruncatingText({
    ctx,
    width,
    leftPadding = 5,
    rightPadding = 20
  }: DrawTruncatingTextOptions): void {
    const { height, y } = this
    const { margin } = BaseWidget

    // Measure label and value
    const { displayName, _displayValue } = this
    const labelWidth = ctx.measureText(displayName).width
    const valueWidth = ctx.measureText(_displayValue).width

    const gap = BaseWidget.labelValueGap
    const x = margin * 2 + leftPadding

    const totalWidth = width - x - 2 * margin - rightPadding
    const requiredWidth = labelWidth + gap + valueWidth

    const area = new Rectangle(x, y, totalWidth, height * 0.7)

    ctx.fillStyle = this.secondary_text_color

    if (requiredWidth <= totalWidth) {
      // Draw label & value normally
      drawTextInArea({ ctx, text: displayName, area, align: 'left' })
    } else if (LiteGraph.truncateWidgetTextEvenly) {
      // Label + value will not fit - scale evenly to fit
      const scale = (totalWidth - gap) / (requiredWidth - gap)
      area.width = labelWidth * scale

      drawTextInArea({ ctx, text: displayName, area, align: 'left' })

      // Move the area to the right to render the value
      area.right = x + totalWidth
      area.setWidthRightAnchored(valueWidth * scale)
    } else if (LiteGraph.truncateWidgetValuesFirst) {
      // Label + value will not fit - use legacy scaling of value first
      const cappedLabelWidth = Math.min(labelWidth, totalWidth)

      area.width = cappedLabelWidth
      drawTextInArea({ ctx, text: displayName, area, align: 'left' })

      area.right = x + totalWidth
      area.setWidthRightAnchored(
        Math.max(totalWidth - gap - cappedLabelWidth, 0)
      )
    } else {
      // Label + value will not fit - scale label first
      const cappedValueWidth = Math.min(valueWidth, totalWidth)

      area.width = Math.max(totalWidth - gap - cappedValueWidth, 0)
      drawTextInArea({ ctx, text: displayName, area, align: 'left' })

      area.right = x + totalWidth
      area.setWidthRightAnchored(cappedValueWidth)
    }
    ctx.fillStyle = this.text_color
    drawTextInArea({ ctx, text: _displayValue, area, align: 'right' })
  }

  /**
   * Handles the click event for the widget
   * @param options The options for handling the click event
   */
  abstract onClick(options: WidgetEventOptions): void

  /**
   * Handles the drag event for the widget
   * @param options The options for handling the drag event
   */
  onDrag?(options: WidgetEventOptions): void

  /**
   * Sets the value of the widget
   * @param value The value to set
   * @param options The options for setting the value
   */
  setValue(
    value: TWidget['value'],
    { e, node, canvas }: WidgetEventOptions
  ): void {
    const oldValue = this.value
    if (value === this.value) return

    const v = this.type === 'number' ? Number(value) : value
    this.value = v
    if (
      this.options?.property &&
      node.properties[this.options.property] !== undefined
    ) {
      node.setProperty(this.options.property, v)
    }
    const pos = canvas.graph_mouse
    this.callback?.(this.value, canvas, node, pos, e)

    node.onWidgetChanged?.(this.name ?? '', v, oldValue, this)
    if (node.graph) node.graph._version++
  }

  /**
   * Clones the widget.
   * @param node The node that will own the cloned widget.
   * @returns A new widget with the same properties as the original
   * @remarks Subclasses with custom constructors must override this method.
   *
   * Correctly and safely typing this is currently not possible (practical?) in TypeScript 5.8.
   */
  createCopyForNode(node: LGraphNode): this {
    // @ts-expect-error - Constructor type casting for widget cloning
    const cloned: this = new (this.constructor as typeof this)(this, node)
    cloned.value = this.value
    return cloned
  }
}
