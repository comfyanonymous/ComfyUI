import _ from 'es-toolkit/compat'
import { type Component, toRaw } from 'vue'

import { useChainCallback } from '@/composables/functional/useChainCallback'
import {
  LGraphNode,
  LegacyWidget,
  LiteGraph
} from '@/lib/litegraph/src/litegraph'
import type {
  IBaseWidget,
  IWidgetOptions
} from '@/lib/litegraph/src/types/widgets'
import type { InputSpec } from '@/schemas/nodeDef/nodeDefSchemaV2'
import { useDomWidgetStore } from '@/stores/domWidgetStore'
import { usePromotionStore } from '@/stores/promotionStore'
import { generateUUID } from '@/utils/formatUtil'

export interface BaseDOMWidget<
  V extends object | string = object | string
> extends IBaseWidget<V, string, DOMWidgetOptions<V>> {
  // ICustomWidget properties
  type: string
  options: DOMWidgetOptions<V>
  value: V
  callback?: (value: V) => void

  // BaseDOMWidget properties
  /** The unique ID of the widget. */
  readonly id: string
  /** The node that the widget belongs to. */
  readonly node: LGraphNode
  /** Whether the widget is visible. */
  isVisible(): boolean
  /** The margin of the widget. */
  margin: number
}

/**
 * A DOM widget that wraps a custom HTML element as a litegraph widget.
 */
export interface DOMWidget<
  T extends HTMLElement,
  V extends object | string
> extends BaseDOMWidget<V> {
  element: T
  /**
   * @deprecated Legacy property used by some extensions for customtext
   * (textarea) widgets. Use {@link element} instead as it provides the same
   * functionality and works for all DOMWidget types.
   */
  inputEl?: T
}

/**
 * Additional props that can be passed to component widgets.
 * These are in addition to the standard props that are always provided:
 * - modelValue: The widget's value (handled by v-model)
 * - widget: Reference to the widget instance
 * - onUpdate:modelValue: The update handler for v-model
 */
type ComponentWidgetCustomProps = Record<string, unknown>

/**
 * Standard props that are handled separately by DomWidget.vue and should be
 * omitted when defining custom props for component widgets
 */
export type ComponentWidgetStandardProps =
  | 'modelValue'
  | 'widget'
  | 'onUpdate:modelValue'

/**
 * A DOM widget that wraps a Vue component as a litegraph widget.
 */
export interface ComponentWidget<
  V extends object | string,
  P extends ComponentWidgetCustomProps = ComponentWidgetCustomProps
> extends BaseDOMWidget<V> {
  readonly component: Component
  readonly inputSpec: InputSpec
  readonly props?: P
}

export interface DOMWidgetOptions<
  V extends object | string
> extends IWidgetOptions {
  /**
   * Whether to render a placeholder rectangle when zoomed out.
   */
  hideOnZoom?: boolean
  selectOn?: string[]
  onHide?: (widget: BaseDOMWidget<V>) => void
  getValue?: () => V
  setValue?: (value: V) => void
  getMinHeight?: () => number
  getMaxHeight?: () => number
  getHeight?: () => string | number
  onDraw?: (widget: BaseDOMWidget<V>) => void
  margin?: number
  /**
   * @deprecated Use `afterResize` instead. This callback is a legacy API
   * that fires before resize happens, but it is no longer supported. Now it
   * fires after resize happens.
   * The resize logic has been upstreamed to litegraph in
   * https://github.com/Comfy-Org/ComfyUI_frontend/pull/2557
   */
  beforeResize?: (this: BaseDOMWidget<V>, node: LGraphNode) => void
  afterResize?: (this: BaseDOMWidget<V>, node: LGraphNode) => void
}

export const isDOMWidget = <T extends HTMLElement, V extends object | string>(
  widget: IBaseWidget
): widget is DOMWidget<T, V> => 'element' in widget && !!widget.element

export const isComponentWidget = <V extends object | string>(
  widget: IBaseWidget
): widget is ComponentWidget<V> => 'component' in widget && !!widget.component

abstract class BaseDOMWidgetImpl<V extends object | string>
  extends LegacyWidget<IBaseWidget<V, string, DOMWidgetOptions<V>>>
  implements BaseDOMWidget<V>
{
  static readonly DEFAULT_MARGIN = 10
  declare readonly name: string
  declare readonly options: DOMWidgetOptions<V>
  declare callback?: (value: V) => void
  readonly promotionStore = usePromotionStore()

  readonly id: string

  constructor(obj: {
    node: LGraphNode
    name: string
    type: string
    options: DOMWidgetOptions<V>
  }) {
    const { node, name, type, options } = obj
    super({ y: 0, name, type, options }, node)

    this.id = generateUUID()
  }

  override get value(): V {
    return this.options.getValue?.() ?? ('' as V)
  }

  override set value(v: V) {
    this.options.setValue?.(v)
    this.callback?.(this.value)
  }

  get margin(): number {
    return this.options.margin ?? BaseDOMWidgetImpl.DEFAULT_MARGIN
  }

  isVisible(): boolean {
    return !this.hidden && this.node.isWidgetVisible(this)
  }

  override draw(
    ctx: CanvasRenderingContext2D,
    _node: LGraphNode,
    widget_width: number,
    y: number,
    widget_height: number,
    lowQuality?: boolean
  ): void {
    const isVisible = this.isVisible()

    if (this.options.hideOnZoom && lowQuality && isVisible) {
      // Draw a placeholder rectangle
      const originalFillStyle = ctx.fillStyle
      ctx.beginPath()
      ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR
      ctx.rect(
        this.margin,
        y + this.margin,
        widget_width - this.margin * 2,
        (this.computedHeight ?? widget_height) - 2 * this.margin
      )
      ctx.fill()
      ctx.fillStyle = originalFillStyle
    } else {
      if (!isVisible) {
        this.options.onDraw?.(this)
        return
      }

      const graphId = this.node.graph?.rootGraph.id
      const isPromoted =
        graphId &&
        this.promotionStore.isPromotedByAny(
          graphId,
          String(this.node.id),
          this.name
        )
      if (!isPromoted) {
        this.options.onDraw?.(this)
        return
      }

      ctx.save()
      const adjustedMargin = this.margin - 1
      ctx.beginPath()
      ctx.strokeStyle = LiteGraph.WIDGET_PROMOTED_OUTLINE_COLOR
      ctx.strokeRect(
        adjustedMargin,
        y + adjustedMargin,
        widget_width - adjustedMargin * 2,
        (this.computedHeight ?? widget_height) - 2 * adjustedMargin
      )
      ctx.restore()
    }
    this.options.onDraw?.(this)
  }

  override onRemove(): void {
    useDomWidgetStore().unregisterWidget(this.id)
  }

  override createCopyForNode(node: LGraphNode): this {
    // @ts-expect-error
    const cloned: this = new (this.constructor as typeof this)({
      node: node,
      name: this.name,
      type: this.type,
      options: this.options
    })
    cloned.value = this.value
    // Preserve the Y position from the original widget to maintain proper positioning
    // when widgets are promoted through subgraph nesting
    cloned.y = this.y
    return cloned
  }
}

export class DOMWidgetImpl<T extends HTMLElement, V extends object | string>
  extends BaseDOMWidgetImpl<V>
  implements DOMWidget<T, V>
{
  override readonly element: T

  constructor(obj: {
    node: LGraphNode
    name: string
    type: string
    element: T
    options: DOMWidgetOptions<V>
  }) {
    super(obj)
    this.element = obj.element
  }

  override createCopyForNode(node: LGraphNode): this {
    // @ts-expect-error
    const cloned: this = new (this.constructor as typeof this)({
      node: node,
      name: this.name,
      type: this.type,
      element: this.element, // Include the element!
      options: this.options
    })
    cloned.value = this.value
    // Preserve the Y position from the original widget to maintain proper positioning
    // when widgets are promoted through subgraph nesting
    cloned.y = this.y
    return cloned
  }

  /** Extract DOM widget size info */
  override computeLayoutSize(node: LGraphNode) {
    if (this.type === 'hidden') {
      return {
        minHeight: 0,
        maxHeight: 0,
        minWidth: 0
      }
    }

    const styles = getComputedStyle(this.element)
    let minHeight =
      this.options.getMinHeight?.() ??
      parseInt(styles.getPropertyValue('--comfy-widget-min-height'))
    let maxHeight =
      this.options.getMaxHeight?.() ??
      parseInt(styles.getPropertyValue('--comfy-widget-max-height'))

    let prefHeight: string | number =
      this.options.getHeight?.() ??
      styles.getPropertyValue('--comfy-widget-height')

    if (typeof prefHeight === 'string' && prefHeight.endsWith?.('%')) {
      prefHeight =
        node.size[1] *
        (parseFloat(prefHeight.substring(0, prefHeight.length - 1)) / 100)
    } else {
      prefHeight =
        typeof prefHeight === 'number' ? prefHeight : parseInt(prefHeight)

      if (isNaN(minHeight)) {
        minHeight = prefHeight
      }
    }

    return {
      minHeight: isNaN(minHeight) ? 50 : minHeight,
      maxHeight: isNaN(maxHeight) ? undefined : maxHeight,
      minWidth: 0
    }
  }
}

export class ComponentWidgetImpl<
  V extends object | string,
  P extends ComponentWidgetCustomProps = ComponentWidgetCustomProps
>
  extends BaseDOMWidgetImpl<V>
  implements ComponentWidget<V, P>
{
  readonly component: Component
  readonly inputSpec: InputSpec
  readonly props?: P

  constructor(obj: {
    node: LGraphNode
    name: string
    component: Component
    inputSpec: InputSpec
    props?: P
    options: DOMWidgetOptions<V>
    type?: string
  }) {
    super({
      type: 'custom',
      ...obj
    })
    this.component = obj.component
    this.inputSpec = obj.inputSpec
    this.props = obj.props
  }

  override computeLayoutSize() {
    const minHeight = this.options.getMinHeight?.() ?? 50
    const maxHeight = this.options.getMaxHeight?.()
    return {
      minHeight,
      maxHeight,
      minWidth: 0
    }
  }

  override serializeValue(): V {
    return toRaw(this.value)
  }
}

export const addWidget = <W extends BaseDOMWidget<object | string>>(
  node: LGraphNode,
  widget: W
) => {
  node.addCustomWidget(widget)

  if (node.graph) {
    useDomWidgetStore().registerWidget(widget)
  }

  node.onAdded = useChainCallback(node.onAdded, () => {
    useDomWidgetStore().registerWidget(widget)
  })

  node.onRemoved = useChainCallback(node.onRemoved, () => {
    widget.onRemove?.()
  })

  node.onResize = useChainCallback(node.onResize, () => {
    widget.options.beforeResize?.call(widget, node)
    widget.options.afterResize?.call(widget, node)
  })
}

LGraphNode.prototype.addDOMWidget = function <
  T extends HTMLElement,
  V extends object | string
>(
  this: LGraphNode,
  name: string,
  type: string,
  element: T,
  options: DOMWidgetOptions<V> = {}
): DOMWidget<T, V> {
  const widget = new DOMWidgetImpl({
    node: this,
    name,
    type,
    element,
    options: { hideOnZoom: true, ...options }
  })
  // Note: Before `LGraphNode.configure` is called, `this.id` is always `-1`.
  addWidget(this, widget as unknown as BaseDOMWidget<object | string>)

  // Workaround for https://github.com/Comfy-Org/ComfyUI_frontend/issues/2493
  // Some custom nodes are explicitly expecting getter and setter of `value`
  // property to be on instance instead of prototype.
  Object.defineProperty(widget, 'value', {
    get(this: DOMWidgetImpl<T, V>): V {
      return this.options.getValue?.() ?? ('' as V)
    },
    set(this: DOMWidgetImpl<T, V>, v: V) {
      this.options.setValue?.(v)
      this.callback?.(this.value)
    }
  })

  return widget
}
