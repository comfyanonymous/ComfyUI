import type { LGraphNode } from '@/lib/litegraph/src/LGraphNode'
import type { LinkId } from '@/lib/litegraph/src/LLink'
import { LabelPosition } from '@/lib/litegraph/src/draw'
import type {
  INodeInputSlot,
  INodeOutputSlot,
  OptionalProps,
  Point
} from '@/lib/litegraph/src/interfaces'
import { LiteGraph } from '@/lib/litegraph/src/litegraph'
import { NodeSlot } from '@/lib/litegraph/src/node/NodeSlot'
import type { IDrawOptions } from '@/lib/litegraph/src/node/NodeSlot'
import type { SubgraphInput } from '@/lib/litegraph/src/subgraph/SubgraphInput'
import type { SubgraphOutput } from '@/lib/litegraph/src/subgraph/SubgraphOutput'
import { isSubgraphInput } from '@/lib/litegraph/src/subgraph/subgraphUtils'
import type { IBaseWidget } from '@/lib/litegraph/src/types/widgets'

export class NodeInputSlot extends NodeSlot implements INodeInputSlot {
  link: LinkId | null
  alwaysVisible?: boolean

  get isWidgetInputSlot(): boolean {
    return !!this.widget
  }

  private _widgetRef: WeakRef<IBaseWidget> | undefined

  /** Internal use only; API is not finalised and may change at any time. */
  get _widget(): IBaseWidget | undefined {
    return this._widgetRef?.deref()
  }

  set _widget(widget: IBaseWidget | undefined) {
    this._widgetRef = widget ? new WeakRef(widget) : undefined
  }

  get collapsedPos(): Readonly<Point> {
    return [0, LiteGraph.NODE_TITLE_HEIGHT * -0.5]
  }

  constructor(
    slot: OptionalProps<INodeInputSlot, 'boundingRect'>,
    node: LGraphNode
  ) {
    super(slot, node)
    this.link = slot.link
  }

  override get isConnected(): boolean {
    return this.link != null
  }

  override isValidTarget(
    fromSlot: INodeInputSlot | INodeOutputSlot | SubgraphInput | SubgraphOutput
  ): boolean {
    if ('links' in fromSlot) {
      return LiteGraph.isValidConnection(fromSlot.type, this.type)
    }

    if (isSubgraphInput(fromSlot)) {
      return LiteGraph.isValidConnection(fromSlot.type, this.type)
    }

    return false
  }

  override draw(
    ctx: CanvasRenderingContext2D,
    options: Omit<IDrawOptions, 'doStroke' | 'labelPosition'>
  ) {
    const { textAlign } = ctx
    ctx.textAlign = 'left'

    super.draw(ctx, {
      ...options,
      labelPosition: LabelPosition.Right,
      doStroke: false
    })

    ctx.textAlign = textAlign
  }

  override toJSON(): INodeInputSlot {
    return {
      ...super.toJSON(),
      link: this.link,
      widget: this.widget
    }
  }
}
