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
import { isSubgraphOutput } from '@/lib/litegraph/src/subgraph/subgraphUtils'

export class NodeOutputSlot extends NodeSlot implements INodeOutputSlot {
  links: LinkId[] | null
  _data?: unknown
  slot_index?: number

  get isWidgetInputSlot(): false {
    return false
  }

  get collapsedPos(): Readonly<Point> {
    return [
      this._node._collapsed_width ?? LiteGraph.NODE_COLLAPSED_WIDTH,
      LiteGraph.NODE_TITLE_HEIGHT * -0.5
    ]
  }

  constructor(
    slot: OptionalProps<INodeOutputSlot, 'boundingRect'>,
    node: LGraphNode
  ) {
    super(slot, node)
    this.links = slot.links
    this._data = slot._data
    this.slot_index = slot.slot_index
  }

  override isValidTarget(
    fromSlot: INodeInputSlot | INodeOutputSlot | SubgraphInput | SubgraphOutput
  ): boolean {
    if ('link' in fromSlot) {
      return LiteGraph.isValidConnection(this.type, fromSlot.type)
    }

    if (isSubgraphOutput(fromSlot)) {
      return LiteGraph.isValidConnection(this.type, fromSlot.type)
    }

    return false
  }

  override get isConnected(): boolean {
    return this.links != null && this.links.length > 0
  }

  override draw(
    ctx: CanvasRenderingContext2D,
    options: Omit<IDrawOptions, 'doStroke' | 'labelPosition'>
  ) {
    const { textAlign, strokeStyle } = ctx
    ctx.textAlign = 'right'
    ctx.strokeStyle = 'black'

    super.draw(ctx, {
      ...options,
      labelPosition: LabelPosition.Left,
      doStroke: true
    })

    ctx.textAlign = textAlign
    ctx.strokeStyle = strokeStyle
  }

  override toJSON(): INodeOutputSlot {
    return {
      ...super.toJSON(),
      links: this.links,
      slot_index: this.slot_index
    }
  }
}
