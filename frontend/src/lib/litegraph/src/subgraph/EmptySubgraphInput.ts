import type { LGraphNode } from '@/lib/litegraph/src/LGraphNode'
import type { LLink } from '@/lib/litegraph/src/LLink'
import type { RerouteId } from '@/lib/litegraph/src/Reroute'
import type { INodeInputSlot, Point } from '@/lib/litegraph/src/interfaces'
import { nextUniqueName } from '@/lib/litegraph/src/strings'
import { zeroUuid } from '@/lib/litegraph/src/utils/uuid'

import { SubgraphInput } from './SubgraphInput'
import type { SubgraphInputNode } from './SubgraphInputNode'

/**
 * A virtual slot that simply creates a new input slot when connected to.
 */
export class EmptySubgraphInput extends SubgraphInput {
  declare parent: SubgraphInputNode

  constructor(parent: SubgraphInputNode) {
    super(
      {
        id: zeroUuid,
        name: '',
        type: ''
      },
      parent
    )
  }

  override connect(
    slot: INodeInputSlot,
    node: LGraphNode,
    afterRerouteId?: RerouteId
  ): LLink | undefined {
    const { subgraph } = this.parent
    const existingNames = subgraph.inputs.map((x) => x.name)

    const name = nextUniqueName(slot.name, existingNames)
    const input = subgraph.addInput(name, String(slot.type))
    return input.connect(slot, node, afterRerouteId)
  }

  override get labelPos(): Point {
    const [x, y, , height] = this.boundingRect
    return [x, y + height * 0.5]
  }
}
