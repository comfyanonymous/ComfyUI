import {
  ExecutableNodeDTO,
  LGraphEventMode
} from '@/lib/litegraph/src/litegraph'
import type {
  ExecutableLGraphNode,
  ISlotType,
  LGraphNode
} from '@/lib/litegraph/src/litegraph'

export const GROUP = Symbol()

export function isGroupNode(node: LGraphNode): boolean {
  return node.constructor?.nodeData?.[GROUP] !== undefined
}

export class ExecutableGroupNodeDTO extends ExecutableNodeDTO {
  override get isVirtualNode(): true {
    return true
  }

  override getInnerNodes(): ExecutableLGraphNode[] {
    return this.node.getInnerNodes?.(this.nodesByExecutionId) ?? []
  }

  override resolveOutput(slot: number, type: ISlotType, visited: Set<string>) {
    // Temporary duplication: Bypass nodes are bypassed using the first input with matching type
    if (this.mode === LGraphEventMode.BYPASS) {
      const { inputs } = this

      // Bypass nodes by finding first input with matching type
      const parentInputIndexes = Object.keys(inputs).map(Number)
      // Prioritise exact slot index
      const indexes = [slot, ...parentInputIndexes]
      const matchingIndex = indexes.find((i) => inputs[i]?.type === type)

      // No input types match
      if (matchingIndex === undefined) return

      return this.resolveInput(matchingIndex, visited)
    }

    const linkId = this.node.outputs[slot]?.links?.at(0)
    const link = this.node.graph?.getLink(linkId)
    if (!link) {
      throw new Error(
        `Failed to get link for group node ${this.node.id} with link ${linkId}`
      )
    }

    const updated = this.node.updateLink?.(link)
    if (!updated) {
      throw new Error(
        `Failed to update link for group node ${this.node.id} with link ${linkId}`
      )
    }

    const node = this.node
      .getInnerNodes?.(this.nodesByExecutionId)
      .find((node) => node.id === updated.origin_id)
    if (!node) {
      throw new Error(
        `Failed to get node for group node ${this.node.id} with link ${linkId}`
      )
    }

    return {
      node,
      origin_id: `${this.id}:${(updated.origin_id as string).split(':').at(-1)}`,
      origin_slot: updated.origin_slot
    }
  }
}
