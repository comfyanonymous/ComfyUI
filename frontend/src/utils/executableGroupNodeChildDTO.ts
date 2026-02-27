import type { GroupNodeHandler } from '@/extensions/core/groupNode'
import { ExecutableNodeDTO } from '@/lib/litegraph/src/litegraph'
import type {
  ExecutableLGraphNode,
  ExecutionId,
  LGraphNode,
  NodeId,
  SubgraphNode
} from '@/lib/litegraph/src/litegraph'

export class ExecutableGroupNodeChildDTO extends ExecutableNodeDTO {
  groupNodeHandler?: GroupNodeHandler

  constructor(
    /** The actual node that this DTO wraps. */
    node: LGraphNode | SubgraphNode,
    /** A list of subgraph instance node IDs from the root graph to the containing instance. @see {@link id} */
    subgraphNodePath: readonly NodeId[],
    /** A flattened map of all DTOs in this node network. Subgraph instances have been expanded into their inner nodes. */
    nodesByExecutionId: Map<ExecutionId, ExecutableLGraphNode>,
    /** The actual subgraph instance that contains this node, otherwise undefined. */
    subgraphNode?: SubgraphNode | undefined,
    groupNodeHandler?: GroupNodeHandler
  ) {
    super(node, subgraphNodePath, nodesByExecutionId, subgraphNode)
    this.groupNodeHandler = groupNodeHandler
  }

  override resolveInput(slot: number) {
    // Check if this group node is inside a subgraph (unsupported)
    if (this.id.split(':').length > 2) {
      throw new Error(
        'Group nodes inside subgraphs are not supported. Please convert the group node to a subgraph instead.'
      )
    }

    const inputNode = this.node.getInputNode(slot)
    if (!inputNode) return

    const link = this.node.getInputLink(slot)
    if (!link) throw new Error('Failed to get input link')

    const inputNodeId = String(inputNode.id)

    // Try to find the node using the full ID first (for nodes outside the group)
    let inputNodeDto = this.nodesByExecutionId?.get(inputNodeId)

    // If not found, try with just the last part of the ID (for nodes inside the group)
    if (!inputNodeDto) {
      const id = inputNodeId.split(':').at(-1)
      if (id !== undefined) {
        inputNodeDto = this.nodesByExecutionId?.get(id)
      }
    }

    if (!inputNodeDto) {
      throw new Error(
        `Failed to get input node ${inputNodeId} for group node child ${this.id} with slot ${slot}`
      )
    }

    return {
      node: inputNodeDto,
      origin_id: inputNodeId,
      origin_slot: link.origin_slot
    }
  }
}
