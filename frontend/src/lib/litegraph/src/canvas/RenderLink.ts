import type { LGraphNode } from '@/lib/litegraph/src/LGraphNode'
import type { CustomEventTarget } from '@/lib/litegraph/src/infrastructure/CustomEventTarget'
import type { LinkConnectorEventMap } from '@/lib/litegraph/src/infrastructure/LinkConnectorEventMap'
import type { LinkNetwork, Point } from '@/lib/litegraph/src/interfaces'
import type {
  INodeInputSlot,
  INodeOutputSlot,
  LLink,
  Reroute
} from '@/lib/litegraph/src/litegraph'
import type { SubgraphIONodeBase } from '@/lib/litegraph/src/subgraph/SubgraphIONodeBase'
import type { SubgraphInput } from '@/lib/litegraph/src/subgraph/SubgraphInput'
import type { SubgraphOutput } from '@/lib/litegraph/src/subgraph/SubgraphOutput'
import type { NodeLike } from '@/lib/litegraph/src/types/NodeLike'
import type { LinkDirection } from '@/lib/litegraph/src/types/globalEnums'

export interface RenderLink {
  /** The type of link being connected. */
  readonly toType: 'input' | 'output'
  /** The source {@link Point} of the link being connected. */
  readonly fromPos: Point
  /** The direction the link starts off as.  If {@link toType} is `output`, this will be the direction the link input faces. */
  readonly fromDirection: LinkDirection
  /** If set, this will force a dragged link "point" from the cursor in the specified direction. */
  dragDirection: LinkDirection

  /** The network that the link belongs to. */
  readonly network: LinkNetwork
  /** The node that the link is being connected from. */
  readonly node: LGraphNode | SubgraphIONodeBase<SubgraphInput | SubgraphOutput>
  /** The slot that the link is being connected from. */
  readonly fromSlot:
    | INodeOutputSlot
    | INodeInputSlot
    | SubgraphInput
    | SubgraphOutput
  /** The index of the slot that the link is being connected from. */
  readonly fromSlotIndex: number
  /** The reroute that the link is being connected from. */
  readonly fromReroute?: Reroute

  /**
   * Capability checks used for hit-testing and validation during drag.
   * Implementations should return `false` when a connection is not possible
   * rather than throwing.
   */
  canConnectToInput(node: NodeLike, input: INodeInputSlot): boolean
  canConnectToOutput(node: NodeLike, output: INodeOutputSlot): boolean
  /** Optional: only some links support validating subgraph IO or reroutes. */
  canConnectToSubgraphInput?(input: SubgraphInput): boolean
  canConnectToReroute?(reroute: Reroute): boolean

  connectToInput(
    node: LGraphNode,
    input: INodeInputSlot,
    events?: CustomEventTarget<LinkConnectorEventMap>
  ): void
  connectToOutput(
    node: LGraphNode,
    output: INodeOutputSlot,
    events?: CustomEventTarget<LinkConnectorEventMap>
  ): void
  connectToSubgraphInput(
    input: SubgraphInput,
    events?: CustomEventTarget<LinkConnectorEventMap>
  ): void
  connectToSubgraphOutput(
    output: SubgraphOutput,
    events?: CustomEventTarget<LinkConnectorEventMap>
  ): void

  connectToRerouteInput(
    reroute: Reroute,
    {
      node,
      input,
      link
    }: { node: LGraphNode; input: INodeInputSlot; link: LLink },
    events: CustomEventTarget<LinkConnectorEventMap>,
    originalReroutes: Reroute[]
  ): void

  connectToRerouteOutput(
    reroute: Reroute,
    outputNode: LGraphNode,
    output: INodeOutputSlot,
    events: CustomEventTarget<LinkConnectorEventMap>
  ): void
}
