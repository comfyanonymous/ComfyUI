import type { LGraphNode } from '@/lib/litegraph/src/LGraphNode'
import type { Reroute } from '@/lib/litegraph/src/Reroute'
import type {
  INodeInputSlot,
  INodeOutputSlot,
  LinkNetwork
} from '@/lib/litegraph/src/litegraph'

import type { LinkConnector } from './LinkConnector'
import { ToInputRenderLink } from './ToInputRenderLink'
import { ToOutputRenderLink } from './ToOutputRenderLink'

/**
 * @internal A workaround class to support connecting to reroutes to node outputs.
 */
export class ToOutputFromRerouteLink extends ToOutputRenderLink {
  constructor(
    network: LinkNetwork,
    node: LGraphNode,
    fromSlot: INodeInputSlot,
    override readonly fromReroute: Reroute,
    readonly linkConnector: LinkConnector
  ) {
    super(network, node, fromSlot, fromReroute)
  }

  override canConnectToReroute(): false {
    return false
  }

  override connectToOutput(node: LGraphNode, output: INodeOutputSlot) {
    const nuRenderLink = new ToInputRenderLink(this.network, node, output)
    this.linkConnector._connectOutputToReroute(this.fromReroute, nuRenderLink)
  }
}
