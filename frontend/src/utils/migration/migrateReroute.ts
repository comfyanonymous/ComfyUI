import _ from 'es-toolkit/compat'

import type {
  ComfyLinkObject,
  ComfyNode,
  NodeId,
  Reroute,
  WorkflowJSON04
} from '@/platform/workflow/validation/schemas/workflowSchema'

type RerouteNode = ComfyNode & {
  type: 'Reroute'
}

type LinkExtension = {
  id: number
  parentId: number
}

/**
 * Identifies all legacy Reroute nodes in a workflow
 */
export function findLegacyRerouteNodes(
  workflow: WorkflowJSON04
): RerouteNode[] {
  return workflow.nodes.filter(
    (node) => node.type === 'Reroute'
  ) as RerouteNode[]
}

/**
 * Checks if the workflow has no native reroutes
 */
export function noNativeReroutes(workflow: WorkflowJSON04): boolean {
  return (
    !workflow.extra?.reroutes?.length && !workflow.extra?.linkExtensions?.length
  )
}

/**
 * Gets the center position of a node
 */
function getNodeCenter(node: ComfyNode): [number, number] {
  return [node.pos[0] + node.size[0] / 2, node.pos[1] + node.size[1] / 2]
}

class ConversionContext {
  nodeById: Record<NodeId, ComfyNode>
  linkById: Record<number, ComfyLinkObject>
  rerouteById: Record<number, Reroute>
  rerouteByNodeId: Record<NodeId, Reroute>
  linkExtensions: LinkExtension[]

  /** Reroutes that has at least a valid link pass through it */
  validReroutes: Set<Reroute>

  private _rerouteIdCounter = 0

  constructor(public workflow: WorkflowJSON04) {
    this.nodeById = _.keyBy(workflow.nodes.map(_.cloneDeep), 'id')
    this.linkById = _.keyBy(
      workflow.links.map((l) => ({
        id: l[0],
        origin_id: l[1],
        origin_slot: l[2],
        target_id: l[3],
        target_slot: l[4],
        type: l[5]
      })),
      'id'
    )

    const reroutes = findLegacyRerouteNodes(workflow).map((node, index) => ({
      nodeId: node.id,
      id: index + 1,
      pos: getNodeCenter(node),
      linkIds: []
    }))
    this._rerouteIdCounter = reroutes.length + 1

    this.rerouteByNodeId = _.keyBy(reroutes, 'nodeId')
    this.rerouteById = _.keyBy(reroutes, 'id')

    this.linkExtensions = []
    this.validReroutes = new Set()
  }

  /**
   * Gets the chain of reroute nodes leading to the given node
   */
  private _getRerouteChain(node: RerouteNode): RerouteNode[] {
    const nodes: RerouteNode[] = []
    let currentNode: RerouteNode = node
    while (currentNode?.type === 'Reroute') {
      nodes.push(currentNode)
      const inputLink: ComfyLinkObject | undefined =
        this.linkById[currentNode.inputs?.[0]?.link ?? 0]

      if (!inputLink) {
        break
      }

      currentNode = this.nodeById[inputLink.origin_id] as RerouteNode
    }

    return nodes
  }

  private _connectRerouteChain(rerouteNodes: RerouteNode[]): Reroute[] {
    const reroutes = rerouteNodes.map((node) => this.rerouteByNodeId[node.id])
    for (const reroute of reroutes) {
      this.validReroutes.add(reroute)
    }

    for (let i = 0; i < rerouteNodes.length - 1; i++) {
      const to = reroutes[i]
      const from = reroutes[i + 1]
      to.parentId = from.id
    }

    return reroutes
  }

  private _createNewLink(
    startingLink: ComfyLinkObject,
    endingLink: ComfyLinkObject,
    rerouteNodes: RerouteNode[]
  ): ComfyLinkObject {
    if (rerouteNodes.length === 0) {
      throw new Error('No reroute nodes found')
    }

    const reroute = this.rerouteByNodeId[rerouteNodes[0].id]
    this.linkExtensions.push({
      id: endingLink.id,
      parentId: reroute.id
    })

    const reroutes = this._connectRerouteChain(rerouteNodes)
    for (const reroute of reroutes) {
      reroute.linkIds ??= []
      reroute.linkIds.push(endingLink.id)
      delete reroute.floating
    }

    return {
      id: endingLink.id,
      origin_id: startingLink.origin_id,
      origin_slot: startingLink.origin_slot,
      target_id: endingLink.target_id,
      target_slot: endingLink.target_slot,
      type: endingLink.type
    }
  }

  private _createNewInputFloatingLink(
    endingLink: ComfyLinkObject,
    rerouteNodes: RerouteNode[]
  ): ComfyLinkObject {
    const reroutes = this._connectRerouteChain(rerouteNodes)
    for (const reroute of reroutes) {
      if (!reroute.linkIds?.length) {
        reroute.floating = {
          slotType: 'input'
        }
      }
    }
    return {
      id: this._rerouteIdCounter++,
      origin_id: -1,
      origin_slot: -1,
      target_id: endingLink.target_id,
      target_slot: endingLink.target_slot,
      type: endingLink.type,
      parentId: reroutes[0].id
    }
  }

  private _createNewOutputFloatingLink(
    startingLink: ComfyLinkObject,
    rerouteNodes: RerouteNode[]
  ): ComfyLinkObject {
    const reroutes = this._connectRerouteChain(rerouteNodes)
    for (const reroute of reroutes) {
      if (!reroute.linkIds?.length) {
        reroute.floating = {
          slotType: 'output'
        }
      }
    }

    return {
      id: this._rerouteIdCounter++,
      origin_id: startingLink.origin_id,
      origin_slot: startingLink.origin_slot,
      target_id: -1,
      target_slot: -1,
      type: startingLink.type,
      parentId: reroutes[0].id
    }
  }

  private _reconnectLinks(nodes: ComfyNode[], links: ComfyLinkObject[]): void {
    // Remove all existing links on sockets
    for (const node of nodes) {
      for (const input of node.inputs ?? []) {
        input.link = null
      }
      for (const output of node.outputs ?? []) {
        output.links = []
      }
    }

    const nodesById = _.keyBy(nodes, 'id')

    // Reconnect the links
    for (const link of links) {
      const sourceNode = nodesById[link.origin_id]
      sourceNode.outputs![link.origin_slot]!.links!.push(link.id)

      const targetNode = nodesById[link.target_id]
      targetNode.inputs![link.target_slot]!.link = link.id
    }
  }

  migrateReroutes(): WorkflowJSON04 {
    const links: ComfyLinkObject[] = []
    const floatingLinks: ComfyLinkObject[] = []

    const endingLinks: ComfyLinkObject[] = []

    for (const link of Object.values(this.linkById)) {
      const sourceIsReroute = !!this.rerouteByNodeId[link.origin_id]
      const targetIsReroute = !!this.rerouteByNodeId[link.target_id]

      // Process links that are not connected to reroute nodes
      if (!sourceIsReroute && !targetIsReroute) {
        links.push(link)
      } else if (sourceIsReroute && !targetIsReroute) {
        endingLinks.push(link)
      }
    }

    for (const endingLink of endingLinks) {
      const endingRerouteNode = this.nodeById[
        endingLink.origin_id
      ] as RerouteNode
      const rerouteNodes = this._getRerouteChain(endingRerouteNode)
      const startingLink =
        this.linkById[
          rerouteNodes[rerouteNodes.length - 1]?.inputs?.[0]?.link ?? -1
        ]
      if (startingLink) {
        // Valid link found, create a new link
        links.push(this._createNewLink(startingLink, endingLink, rerouteNodes))
      } else {
        // Floating link found, create a new floating link
        floatingLinks.push(
          this._createNewInputFloatingLink(endingLink, rerouteNodes)
        )
      }
    }

    const floatingEndingRerouteNodes = Object.keys(this.rerouteByNodeId)
      .map((nodeId) => this.nodeById[nodeId] as RerouteNode)
      .filter((rerouteNode) => {
        const output = rerouteNode.outputs?.[0]
        if (!output) return false
        return !output.links?.length
      })

    for (const rerouteNode of floatingEndingRerouteNodes) {
      const rerouteNodes = this._getRerouteChain(rerouteNode)
      const startingLink =
        this.linkById[
          rerouteNodes[rerouteNodes.length - 1]?.inputs?.[0]?.link ?? -1
        ]
      if (startingLink) {
        floatingLinks.push(
          this._createNewOutputFloatingLink(startingLink, rerouteNodes)
        )
      }
    }

    const nodes = Object.values(this.nodeById).filter(
      (node) => node.type !== 'Reroute'
    )
    this._reconnectLinks(nodes, links)

    return {
      ...this.workflow,
      nodes,
      links: links.map((link) => [
        link.id,
        link.origin_id,
        link.origin_slot,
        link.target_id,
        link.target_slot,
        link.type
      ]),
      floatingLinks: floatingLinks.length > 0 ? floatingLinks : undefined,
      extra: {
        ...this.workflow.extra,
        reroutes: Array.from(this.validReroutes).map(
          (reroute) => _.omit(reroute, 'nodeId') as Reroute
        ),
        linkExtensions: this.linkExtensions
      }
    }
  }
}

/**
 * Main function to migrate legacy reroute nodes to native reroute points
 */
export const migrateLegacyRerouteNodes = (
  workflow: WorkflowJSON04
): WorkflowJSON04 => {
  // Find all legacy Reroute nodes
  const legacyRerouteNodes = findLegacyRerouteNodes(workflow)

  // If no reroute nodes, return the workflow unchanged
  if (legacyRerouteNodes.length === 0) {
    return workflow
  }

  // Create a deep copy of the workflow to avoid mutating the original
  const newWorkflow = JSON.parse(JSON.stringify(workflow)) as WorkflowJSON04

  // Initialize extra structure if needed
  if (!newWorkflow.extra) {
    newWorkflow.extra = {}
  }

  const context = new ConversionContext(newWorkflow)
  return context.migrateReroutes()
}
