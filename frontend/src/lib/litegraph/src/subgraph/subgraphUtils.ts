import type { LGraph } from '@/lib/litegraph/src/LGraph'
import { LGraphGroup } from '@/lib/litegraph/src/LGraphGroup'
import { LGraphNode } from '@/lib/litegraph/src/LGraphNode'
import { LLink } from '@/lib/litegraph/src/LLink'
import type { ResolvedConnection } from '@/lib/litegraph/src/LLink'
import { Reroute } from '@/lib/litegraph/src/Reroute'
import type { RerouteId } from '@/lib/litegraph/src/Reroute'
import {
  SUBGRAPH_INPUT_ID,
  SUBGRAPH_OUTPUT_ID
} from '@/lib/litegraph/src/constants'
import type {
  INodeInputSlot,
  INodeOutputSlot,
  Positionable
} from '@/lib/litegraph/src/interfaces'
import { LiteGraph, createUuidv4 } from '@/lib/litegraph/src/litegraph'
import { nextUniqueName } from '@/lib/litegraph/src/strings'
import type {
  ISerialisedNode,
  SerialisableLLink,
  SubgraphIO
} from '@/lib/litegraph/src/types/serialisation'
import type { UUID } from '@/lib/litegraph/src/utils/uuid'

import type { GraphOrSubgraph } from './Subgraph'
import type { SubgraphInput } from './SubgraphInput'
import { SubgraphInputNode } from './SubgraphInputNode'
import type { SubgraphOutput } from './SubgraphOutput'
import { SubgraphOutputNode } from './SubgraphOutputNode'

interface FilteredItems {
  nodes: Set<LGraphNode>
  reroutes: Set<Reroute>
  groups: Set<LGraphGroup>
  subgraphInputNodes: Set<SubgraphInputNode>
  subgraphOutputNodes: Set<SubgraphOutputNode>
  unknown: Set<Positionable>
}

export function splitPositionables(
  items: Iterable<Positionable>
): FilteredItems {
  const nodes = new Set<LGraphNode>()
  const reroutes = new Set<Reroute>()
  const groups = new Set<LGraphGroup>()
  const subgraphInputNodes = new Set<SubgraphInputNode>()
  const subgraphOutputNodes = new Set<SubgraphOutputNode>()

  const unknown = new Set<Positionable>()

  for (const item of items) {
    switch (true) {
      case item instanceof LGraphNode:
        nodes.add(item)
        break
      case item instanceof LGraphGroup:
        groups.add(item)
        break
      case item instanceof Reroute:
        reroutes.add(item)
        break
      case item instanceof SubgraphInputNode:
        subgraphInputNodes.add(item)
        break
      case item instanceof SubgraphOutputNode:
        subgraphOutputNodes.add(item)
        break
      default:
        unknown.add(item)
        break
    }
  }

  return {
    nodes,
    reroutes,
    groups,
    subgraphInputNodes,
    subgraphOutputNodes,
    unknown
  }
}

interface BoundaryLinks {
  boundaryLinks: LLink[]
  boundaryFloatingLinks: LLink[]
  internalLinks: LLink[]
  boundaryInputLinks: LLink[]
  boundaryOutputLinks: LLink[]
}

export function getBoundaryLinks(
  graph: LGraph,
  items: Set<Positionable>
): BoundaryLinks {
  const internalLinks: LLink[] = []
  const boundaryLinks: LLink[] = []
  const boundaryInputLinks: LLink[] = []
  const boundaryOutputLinks: LLink[] = []
  const boundaryFloatingLinks: LLink[] = []
  const visited = new WeakSet<Positionable>()

  for (const item of items) {
    if (visited.has(item)) continue
    visited.add(item)

    // Nodes
    if (item instanceof LGraphNode) {
      const node = item

      // Inputs
      if (node.inputs) {
        for (const input of node.inputs) {
          addFloatingLinks(input._floatingLinks)

          if (input.link == null) continue

          const resolved = LLink.resolve(input.link, graph)
          if (!resolved) {
            console.warn(`Failed to resolve link ID [${input.link}]`)
            continue
          }

          // Output end of this link is outside the items set
          const { link, outputNode } = resolved
          if (outputNode) {
            if (!items.has(outputNode)) {
              boundaryInputLinks.push(link)
            } else {
              internalLinks.push(link)
            }
          } else if (link.origin_id === SUBGRAPH_INPUT_ID) {
            // Subgraph input node - always boundary
            boundaryInputLinks.push(link)
          }
        }
      }

      // Outputs
      if (node.outputs) {
        for (const output of node.outputs) {
          addFloatingLinks(output._floatingLinks)

          if (!output.links) continue

          const many = LLink.resolveMany(output.links, graph)
          for (const { link, inputNode } of many) {
            if (
              // Subgraph output node
              link.target_id === SUBGRAPH_OUTPUT_ID ||
              // Input end of this link is outside the items set
              (inputNode && !items.has(inputNode))
            ) {
              boundaryOutputLinks.push(link)
            }
            // Internal links are discovered on input side.
          }
        }
      }
    } else if (item instanceof Reroute) {
      // Reroutes
      const reroute = item

      // TODO: This reroute should be on one side of the boundary.  We should mark the reroute that is on each side of the boundary.
      // TODO: This could occur any number of times on a link; each time should be marked as a separate boundary.
      // TODO: e.g. A link with 3 reroutes, the first and last reroute are in `items`, but the middle reroute is not. This will be two "in" and two "out" boundaries.
      const results = LLink.resolveMany(reroute.linkIds, graph)
      for (const { link } of results) {
        const reroutes = LLink.getReroutes(graph, link)
        const reroutesOutside = reroutes.filter(
          (reroute) => !items.has(reroute)
        )

        // for (const reroute of reroutes) {
        //   // TODO: Do the checks here.
        // }

        const { inputNode, outputNode } = link.resolve(graph)

        if (
          reroutesOutside.length ||
          (inputNode && !items.has(inputNode)) ||
          (outputNode && !items.has(outputNode))
        ) {
          boundaryLinks.push(link)
        }
      }
    }
  }

  return {
    boundaryLinks,
    boundaryFloatingLinks,
    internalLinks,
    boundaryInputLinks,
    boundaryOutputLinks
  }

  /**
   * Adds any floating links that cross the boundary.
   * @param floatingLinks The floating links to check
   */
  function addFloatingLinks(floatingLinks: Set<LLink> | undefined): void {
    if (!floatingLinks) return

    for (const link of floatingLinks) {
      const crossesBoundary = LLink.getReroutes(graph, link).some(
        (reroute) => !items.has(reroute)
      )

      if (crossesBoundary) boundaryFloatingLinks.push(link)
    }
  }
}

export function multiClone(nodes: Iterable<LGraphNode>): ISerialisedNode[] {
  const clonedNodes: ISerialisedNode[] = []

  // Selectively clone - keep IDs & links
  for (const node of nodes) {
    const newNode = LiteGraph.createNode(node.type)
    if (!newNode) {
      console.warn('Failed to create node', node.type)
      const serializedData = structuredClone(node.serialize())
      clonedNodes.push(serializedData)
      continue
    }

    // Must be cloned; litegraph "serialize" is mostly shallow clone
    const data = structuredClone(node.serialize())
    newNode.configure(data)

    clonedNodes.push(newNode.serialize())
  }

  return clonedNodes
}

/**
 * Groups resolved connections by output object. If the output is nullish, the connection will be in its own group.
 * @param resolvedConnections The resolved connections to group
 * @returns A map of grouped connections.
 */
export function groupResolvedByOutput(
  resolvedConnections: ResolvedConnection[]
): Map<SubgraphIO | INodeOutputSlot | object, ResolvedConnection[]> {
  const groupedByOutput: ReturnType<typeof groupResolvedByOutput> = new Map()

  for (const resolved of resolvedConnections) {
    // Force no group (unique object) if output is undefined; corruption or an error has occurred
    const groupBy = resolved.subgraphInput ?? resolved.output ?? {}
    const group = groupedByOutput.get(groupBy)
    if (group) {
      group.push(resolved)
    } else {
      groupedByOutput.set(groupBy, [resolved])
    }
  }

  return groupedByOutput
}
function mapReroutes(
  link: SerialisableLLink,
  reroutes: Map<RerouteId, Reroute>
) {
  let child: SerialisableLLink | Reroute = link
  let nextReroute =
    child.parentId === undefined ? undefined : reroutes.get(child.parentId)

  while (child.parentId !== undefined && nextReroute) {
    child = nextReroute
    nextReroute =
      child.parentId === undefined ? undefined : reroutes.get(child.parentId)
  }

  const lastId = child.parentId
  child.parentId = undefined
  return lastId
}

export function mapSubgraphInputsAndLinks(
  resolvedInputLinks: ResolvedConnection[],
  links: SerialisableLLink[],
  reroutes: Map<RerouteId, Reroute>
): SubgraphIO[] {
  // Group matching links
  const groupedByOutput = groupResolvedByOutput(resolvedInputLinks)

  // Create one input for each output (outside subgraph)
  const inputs: SubgraphIO[] = []

  for (const [, connections] of groupedByOutput) {
    const inputLinks: SerialisableLLink[] = []

    // Create serialised links for all links (will be recreated in subgraph)
    for (const resolved of connections) {
      const { link, input } = resolved
      if (!input) continue

      const linkData = link.asSerialisable()
      link.parentId = mapReroutes(link, reroutes)
      linkData.origin_id = SUBGRAPH_INPUT_ID
      linkData.origin_slot = inputs.length

      links.push(linkData)
      inputLinks.push(linkData)
    }

    // Use first input link
    const { input } = connections[0]
    if (!input) continue

    // Subgraph input slot
    const {
      color_off,
      color_on,
      dir,
      hasErrors,
      label,
      localized_name,
      name,
      shape,
      type
    } = input
    const uniqueName = nextUniqueName(
      name,
      inputs.map((input) => input.name)
    )
    const uniqueLocalizedName = localized_name
      ? nextUniqueName(
          localized_name,
          inputs.map((input) => input.localized_name ?? '')
        )
      : undefined

    const inputData: SubgraphIO = {
      id: createUuidv4(),
      type: String(type),
      linkIds: inputLinks.map((link) => link.id),
      name: uniqueName,
      color_off,
      color_on,
      dir,
      label,
      localized_name: uniqueLocalizedName,
      hasErrors,
      shape
    }

    inputs.push(inputData)
  }

  return inputs
}

/**
 * Clones the output slots, and updates existing links, when converting items to a subgraph.
 * @param resolvedOutputLinks The resolved output links.
 * @param links The links to add to the subgraph.
 * @returns The subgraph output slots.
 */
export function mapSubgraphOutputsAndLinks(
  resolvedOutputLinks: ResolvedConnection[],
  links: SerialisableLLink[],
  reroutes: Map<RerouteId, Reroute>
): SubgraphIO[] {
  // Group matching links
  const groupedByOutput = groupResolvedByOutput(resolvedOutputLinks)

  const outputs: SubgraphIO[] = []

  for (const [, connections] of groupedByOutput) {
    const outputLinks: SerialisableLLink[] = []

    // Create serialised links for all links (will be recreated in subgraph)
    for (const resolved of connections) {
      const { link, output } = resolved
      if (!output) continue

      const linkData = link.asSerialisable()
      linkData.parentId = mapReroutes(link, reroutes)
      linkData.target_id = SUBGRAPH_OUTPUT_ID
      linkData.target_slot = outputs.length

      links.push(linkData)
      outputLinks.push(linkData)
    }

    // Use first output link
    const { output } = connections[0]
    if (!output) continue

    // Subgraph output slot
    const {
      color_off,
      color_on,
      dir,
      hasErrors,
      label,
      localized_name,
      name,
      shape,
      type
    } = output
    const uniqueName = nextUniqueName(
      name,
      outputs.map((output) => output.name)
    )
    const uniqueLocalizedName = localized_name
      ? nextUniqueName(
          localized_name,
          outputs.map((output) => output.localized_name ?? '')
        )
      : undefined

    const outputData = {
      id: createUuidv4(),
      type: String(type),
      linkIds: outputLinks.map((link) => link.id),
      name: uniqueName,
      color_off,
      color_on,
      dir,
      label,
      localized_name: uniqueLocalizedName,
      hasErrors,
      shape
    } satisfies SubgraphIO

    outputs.push(structuredClone(outputData))
  }
  return outputs
}

/**
 * Collects all subgraph IDs used directly in a single graph (non-recursive).
 * @param graph The graph to check for subgraph nodes
 * @returns Set of subgraph IDs used in this graph
 */
export function getDirectSubgraphIds(graph: GraphOrSubgraph): Set<UUID> {
  const subgraphIds = new Set<UUID>()

  for (const node of graph._nodes) {
    if (node.isSubgraphNode()) {
      subgraphIds.add(node.type)
    }
  }

  return subgraphIds
}

/**
 * Collects all subgraph IDs referenced in a graph hierarchy using BFS.
 * @param rootGraph The graph to start from
 * @param subgraphRegistry Map of all available subgraphs
 * @returns Set of all subgraph IDs found
 */
export function findUsedSubgraphIds(
  rootGraph: GraphOrSubgraph,
  subgraphRegistry: Map<UUID, GraphOrSubgraph>
): Set<UUID> {
  const usedSubgraphIds = new Set<UUID>()
  const toVisit: GraphOrSubgraph[] = [rootGraph]

  while (toVisit.length > 0) {
    const graph = toVisit.shift()!
    const directIds = getDirectSubgraphIds(graph)

    for (const id of directIds) {
      if (!usedSubgraphIds.has(id)) {
        usedSubgraphIds.add(id)
        const subgraph = subgraphRegistry.get(id)
        if (subgraph) {
          toVisit.push(subgraph)
        }
      }
    }
  }

  return usedSubgraphIds
}

/**
 * Type guard to check if a slot is a SubgraphInput.
 * @param slot The slot to check
 * @returns true if the slot is a SubgraphInput
 */
export function isSubgraphInput(slot: unknown): slot is SubgraphInput {
  return (
    slot != null &&
    typeof slot === 'object' &&
    'parent' in slot &&
    slot.parent instanceof SubgraphInputNode
  )
}

/**
 * Type guard to check if a slot is a SubgraphOutput.
 * @param slot The slot to check
 * @returns true if the slot is a SubgraphOutput
 */
export function isSubgraphOutput(slot: unknown): slot is SubgraphOutput {
  return (
    slot != null &&
    typeof slot === 'object' &&
    'parent' in slot &&
    slot.parent instanceof SubgraphOutputNode
  )
}

/**
 * Type guard to check if a slot is a regular node slot (INodeInputSlot or INodeOutputSlot).
 * @param slot The slot to check
 * @returns true if the slot is a regular node slot
 */
export function isNodeSlot(
  slot: unknown
): slot is INodeInputSlot | INodeOutputSlot {
  return (
    slot != null &&
    typeof slot === 'object' &&
    ('link' in slot || 'links' in slot)
  )
}
