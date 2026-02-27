/**
 * This code is adapted from rgthree-comfy's link_fixer.ts
 * @see https://github.com/rgthree/rgthree-comfy/blob/b84f39c7c224de765de0b54c55b967329011819d/src_web/common/link_fixer.ts
 *
 * MIT License
 *
 * Copyright (c) 2023 Regis Gaughan, III (rgthree)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
import type { INodeOutputSlot } from '@/lib/litegraph/src/interfaces'
import type { NodeId } from '@/lib/litegraph/src/LGraphNode'
import type { SerialisedLLinkArray } from '@/lib/litegraph/src/LLink'
import type { LGraph, LGraphNode, LLink } from '@/lib/litegraph/src/litegraph'
import type {
  ISerialisedGraph,
  ISerialisedNode
} from '@/lib/litegraph/src/types/serialisation'

interface BadLinksData<T = ISerialisedGraph | LGraph> {
  hasBadLinks: boolean
  fixed: boolean
  graph: T
  patched: number
  deleted: number
}

enum IoDirection {
  INPUT,
  OUTPUT
}

function getNodeById(graph: ISerialisedGraph | LGraph, id: NodeId) {
  if ((graph as LGraph).getNodeById) {
    return (graph as LGraph).getNodeById(id)
  }
  graph = graph as ISerialisedGraph
  return graph.nodes.find((node: ISerialisedNode) => node.id == id)!
}

function extendLink(link: SerialisedLLinkArray) {
  return {
    link: link,
    id: link[0],
    origin_id: link[1],
    origin_slot: link[2],
    target_id: link[3],
    target_slot: link[4],
    type: link[5]
  }
}

/**
 * Takes a ISerialisedGraph or live LGraph and inspects the links and nodes to ensure the linking
 * makes logical sense. Can apply fixes when passed the `fix` argument as true.
 *
 * Note that fixes are a best-effort attempt. Seems to get it correct in most cases, but there is a
 * chance it correct an anomaly that results in placing an incorrect link (say, if there were two
 * links in the data). Users should take care to not overwrite work until manually checking the
 * result.
 */
export function fixBadLinks(
  graph: ISerialisedGraph | LGraph,
  options: {
    fix?: boolean
    silent?: boolean
    logger?: { log: (...args: unknown[]) => void }
  } = {}
): BadLinksData {
  const { fix = false, silent = false, logger: _logger = console } = options
  const logger = {
    log: (...args: unknown[]) => {
      if (!silent) {
        _logger.log(...args)
      }
    }
  }

  const patchedNodeSlots: {
    [nodeId: string]: {
      inputs?: { [slot: number]: number | null }
      outputs?: {
        [slots: number]: {
          links: number[]
          changes: { [linkId: number]: 'ADD' | 'REMOVE' }
        }
      }
    }
  } = {}

  const data: {
    patchedNodes: Array<ISerialisedNode | LGraphNode>
    deletedLinks: number[]
  } = {
    patchedNodes: [],
    deletedLinks: []
  }

  /**
   * Internal patch node. We keep track of changes in patchedNodeSlots in case we're in a dry run.
   */
  function patchNodeSlot(
    node: ISerialisedNode | LGraphNode,
    ioDir: IoDirection,
    slot: number,
    linkId: number,
    op: 'ADD' | 'REMOVE'
  ) {
    patchedNodeSlots[node.id] = patchedNodeSlots[node.id] || {}
    const patchedNode = patchedNodeSlots[node.id]!
    if (ioDir == IoDirection.INPUT) {
      patchedNode['inputs'] = patchedNode['inputs'] || {}
      // We can set to null (delete), so undefined means we haven't set it at all.
      if (patchedNode['inputs']![slot] !== undefined) {
        logger.log(
          ` > Already set ${node.id}.inputs[${slot}] to ${patchedNode[
            'inputs'
          ]![slot]!} Skipping.`
        )
        return false
      }
      const linkIdToSet = op === 'REMOVE' ? null : linkId
      patchedNode['inputs']![slot] = linkIdToSet
      if (fix) {
        // node.inputs[slot]!.link = linkIdToSet;
      }
    } else {
      patchedNode['outputs'] = patchedNode['outputs'] || {}
      patchedNode['outputs']![slot] = patchedNode['outputs']![slot] || {
        links: [...(node.outputs?.[slot]?.links || [])],
        changes: {}
      }
      if (patchedNode['outputs']![slot]!['changes']![linkId] !== undefined) {
        logger.log(
          ` > Already set ${node.id}.outputs[${slot}] to ${
            patchedNode['inputs']![slot]
          }! Skipping.`
        )
        return false
      }
      patchedNode['outputs']![slot]!['changes']![linkId] = op
      if (op === 'ADD') {
        const linkIdIndex =
          patchedNode['outputs']![slot]!['links'].indexOf(linkId)
        if (linkIdIndex !== -1) {
          logger.log(
            ` > Hmmm.. asked to add ${linkId} but it is already in list...`
          )
          return false
        }
        patchedNode['outputs']![slot]!['links'].push(linkId)
        if (fix) {
          node.outputs = node.outputs || []
          node.outputs[slot] =
            node.outputs[slot] ||
            ({} satisfies Partial<INodeOutputSlot> as INodeOutputSlot)
          node.outputs[slot]!.links = node.outputs[slot]!.links || []
          node.outputs[slot]!.links!.push(linkId)
        }
      } else {
        const linkIdIndex =
          patchedNode['outputs']![slot]!['links'].indexOf(linkId)
        if (linkIdIndex === -1) {
          logger.log(
            ` > Hmmm.. asked to remove ${linkId} but it doesn't exist...`
          )
          return false
        }
        patchedNode['outputs']![slot]!['links'].splice(linkIdIndex, 1)
        if (fix) {
          node.outputs?.[slot]!.links!.splice(linkIdIndex, 1)
        }
      }
    }
    data.patchedNodes.push(node)
    return true
  }

  /**
   * Internal to check if a node (or patched data) has a linkId.
   */
  function nodeHasLinkId(
    node: ISerialisedNode | LGraphNode,
    ioDir: IoDirection,
    slot: number,
    linkId: number
  ) {
    // Patched data should be canonical. We can double check if fixing too.
    let has = false
    if (ioDir === IoDirection.INPUT) {
      const nodeHasIt = node.inputs?.[slot]?.link === linkId
      if (patchedNodeSlots[node.id]?.['inputs']) {
        const patchedHasIt =
          patchedNodeSlots[node.id]!['inputs']![slot] === linkId
        // If we're fixing, double check that node matches.
        if (fix && nodeHasIt !== patchedHasIt) {
          throw Error('Error. Expected node to match patched data.')
        }
        has = patchedHasIt
      } else {
        has = !!nodeHasIt
      }
    } else {
      const nodeHasIt = node.outputs?.[slot]?.links?.includes(linkId)
      if (patchedNodeSlots[node.id]?.['outputs']?.[slot]?.['changes'][linkId]) {
        const patchedHasIt =
          patchedNodeSlots[node.id]!['outputs']![slot]?.links.includes(linkId)
        // If we're fixing, double check that node matches.
        if (fix && nodeHasIt !== patchedHasIt) {
          throw Error('Error. Expected node to match patched data.')
        }
        has = !!patchedHasIt
      } else {
        has = !!nodeHasIt
      }
    }
    return has
  }

  /**
   * Internal to check if a node (or patched data) has a linkId.
   */
  function nodeHasAnyLink(
    node: ISerialisedNode | LGraphNode,
    ioDir: IoDirection,
    slot: number
  ) {
    // Patched data should be canonical. We can double check if fixing too.
    let hasAny = false
    if (ioDir === IoDirection.INPUT) {
      const nodeHasAny = node.inputs?.[slot]?.link != null
      if (patchedNodeSlots[node.id]?.['inputs']) {
        const patchedHasAny =
          patchedNodeSlots[node.id]!['inputs']![slot] != null
        // If we're fixing, double check that node matches.
        if (fix && nodeHasAny !== patchedHasAny) {
          throw Error('Error. Expected node to match patched data.')
        }
        hasAny = patchedHasAny
      } else {
        hasAny = !!nodeHasAny
      }
    } else {
      const nodeHasAny = node.outputs?.[slot]?.links?.length
      if (patchedNodeSlots[node.id]?.['outputs']?.[slot]?.['changes']) {
        const patchedHasAny =
          patchedNodeSlots[node.id]!['outputs']![slot]?.links.length
        // If we're fixing, double check that node matches.
        if (fix && nodeHasAny !== patchedHasAny) {
          throw Error('Error. Expected node to match patched data.')
        }
        hasAny = !!patchedHasAny
      } else {
        hasAny = !!nodeHasAny
      }
    }
    return hasAny
  }

  let links: Array<SerialisedLLinkArray | LLink> = []
  if (!Array.isArray(graph.links)) {
    links = Object.values(graph.links).reduce((acc, v) => {
      acc[v.id] = v
      return acc
    }, links)
  } else {
    links = graph.links
  }

  const linksReverse = [...links]
  linksReverse.reverse()
  for (const l of linksReverse) {
    if (!l) continue
    const link =
      (l as LLink).origin_slot != null
        ? (l as LLink)
        : extendLink(l as SerialisedLLinkArray)

    const originNode = getNodeById(graph, link.origin_id)
    const originHasLink = () =>
      nodeHasLinkId(originNode!, IoDirection.OUTPUT, link.origin_slot, link.id)
    const patchOrigin = (op: 'ADD' | 'REMOVE', id = link.id) =>
      patchNodeSlot(originNode!, IoDirection.OUTPUT, link.origin_slot, id, op)

    const targetNode = getNodeById(graph, link.target_id)
    const targetHasLink = () =>
      nodeHasLinkId(targetNode!, IoDirection.INPUT, link.target_slot, link.id)
    const targetHasAnyLink = () =>
      nodeHasAnyLink(targetNode!, IoDirection.INPUT, link.target_slot)
    const patchTarget = (op: 'ADD' | 'REMOVE', id = link.id) =>
      patchNodeSlot(targetNode!, IoDirection.INPUT, link.target_slot, id, op)

    const originLog = `origin(${link.origin_id}).outputs[${link.origin_slot}].links`
    const targetLog = `target(${link.target_id}).inputs[${link.target_slot}].link`

    if (!originNode || !targetNode) {
      if (!originNode && !targetNode) {
        logger.log(
          `Link ${link.id} is invalid, ` +
            `both origin ${link.origin_id} and target ${link.target_id} do not exist`
        )
      } else if (!originNode) {
        logger.log(
          `Link ${link.id} is funky... ` +
            `origin ${link.origin_id} does not exist, but target ${link.target_id} does.`
        )
        if (targetHasLink()) {
          logger.log(
            ` > [PATCH] ${targetLog} does have link, will remove the inputs' link first.`
          )
          patchTarget('REMOVE', -1)
        }
      } else if (!targetNode) {
        logger.log(
          `Link ${link.id} is funky... ` +
            `target ${link.target_id} does not exist, but origin ${link.origin_id} does.`
        )
        if (originHasLink()) {
          logger.log(
            ` > [PATCH] Origin's links' has ${link.id}; will remove the link first.`
          )
          patchOrigin('REMOVE')
        }
      }
      continue
    }

    if (targetHasLink() || originHasLink()) {
      if (!originHasLink()) {
        logger.log(
          `${link.id} is funky... ${originLog} does NOT contain it, but ${targetLog} does.`
        )

        logger.log(
          ` > [PATCH] Attempt a fix by adding this ${link.id} to ${originLog}.`
        )
        patchOrigin('ADD')
      } else if (!targetHasLink()) {
        logger.log(
          `${link.id} is funky... ${targetLog} is NOT correct (is ${
            targetNode.inputs?.[link.target_slot]?.link
          }), but ${originLog} contains it`
        )
        if (!targetHasAnyLink()) {
          logger.log(
            ` > [PATCH] ${targetLog} is not defined, will set to ${link.id}.`
          )
          let patched = patchTarget('ADD')
          if (!patched) {
            logger.log(
              ` > [PATCH] Nvm, ${targetLog} already patched. Removing ${link.id} from ${originLog}.`
            )
            patched = patchOrigin('REMOVE')
          }
        } else {
          logger.log(
            ` > [PATCH] ${targetLog} is defined, removing ${link.id} from ${originLog}.`
          )
          patchOrigin('REMOVE')
        }
      }
    }
  }

  // Now that we've cleaned up the inputs, outputs, run through it looking for dangling links.,
  for (const l of linksReverse) {
    if (!l) continue
    const link =
      (l as LLink).origin_slot != null
        ? (l as LLink)
        : extendLink(l as SerialisedLLinkArray)
    const originNode = getNodeById(graph, link.origin_id)
    const targetNode = getNodeById(graph, link.target_id)
    // Now that we've manipulated the linking, check again if they both exist.
    if (
      (!originNode ||
        !nodeHasLinkId(
          originNode,
          IoDirection.OUTPUT,
          link.origin_slot,
          link.id
        )) &&
      (!targetNode ||
        !nodeHasLinkId(
          targetNode,
          IoDirection.INPUT,
          link.target_slot,
          link.id
        ))
    ) {
      logger.log(
        `${link.id} is def invalid; BOTH origin node ${link.origin_id} ${
          !originNode ? 'is removed' : `doesn't have ${link.id}`
        } and ${link.origin_id} target node ${
          !targetNode ? 'is removed' : `doesn't have ${link.id}`
        }.`
      )
      data.deletedLinks.push(link.id)
      continue
    }
  }

  // If we're fixing, then we've been patching along the way. Now go through and actually delete
  // the zombie links from `app.graph.links`
  if (fix) {
    for (let i = data.deletedLinks.length - 1; i >= 0; i--) {
      logger.log(`Deleting link #${data.deletedLinks[i]}.`)
      if ((graph as LGraph).getNodeById) {
        delete graph.links[data.deletedLinks[i]!]
      } else {
        graph = graph as ISerialisedGraph
        // Sometimes we got objects for links if passed after ComfyUI's loadGraphData modifies the
        // data. We make a copy now, but can handle the bastardized objects just in case.
        const idx = graph.links.findIndex(
          (l) =>
            l &&
            (l[0] === data.deletedLinks[i] ||
              ('id' in l && l.id === data.deletedLinks[i]))
        )
        if (idx === -1) {
          logger.log(`INDEX NOT FOUND for #${data.deletedLinks[i]}`)
        }
        logger.log(`splicing ${idx} from links`)
        graph.links.splice(idx, 1)
      }
    }
    // If we're a serialized graph, we can filter out the links because it's just an array.
    if (!(graph as LGraph).getNodeById) {
      graph.links = (graph as ISerialisedGraph).links.filter((l) => !!l)
    }
  }
  if (!data.patchedNodes.length && !data.deletedLinks.length) {
    return {
      hasBadLinks: false,
      fixed: false,
      graph,
      patched: data.patchedNodes.length,
      deleted: data.deletedLinks.length
    }
  }

  logger.log(
    `${fix ? 'Made' : 'Would make'} ${data.patchedNodes.length || 'no'} node link patches, and ${
      data.deletedLinks.length || 'no'
    } stale link removals.`
  )

  let hasBadLinks: boolean = !!(
    data.patchedNodes.length || data.deletedLinks.length
  )
  // If we're fixing, then let's run it again to see if there are no more bad links.
  if (fix && !silent) {
    const rerun = fixBadLinks(graph, { fix: false, silent: true })
    hasBadLinks = rerun.hasBadLinks
  }

  return {
    hasBadLinks,
    fixed: !!hasBadLinks && fix,
    graph,
    patched: data.patchedNodes.length,
    deleted: data.deletedLinks.length
  }
}
