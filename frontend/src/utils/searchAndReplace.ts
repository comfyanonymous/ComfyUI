import type { LGraph, Subgraph } from '@/lib/litegraph/src/litegraph'
import { formatDate } from '@/utils/formatUtil'
import { collectAllNodes } from '@/utils/graphTraversalUtil'

export function applyTextReplacements(
  graph: LGraph | Subgraph,
  value: string
): string {
  const allNodes = collectAllNodes(graph)

  return value.replace(/%([^%]+)%/g, function (match, text) {
    const split = text.split('.')
    if (split.length !== 2) {
      // Special handling for dates
      if (split[0].startsWith('date:')) {
        return formatDate(split[0].substring(5), new Date())
      }

      if (text !== 'width' && text !== 'height') {
        // Dont warn on standard replacements
        console.warn('Invalid replacement pattern', text)
      }
      return match
    }

    // Find node with matching S&R property name
    let nodes = allNodes.filter(
      (n) => n.properties?.['Node name for S&R'] === split[0]
    )
    // If we can't, see if there is a node with that title
    if (!nodes.length) {
      nodes = allNodes.filter((n) => n.title === split[0])
    }
    if (!nodes.length) {
      console.warn('Unable to find node', split[0])
      return match
    }

    if (nodes.length > 1) {
      console.warn('Multiple nodes matched', split[0], 'using first match')
    }

    const node = nodes[0]

    const widget = node.widgets?.find((w) => w.name === split[1])
    if (!widget) {
      console.warn('Unable to find widget', split[1], 'on node', split[0], node)
      return match
    }

    return ((widget.value ?? '') + '').replaceAll(
      // eslint-disable-next-line no-control-regex
      /[/?<>\\:*|"\x00-\x1F\x7F]/g,
      '_'
    )
  })
}
