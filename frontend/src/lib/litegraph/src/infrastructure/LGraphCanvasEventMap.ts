import type { LGraph } from '@/lib/litegraph/src/LGraph'
import type { LGraphButton } from '@/lib/litegraph/src/LGraphButton'
import type { LGraphGroup } from '@/lib/litegraph/src/LGraphGroup'
import type { LGraphNode } from '@/lib/litegraph/src/LGraphNode'
import type { ConnectingLink } from '@/lib/litegraph/src/interfaces'
import type { Subgraph } from '@/lib/litegraph/src/subgraph/Subgraph'
import type { SubgraphNode } from '@/lib/litegraph/src/subgraph/SubgraphNode'
import type { CanvasPointerEvent } from '@/lib/litegraph/src/types/events'

export interface LGraphCanvasEventMap {
  /** The active graph has changed. */
  'litegraph:set-graph': {
    /** The new active graph. */
    newGraph: LGraph | Subgraph
    /** The old active graph, or `null` if there was no active graph. */
    oldGraph: LGraph | Subgraph | null | undefined
  }
  'subgraph-opened': {
    subgraph: Subgraph
    closingGraph: LGraph
    fromNode: SubgraphNode
  }

  /** Dispatched after a group of items has been converted to a subgraph*/
  'subgraph-converted': {
    subgraphNode: SubgraphNode
  }

  'litegraph:canvas':
    | { subType: 'before-change' | 'after-change' }
    | {
        subType: 'empty-release'
        originalEvent?: CanvasPointerEvent
        linkReleaseContext?: { links: ConnectingLink[] }
      }
    | {
        subType: 'group-double-click'
        originalEvent?: CanvasPointerEvent
        group: LGraphGroup
      }
    | {
        subType: 'empty-double-click'
        originalEvent?: CanvasPointerEvent
      }
    | {
        subType: 'node-double-click'
        originalEvent?: CanvasPointerEvent
        node: LGraphNode
      }

  /** A title button on a node was clicked. */
  'litegraph:node-title-button-clicked': {
    node: LGraphNode
    button: LGraphButton
  }
}
