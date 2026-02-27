import type { LGraph, Subgraph } from '../../src/lib/litegraph/src/litegraph'
import { isSubgraph } from '../../src/utils/typeGuardUtil'

/**
 * Assertion helper for tests where being in a subgraph is a precondition.
 * Throws a clear error if the graph is not a Subgraph.
 */
export function assertSubgraph(
  graph: LGraph | Subgraph | null | undefined
): asserts graph is Subgraph {
  if (!isSubgraph(graph)) {
    throw new Error(
      'Expected to be in a subgraph context, but graph is not a Subgraph'
    )
  }
}
