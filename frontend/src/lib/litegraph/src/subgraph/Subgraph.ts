// Re-export Subgraph and GraphOrSubgraph from LGraph.ts to maintain compatibility
// This is a temporary fix to resolve circular dependency issues
export { Subgraph, type GraphOrSubgraph } from '@/lib/litegraph/src/LGraph'
