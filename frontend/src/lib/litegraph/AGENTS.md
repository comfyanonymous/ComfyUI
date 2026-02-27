# Litegraph Guidelines

## Code Philosophy

- Write concise, legible, and easily maintainable code
- Avoid repetition where possible, but not at expense of legibility
- Prefer running single tests, not the whole suite, for performance

## Widget Serialization

See `docs/WIDGET_SERIALIZATION.md` for the distinction between `widget.serialize` (workflow persistence) and `widget.options.serialize` (API prompt). These are different properties checked by different code paths — a common source of confusion.

## Code Style

- Prefer single line `if` syntax for concise expressions
- Take advantage of `TypedArray` `subarray` when appropriate
- The `size` and `pos` properties of `Rectangle` share the same array buffer
- Prefer returning `undefined` over `null`
- Type assertions are a last resort (acceptable for legacy code interop)

## Circular Dependencies in Tests

**CRITICAL**: Always import from the barrel export for subgraph code:

```typescript
// ✅ Correct - barrel import
import { LGraph, Subgraph, SubgraphNode } from '@/lib/litegraph/src/litegraph'

// ❌ Wrong - causes circular dependency
import { LGraph } from '@/lib/litegraph/src/LGraph'
```

**Root cause**: `LGraph` ↔ `Subgraph` circular dependency (Subgraph extends LGraph, LGraph creates Subgraph instances).

## Known Limitations

- Subgraph dynamic input limitations (autogrow/matchtype): see [`src/core/graph/subgraph-dynamic-input-limitations.md`](../../core/graph/subgraph-dynamic-input-limitations.md)

## Test Helpers

```typescript
import {
  createTestSubgraph,
  createTestSubgraphNode
} from './__fixtures__/subgraphHelpers'

function createTestSetup() {
  const subgraph = createTestSubgraph()
  const subgraphNode = createTestSubgraphNode(subgraph)
  return { subgraph, subgraphNode }
}
```
