# Subgraph Testing Fixtures and Utilities

This directory contains the testing infrastructure for LiteGraph's subgraph functionality. These utilities provide a consistent, easy-to-use API for writing subgraph tests.

## What is a Subgraph?

A subgraph in LiteGraph is a graph-within-a-graph that can be reused as a single node. It has:

- Input slots that map to an internal input node
- Output slots that map to an internal output node
- Internal nodes and connections
- The ability to be instantiated multiple times as SubgraphNode instances

## Quick Start

```typescript
// Import what you need
import {
  createTestSubgraph,
  assertSubgraphStructure
} from './__fixtures__/subgraphHelpers'
import { subgraphTest } from './__fixtures__/subgraphFixtures'

// Option 1: Create a subgraph manually
it('should do something', () => {
  const subgraph = createTestSubgraph({
    name: 'My Test Subgraph',
    inputCount: 2,
    outputCount: 1
  })

  // Test your functionality
  expect(subgraph.inputs).toHaveLength(2)
})

// Option 2: Use pre-configured fixtures
subgraphTest('should handle events', ({ simpleSubgraph, eventCapture }) => {
  // simpleSubgraph comes pre-configured with 1 input, 1 output, and 2 nodes
  expect(simpleSubgraph.inputs).toHaveLength(1)
  // Your test logic here
})
```

## Files Overview

### `subgraphHelpers.ts` - Core Helper Functions

**Main Factory Functions:**

- `createTestSubgraph(options?)` - Creates a fully configured Subgraph instance with root graph
- `createTestSubgraphNode(subgraph, options?)` - Creates a SubgraphNode (instance of a subgraph)
- `createNestedSubgraphs(options?)` - Creates nested subgraph hierarchies for testing deep structures

**Assertion & Validation:**

- `assertSubgraphStructure(subgraph, expected)` - Validates subgraph has expected inputs/outputs/nodes
- `verifyEventSequence(events, expectedSequence)` - Ensures events fired in correct order
- `logSubgraphStructure(subgraph, label?)` - Debug helper to print subgraph structure

**Test Data & Events:**

- `createTestSubgraphData(overrides?)` - Creates raw ExportedSubgraph data for serialization tests
- `createComplexSubgraphData(nodeCount?)` - Generates complex subgraph with internal connections
- `createEventCapture(eventTarget, eventTypes)` - Sets up event monitoring with automatic cleanup

### `subgraphFixtures.ts` - Vitest Fixtures

Pre-configured test scenarios that automatically set up and tear down:

**Basic Fixtures (`subgraphTest`):**

- `emptySubgraph` - Minimal subgraph with no inputs/outputs/nodes
- `simpleSubgraph` - 1 input ("input": number), 1 output ("output": number), 2 internal nodes
- `complexSubgraph` - 3 inputs (data, control, text), 2 outputs (result, status), 5 nodes
- `nestedSubgraph` - 3-level deep hierarchy with 2 nodes per level
- `subgraphWithNode` - Complete setup: subgraph definition + SubgraphNode instance + parent graph
- `eventCapture` - Subgraph with event monitoring for all I/O events

**Edge Case Fixtures (`edgeCaseTest`):**

- `circularSubgraph` - Two subgraphs set up for circular reference testing
- `deeplyNestedSubgraph` - 50 levels deep for performance/limit testing
- `maxIOSubgraph` - 20 inputs and 20 outputs for stress testing

### `testSubgraphs.json` - Sample Test Data

Pre-defined subgraph configurations for consistent testing across different scenarios.

**Note on Static UUIDs**: The hardcoded UUIDs in this file (e.g., "simple-subgraph-uuid", "complex-subgraph-uuid") are intentionally static to ensure test reproducibility and snapshot testing compatibility.

## Usage Examples

### Basic Test Creation

```typescript
import { describe, expect, it } from 'vitest'
import {
  createTestSubgraph,
  assertSubgraphStructure
} from './fixtures/subgraphHelpers'

describe('My Subgraph Feature', () => {
  it('should work correctly', () => {
    const subgraph = createTestSubgraph({
      name: 'My Test',
      inputCount: 2,
      outputCount: 1,
      nodeCount: 3
    })

    assertSubgraphStructure(subgraph, {
      inputCount: 2,
      outputCount: 1,
      nodeCount: 3,
      name: 'My Test'
    })

    // Your specific test logic...
  })
})
```

### Using Fixtures

```typescript
import { subgraphTest } from './fixtures/subgraphFixtures'

subgraphTest('should handle events', ({ eventCapture }) => {
  const { subgraph, capture } = eventCapture

  subgraph.addInput('test', 'number')

  expect(capture.events).toHaveLength(2) // adding-input, input-added
})
```

### Event Testing

```typescript
import {
  createEventCapture,
  verifyEventSequence
} from './fixtures/subgraphHelpers'

it('should fire events in correct order', () => {
  const subgraph = createTestSubgraph()
  const capture = createEventCapture(subgraph.events, [
    'adding-input',
    'input-added'
  ])

  subgraph.addInput('test', 'number')

  verifyEventSequence(capture.events, ['adding-input', 'input-added'])

  capture.cleanup() // Important: clean up listeners
})
```

### Nested Structure Testing

```typescript
import { createNestedSubgraphs } from './fixtures/subgraphHelpers'

it('should handle deep nesting', () => {
  const nested = createNestedSubgraphs({
    depth: 5,
    nodesPerLevel: 2
  })

  expect(nested.subgraphs).toHaveLength(5)
  expect(nested.leafSubgraph.nodes).toHaveLength(2)
})
```

## Common Patterns

### Testing SubgraphNode Instances

```typescript
it('should create and configure a SubgraphNode', () => {
  // First create the subgraph definition
  const subgraph = createTestSubgraph({
    inputs: [{ name: 'value', type: 'number' }],
    outputs: [{ name: 'result', type: 'number' }]
  })

  // Then create an instance of it
  const subgraphNode = createTestSubgraphNode(subgraph, {
    pos: [100, 200],
    size: [180, 100]
  })

  // The SubgraphNode will have matching slots
  expect(subgraphNode.inputs).toHaveLength(1)
  expect(subgraphNode.outputs).toHaveLength(1)
  expect(subgraphNode.subgraph).toBe(subgraph)
})
```

### Complete Test with Parent Graph

```typescript
subgraphTest('should work in a parent graph', ({ subgraphWithNode }) => {
  const { subgraph, subgraphNode, parentGraph } = subgraphWithNode

  // Everything is pre-configured and connected
  expect(parentGraph.nodes).toContain(subgraphNode)
  expect(subgraphNode.graph).toBe(parentGraph)
  expect(subgraphNode.subgraph).toBe(subgraph)
})
```

## Configuration Options

### `createTestSubgraph(options)`

```typescript
interface TestSubgraphOptions {
  id?: UUID // Custom UUID
  name?: string // Custom name
  nodeCount?: number // Number of internal nodes
  inputCount?: number // Number of inputs (uses generic types)
  outputCount?: number // Number of outputs (uses generic types)
  inputs?: Array<{
    // Specific input definitions
    name: string
    type: ISlotType
  }>
  outputs?: Array<{
    // Specific output definitions
    name: string
    type: ISlotType
  }>
}
```

**Note**: Cannot specify both `inputs` array and `inputCount` (or `outputs` array and `outputCount`) - the function will throw an error with details.

### `createNestedSubgraphs(options)`

```typescript
interface NestedSubgraphOptions {
  depth?: number // Nesting depth (default: 2)
  nodesPerLevel?: number // Nodes per subgraph (default: 2)
  inputsPerSubgraph?: number // Inputs per subgraph (default: 1)
  outputsPerSubgraph?: number // Outputs per subgraph (default: 1)
}
```

## Important Architecture Notes

### Subgraph vs SubgraphNode

- **Subgraph**: The definition/template (like a class definition)
- **SubgraphNode**: An instance of a subgraph placed in a graph (like a class instance)
- One Subgraph can have many SubgraphNode instances

### Special Node IDs

- Input node always has ID `-10` (SUBGRAPH_INPUT_ID)
- Output node always has ID `-20` (SUBGRAPH_OUTPUT_ID)
- These are virtual nodes that exist in every subgraph

### Common Pitfalls

1. **Array vs Index**: The `inputs` and `outputs` arrays don't have an `index` property on items. Use `indexOf()`:

   ```typescript
   // ❌ Wrong
   expect(input.index).toBe(0)

   // ✅ Correct
   expect(subgraph.inputs.indexOf(input)).toBe(0)
   ```

2. **Graph vs Subgraph Property**: SubgraphInputNode/OutputNode have `subgraph`, not `graph`:

   ```typescript
   // ❌ Wrong
   expect(inputNode.graph).toBe(subgraph)

   // ✅ Correct
   expect(inputNode.subgraph).toBe(subgraph)
   ```

3. **Event Detail Structure**: Events have specific detail structures:

   ```typescript
   // Input events
   "adding-input": { name: string, type: string }
   "input-added": { input: SubgraphInput, index: number }

   // Output events
   "adding-output": { name: string, type: string }
   "output-added": { output: SubgraphOutput, index: number }
   ```

4. **Links are stored in a Map**: Use `.size` not `.length`:

   ```typescript
   // ❌ Wrong
   expect(subgraph.links.length).toBe(1)

   // ✅ Correct
   expect(subgraph.links.size).toBe(1)
   ```

## Testing Best Practices

- Always use helper functions instead of manual setup
- Use fixtures for common scenarios to avoid repetitive code
- Clean up event listeners with `capture.cleanup()` after event tests
- Use `verifyEventSequence()` to test event ordering
- Remember fixtures are created fresh for each test (no shared state)
- Use `assertSubgraphStructure()` for comprehensive validation

## Debugging Tips

- Use `logSubgraphStructure(subgraph)` to print subgraph details
- Check `subgraph.rootGraph` to verify graph hierarchy
- Event capture includes timestamps for debugging timing issues
- All factory functions accept optional parameters for customization

## Adding New Test Utilities

When extending the test infrastructure:

1. Add new helper functions to `subgraphHelpers.ts`
2. Add new fixtures to `subgraphFixtures.ts`
3. Update this README with usage examples
4. Follow existing patterns for consistency
5. Add TypeScript types for all parameters

## Performance Notes

- Helper functions are optimized for test clarity, not performance
- Use `structuredClone()` for deep copying test data
- Event capture systems automatically clean up listeners
- Fixtures are created fresh for each test to avoid state contamination
